#!/usr/bin/env python3
"""Make a copy of a split GGUF model in which one tensor has another ggml type (now Q8_0, from BF16, F16 or F32). The
shard holding the tensor is copied with direct I/O, so neither side of the copy goes through the page cache and a
running server keeps its hot pages. In the copy the tensor's info gets the new type, its data is replaced in place, and
the tensors after it in the shard move by the change in size: the GGUF reader wants every offset to be the padded end
of the tensor before it, in header order. The header keeps its size, so the tensors before it stay where they were. The
other files of the model are symlinked. The quantization is ggml_quantize_chunk of the production build
(v41convert.quantize), so it rounds as the fork does.

    retype_tensor.py SRC_DIR DST_DIR TENSOR TYPE
"""

import argparse
import glob
import os
import random
import struct
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v41convert as cv  # noqa: E402
import v41io  # noqa: E402

TYPES = {"Q8_0": cv.Q8_0}
SOURCE = {cv.F32: torch.float32, 1: torch.float16, cv.BF16: torch.bfloat16}
ROWS = 4096
PIECE = 64 << 20


def info_fields(path):
    """File position of each tensor info's type (a u32, followed by the tensor's u64 offset), by tensor name."""
    pos = {}
    with open(path, "rb") as f:
        r = v41io._Reader(f)
        assert f.read(4) == b"GGUF", path
        r.get("I")
        n_tensors, n_kv = r.get("Q"), r.get("Q")
        for _ in range(n_kv):
            r.string()
            r.value(r.get("I"))
        for _ in range(n_tensors):
            name = r.string()
            for _ in range(r.get("I")):
                r.get("Q")
            pos[name] = f.tell()
            r.get("I")
            r.get("Q")
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("tensor")
    ap.add_argument("type", choices=sorted(TYPES))
    args = ap.parse_args()
    new_type = TYPES[args.type]

    files = sorted(glob.glob(os.path.join(args.src, "*.gguf")))
    holder = None
    for path in files:
        g = v41io.read_gguf(path)
        if any(t["name"] == args.tensor for t in g["tensors"]):
            holder = g
    assert holder, f"{args.tensor} is in no GGUF of {args.src}"
    order = holder["tensors"]
    i = next(k for k, t in enumerate(order) if t["name"] == args.tensor)
    info = order[i]
    assert info["type"] in SOURCE, (args.tensor, v41io.GGUF_TYPE_NAMES.get(info["type"], info["type"]))
    cols = info["dims"][0]
    rows = info["nbytes"] // v41io.tensor_nbytes([cols], info["type"])
    src_row, dst_row = v41io.tensor_nbytes([cols], info["type"]), v41io.tensor_nbytes([cols], new_type)
    new_bytes = rows * dst_row

    # the new layout from the tensor on: each offset the padded end of the one before
    pad = lambda n: (n + holder["alignment"] - 1) // holder["alignment"] * holder["alignment"]
    base = holder["data_start"]
    new_off, off = {}, info["offset"]
    for t in order[i:]:
        new_off[t["name"]] = off
        off += pad(new_bytes if t is info else t["nbytes"])
    new_size = base + off

    os.makedirs(args.dst, exist_ok=True)
    for path in files:
        link = os.path.join(args.dst, os.path.basename(path))
        if path != holder["path"] and not os.path.lexists(link):
            os.symlink(os.path.realpath(path), link)
    out = os.path.join(args.dst, os.path.basename(holder["path"]))
    tmp = out + ".tmp"
    subprocess.run(["dd", f"if={holder['path']}", f"of={tmp}", "bs=64M", "iflag=direct", "oflag=direct", "status=none"],
                   check=True)
    assert os.path.getsize(tmp) == holder["size"], tmp

    # everything after the tensor's start is written from the source, so moving down over old bytes is safe
    src = os.open(holder["path"], os.O_RDONLY)
    fd = os.open(tmp, os.O_RDWR)
    for r0 in range(0, rows, ROWS):
        n = min(ROWS, rows - r0)
        b = os.pread(src, n * src_row, base + info["offset"] + r0 * src_row)
        assert len(b) == n * src_row
        x = torch.frombuffer(bytearray(b), dtype=SOURCE[info["type"]]).to(torch.float32).reshape(n, cols).numpy()
        q = cv.quantize(x, new_type)
        assert len(q) == n * dst_row
        os.pwrite(fd, q, base + new_off[args.tensor] + r0 * dst_row)
    os.pwrite(fd, bytes(pad(new_bytes) - new_bytes), base + new_off[args.tensor] + new_bytes)
    for t in order[i + 1:]:
        for p in range(0, t["nbytes"], PIECE):
            n = min(PIECE, t["nbytes"] - p)
            os.pwrite(fd, os.pread(src, n, base + t["offset"] + p), base + new_off[t["name"]] + p)
        os.pwrite(fd, bytes(pad(t["nbytes"]) - t["nbytes"]), base + new_off[t["name"]] + t["nbytes"])
    os.ftruncate(fd, new_size)
    pos = info_fields(tmp)
    for t in order[i:]:
        os.pwrite(fd, struct.pack("<IQ", new_type if t is info else t["type"], new_off[t["name"]]), pos[t["name"]])
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

    # the header reads back as the source's but for the one type and the moved offsets, the offsets are what the GGUF
    # reader checks, and sampled bytes of every other tensor are the source's
    g = v41io.read_gguf(tmp)
    assert g["kv"] == holder["kv"] and g["data_start"] == base and g["size"] == new_size
    off = 0
    for t, s in zip(g["tensors"], order):
        assert (t["name"], t["dims"]) == (s["name"], s["dims"]) and t["offset"] == off, t["name"]
        assert t["type"] == (new_type if s is info else s["type"]), t["name"]
        off += pad(t["nbytes"])
    assert base + off == new_size
    moved = {t["name"]: t["offset"] for t in g["tensors"]}
    rng = random.Random(0)
    fd = os.open(tmp, os.O_RDONLY)
    for s in rng.sample([s for s in order if s is not info], min(32, len(order) - 1)):
        p = rng.randrange(max(s["nbytes"] - (1 << 20), 0) + 1)
        n = min(1 << 20, s["nbytes"] - p)
        assert os.pread(src, n, base + s["offset"] + p) == os.pread(fd, n, base + moved[s["name"]] + p), s["name"]
    for s in order[i + 1:]:  # the moved ones in full
        for p in range(0, s["nbytes"], PIECE):
            n = min(PIECE, s["nbytes"] - p)
            assert os.pread(src, n, base + s["offset"] + p) == os.pread(fd, n, base + moved[s["name"]] + p), s["name"]
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    os.close(src)
    os.rename(tmp, out)
    print(f"{out}: {args.tensor} {v41io.GGUF_TYPE_NAMES[info['type']]} -> {args.type}, {rows} rows, "
          f"{info['nbytes'] / 2**30:.3f} -> {new_bytes / 2**30:.3f} GiB, {len(order) - i - 1} tensors after it moved")


if __name__ == "__main__":
    main()
