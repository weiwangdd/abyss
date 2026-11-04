#!/usr/bin/env python3
"""Where a GGUF of the model is not at the checkpoint's own precision. Every tensor of the main GGUF is grouped by how
v41convert.py made it from the DeepSeek-V4.1-Flash checkpoint, with its size and where the server keeps it; for the
kinds that round a second time (FP8 -> Q8_0, FP8 Engram -> MXFP4, and a BF16 or F32 tensor given another type, as
retype_tensor.py does) sampled rows are dequantized and compared with the checkpoint's own values, which are what the
official inference/model.py computes with (FP8 times its E8M0 scale, exact in BF16). With --acts, the activations
dumped by expert-trace.sh (router inputs) are quantized both as the fork does before its int8 matmuls (32 per block,
an FP16 scale) and as the official act_quant does before its FP8 and FP4 GEMMs (E4M3, 32 per block, a power-of-two
scale), and pushed through a shared expert's gate. Reads only; GGUF reads go through the page cache like the
server's, checkpoint reads are dropped from it.

    precision_audit.py GGUF_DIR CHECKPOINT_DIR [--rows N] [--seed S] [--acts TRACE_DIR]
"""

import argparse
import collections
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "convert"))
import v41convert as cv  # noqa: E402
import v41io  # noqa: E402

TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 30: "BF16", 39: "MXFP4"}
KV_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
CPU_EXPERTS = re.compile(r"blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.")
ACT_WEIGHT = "layers.30.ffn.shared_experts.w1"


def where(name):
    if CPU_EXPERTS.match(name) or name == "token_embd.weight":
        return "DRAM"
    if ".engram_embed." in name:
        return "NVMe mmap"
    return "VRAM"


def deq_q8_0(buf, rows, cols):
    b = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols // 32, 34)
    d = b[..., :2].copy().view(np.float16).astype(np.float32)
    q = b[..., 2:].copy().view(np.int8).astype(np.float32)
    return (d * q).reshape(rows, cols)


def deq_mxfp4(buf, rows, cols):
    """ggml block_mxfp4: an E8M0 byte, then elements j and j + 16 in the low and high nibble of byte j; the table holds
    twice the e2m1 values, so the scale is 2^(e - 128)."""
    b = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols // 32, 17)
    scale = np.ldexp(np.float32(1), b[..., 0].astype(np.int32) - 128).astype(np.float32)
    qs = b[..., 1:]
    vals = np.concatenate([KV_MXFP4[qs & 0x0F], KV_MXFP4[qs >> 4]], axis=-1)
    return (vals * scale[..., None]).reshape(rows, cols)


def act_int8(x):
    """The fork's activation quantization before its int8 matmuls (q8_1 on the GPU, Q8_0 on the CPU): 32 per block,
    q = round(x / d) with d = absmax / 127, d stored as FP16."""
    b = x.reshape(x.shape[0], -1, 32)
    d = np.abs(b).max(-1, keepdims=True) / 127
    q = np.round(b / np.where(d > 0, d, 1))
    return (q * d.astype(np.float16).astype(np.float32)).reshape(x.shape)


def act_fp8(x):
    """inference/kernel.py act_quant with scale_fmt ue8m0 and block 32: s = 2^ceil(log2(max(absmax, 1e-4) / 448)),
    x / s to E4M3."""
    t = torch.from_numpy(x).reshape(x.shape[0], -1, 32)
    amax = t.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return ((t / s).to(torch.float8_e4m3fn).to(torch.float32) * s).reshape(x.shape).numpy()


class Gguf:
    def __init__(self, gguf_dir):
        self.where = {}
        for path in sorted(glob.glob(os.path.join(gguf_dir, "*.gguf"))):
            if "DSpark" in os.path.basename(path):
                continue
            g = v41io.read_gguf(path)
            for t in g["tensors"]:
                self.where[t["name"]] = (path, g["data_start"] + t["offset"], t)
        self.fds = {}

    def rows(self, name, r0, n):
        path, base, t = self.where[name]
        if path not in self.fds:
            self.fds[path] = os.open(path, os.O_RDONLY)
        fd = self.fds[path]
        row_bytes = v41io.tensor_nbytes([t["dims"][0]], t["type"])
        data = os.pread(fd, n * row_bytes, base + r0 * row_bytes)
        assert len(data) == n * row_bytes
        deq = deq_q8_0 if t["type"] == cv.Q8_0 else deq_mxfp4
        return deq(data, n, t["dims"][0])


def errors(ref, got):
    """Relative L2 error over everything, the worst row's, the largest error as a share of its 32-block's absmax, and
    the share of the checkpoint's nonzero values that came out zero."""
    d = got - ref
    rel = np.linalg.norm(d) / np.linalg.norm(ref)
    row = np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
    blocks = ref.reshape(ref.shape[0], -1, 32)
    amax = np.abs(blocks).max(axis=-1, keepdims=True)
    worst = (np.abs(d.reshape(blocks.shape)) / np.maximum(amax, 1e-30)).max()
    nz = ref != 0
    lost = ((got == 0) & nz).sum() / max(nz.sum(), 1)
    return rel, row.max(), worst, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf_dir")
    ap.add_argument("checkpoint")
    ap.add_argument("--rows", type=int, default=4096, help="Engram rows per table, and FP8 weight rows per tensor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--acts", help="an expert-trace.sh output directory with ffn_norm-*.bin dumps")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    gg = Gguf(args.gguf_dir)
    ck = v41io.Checkpoint(args.checkpoint)

    # what each tensor is, by how it was made
    census = collections.defaultdict(lambda: [0, 0, set()])
    for name, (_, _, t) in gg.where.items():
        kind, _ = cv.rule_for(name, False)
        m = re.match(r"blk\.\d+\.(.+)", name)
        key = (kind, TYPE_NAMES.get(t["type"], str(t["type"])), where(name))
        census[key][0] += 1
        census[key][1] += t["nbytes"]
        census[key][2].add(m.group(1) if m else name)
    print("kind       GGUF   where      tensors      GiB  names")
    for (kind, typ, loc), (n, nbytes, names) in sorted(census.items(), key=lambda kv: -kv[1][1]):
        print(f"{kind:10s} {typ:6s} {loc:10s} {n:7d} {nbytes / 2**30:8.2f}  {', '.join(sorted(names))}")

    # FP8 weights -> Q8_0: a block of rows of each kind in a few layers
    print("\nFP8 weight -> Q8_0, against the checkpoint's FP8 x E8M0 (relative L2, worst row, worst |err| / block absmax,"
          " nonzero -> 0)")
    by_kind = collections.defaultdict(list)
    for name, (_, _, t) in gg.where.items():
        kind, src = cv.rule_for(name, False)
        if kind == "fp8_q8":
            by_kind[re.sub(r"blk\.\d+\.", "", name)].append((name, src, t))
    for key in sorted(by_kind):
        items = sorted(by_kind[key], key=lambda x: int(x[0].split(".")[1]))
        picks = [items[0], items[len(items) // 2], items[-1]] if len(items) >= 3 else items
        res = []
        for name, src, t in picks:
            _, (rows, cols), _ = ck.info(src + ".weight")
            _, (srows, scols), _ = ck.info(src + ".scale")
            br = rows // srows
            n = min(args.rows // 16, rows) // br * br
            r0 = int(rng.integers(0, (rows - n) // br + 1)) * br
            w = ck.read(src + ".weight", r0 * cols, n * cols)
            s = ck.read(src + ".scale", r0 // br * scols, n // br * scols)
            ref = cv.dequant_fp8(w, n, cols, s, n // br, scols)
            got = gg.rows(name, r0, n)
            res.append(errors(ref, got))
        rel, row, worst, lost = (max(r[i] for r in res) if i else float(np.mean([r[0] for r in res])) for i in range(4))
        layers = ",".join(p[0].split(".")[1] for p in picks)
        print(f"  {key:32s} layers {layers:9s} rel {rel * 100:6.3f}%  worst row {row * 100:6.3f}%  "
              f"worst {worst * 100:5.3f}% of absmax  lost {lost * 100:5.2f}%")

    # Engram tables, FP8 with one E8M0 per 32 -> MXFP4: random rows
    print("\nEngram FP8 -> MXFP4, random rows against the checkpoint")
    for name in sorted(n for n in gg.where if ".engram_embed." in n):
        _, src = cv.rule_for(name, False)
        _, (rows, cols), _ = ck.info(src + ".weight")
        _, (srows, scols), _ = ck.info(src + ".scale")
        idx = np.sort(rng.choice(rows, size=args.rows, replace=False))
        ref = np.empty((len(idx), cols), np.float32)
        got = np.empty_like(ref)
        for i, r in enumerate(idx):
            w = ck.read(src + ".weight", int(r) * cols, cols)
            s = ck.read(src + ".scale", int(r) * scols, scols)
            ref[i] = cv.dequant_fp8(w, 1, cols, s, 1, scols)
            got[i] = gg.rows(name, int(r), 1)
        rel, row, worst, lost = errors(ref, got)
        exact = np.mean(np.all(ref == got, axis=1))
        print(f"  {name:28s} {len(idx)} rows: rel {rel * 100:6.2f}%  worst row {row * 100:6.2f}%  "
              f"worst {worst * 100:5.2f}% of absmax  nonzero -> 0 {lost * 100:5.2f}%  rows exact {exact * 100:4.1f}%")

    # tensors the checkpoint has in BF16 or F32 that this GGUF holds in another type: random rows
    retyped = sorted(n for n, (_, _, t) in gg.where.items()
                     if cv.rule_for(n, False)[0] == "copy" and t["type"] not in (cv.BF16, cv.F32))
    if retyped:
        print("\nBF16 / F32 -> another type, random rows against the checkpoint")
    for name in retyped:
        _, src = cv.rule_for(name, False)
        dtype, shape, _ = ck.info(src)
        rows, cols = shape[0], int(np.prod(shape[1:]))
        esize = {"BF16": 2, "F32": 4}[dtype]
        idx = np.sort(rng.choice(rows, size=min(args.rows, rows), replace=False))
        ref = np.empty((len(idx), cols), np.float32)
        got = np.empty_like(ref)
        for i, r in enumerate(idx):
            b = ck.read(src, int(r) * cols * esize, cols * esize)
            ref[i] = (torch.frombuffer(bytearray(b), dtype=torch.bfloat16).float().numpy() if dtype == "BF16"
                      else np.frombuffer(b, dtype=np.float32))
            got[i] = gg.rows(name, int(r), 1)
        rel, row, worst, lost = errors(ref, got)
        print(f"  {name:28s} {dtype} -> {TYPE_NAMES[gg.where[name][2]['type']]}, {len(idx)} rows: rel {rel * 100:6.3f}%  "
              f"worst row {row * 100:6.3f}%  worst {worst * 100:5.3f}% of absmax  nonzero -> 0 {lost * 100:5.2f}%")

    # activations: the fork's int8 against the official FP8, on dumped router inputs and through a shared expert's gate
    if args.acts:
        paths = sorted(glob.glob(os.path.join(args.acts, "ffn_norm-*.bin")))
        xs = []
        for i in rng.choice(len(paths), size=min(600, len(paths)), replace=False):
            a = np.fromfile(paths[i], dtype=np.float32, offset=40)  # after the dump's header: magic, type, ne[4]
            if a.size and a.size % 5120 == 0:
                xs.append(a.reshape(-1, 5120)[-1])
        x = np.stack(xs)
        _, (rows, cols), _ = ck.info(ACT_WEIGHT + ".weight")
        _, (srows, scols), _ = ck.info(ACT_WEIGHT + ".scale")
        w = cv.dequant_fp8(ck.read(ACT_WEIGHT + ".weight"), rows, cols, ck.read(ACT_WEIGHT + ".scale"), srows, scols)
        ref = x @ w.T
        print(f"\nactivations, {len(x)} dumped router inputs, and through {ACT_WEIGHT}")
        for label, f in (("int8 per 32, FP16 scale (fork)", act_int8), ("FP8 e4m3 per 32, ue8m0 (official)", act_fp8)):
            xq = f(x)
            e_x = np.linalg.norm(xq - x) / np.linalg.norm(x)
            e_y = np.linalg.norm(xq @ w.T - ref) / np.linalg.norm(ref)
            print(f"  {label:34s} activation rel L2 {e_x * 100:.3f}%   gate output rel L2 {e_y * 100:.3f}%")


if __name__ == "__main__":
    main()
