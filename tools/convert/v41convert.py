#!/usr/bin/env python3
"""Convert a DeepSeek-V4.1-Flash Hugging Face checkpoint into GGUF by following an existing mx-llama.cpp conversion
of the same architecture (the template): every output file takes its template's header as is, so the key/values,
tensor names, types, shapes and offsets stay the same, and each tensor's data is converted from the checkpoint the
way the template's was:

  routed experts    FP4, two per byte, E8M0 scale per 32   -> MXFP4, a repack without rounding
  FP8 weights       E4M3, E8M0 scale per 32 x 32 block     -> dequantized, ggml Q8_0
  Engram tables     E4M3, E8M0 scale per 32                -> dequantized, ggml MXFP4
  norms             BF16                                    -> F32
  everything else   BF16 or F32                             -> copied

    v41convert.py verify  TEMPLATE_DIR REPO [REVISION]
        checks that every template tensor has its sources in the checkpoint, then converts samples of every tensor
        kind from the checkpoint the template was made from, read remotely, and compares them with the template bytes
    v41convert.py convert TEMPLATE_DIR CHECKPOINT_DIR OUT_DIR [--jobs N] [--rename OLD=NEW] [--skip-missing]
        converts a local checkpoint; a journal in OUT_DIR lets an interrupted run continue, and --skip-missing
        leaves out the tensors whose shards are not downloaded yet, for a later run to fill in
    v41convert.py check   TEMPLATE_DIR CHECKPOINT_DIR OUT_DIR [--rename OLD=NEW] [--samples N]
        compares each output file's header and size with its template, and converts random pieces again to
        compare them with what was written

Q8_0 and MXFP4 come from the ggml library of a build (GGML_LIB, default the production build), so they round
exactly as the fork does. Reads and writes are dropped from the page cache, and workers run at nice 19, so a
conversion next to a running server does not push its hot pages (the Engram tables) out of memory.
"""

import argparse
import ctypes
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v41io  # noqa: E402

F32, BF16, Q8_0, MXFP4 = 0, 30, 8, 39
GGML_LIB = os.environ.get("GGML_LIB", os.path.expanduser("~/src/mx-llama.cpp/build-m2/bin/libggml-base.so"))
COPY_PIECE = 64 << 20
ROWS_PIECE = {"fp8_q8": 4096, "fp8_mxfp4": 1 << 19}

# GGUF tensor name after "blk.N." -> how it is made, and from what under "layers.N." (or "mtp.N." in the DSpark file)
LAYER_RULES = {
    "ffn_gate_exps.weight": ("experts", "ffn.experts.{e}.w1"),
    "ffn_up_exps.weight": ("experts", "ffn.experts.{e}.w3"),
    "ffn_down_exps.weight": ("experts", "ffn.experts.{e}.w2"),
    "attn_sinks.weight": ("copy", "attn.attn_sink"),
    "attn_kv_a_norm.weight": ("bf16_f32", "attn.kv_norm.weight"),
    "attn_q_a_norm.weight": ("bf16_f32", "attn.q_norm.weight"),
    "attn_kv.weight": ("fp8_q8", "attn.wkv"),
    "attn_output_a.weight": ("fp8_q8", "attn.wo_a"),
    "attn_output_b.weight": ("fp8_q8", "attn.wo_b"),
    "attn_q_a.weight": ("fp8_q8", "attn.wq_a"),
    "attn_q_b.weight": ("fp8_q8", "attn.wq_b"),
    "attn_norm.weight": ("bf16_f32", "attn_norm.weight"),
    "exp_probs_b.bias": ("copy", "ffn.gate.bias"),
    "ffn_gate_inp.weight": ("copy", "ffn.gate.weight"),
    "ffn_gate_shexp.weight": ("fp8_q8", "ffn.shared_experts.w1"),
    "ffn_down_shexp.weight": ("fp8_q8", "ffn.shared_experts.w2"),
    "ffn_up_shexp.weight": ("fp8_q8", "ffn.shared_experts.w3"),
    "ffn_norm.weight": ("bf16_f32", "ffn_norm.weight"),
    "hc_attn_base.weight": ("copy", "hc_attn_base"),
    "hc_attn_fn.weight": ("copy", "hc_attn_fn"),
    "hc_attn_scale.weight": ("copy", "hc_attn_scale"),
    "hc_ffn_base.weight": ("copy", "hc_ffn_base"),
    "hc_ffn_fn.weight": ("copy", "hc_ffn_fn"),
    "hc_ffn_scale.weight": ("copy", "hc_ffn_scale"),
    "attn_compressor_norm.weight": ("bf16_f32", "attn.compressor.norm.weight"),
    "attn_compressor_gate.weight": ("copy", "attn.compressor.wgate.weight"),
    "attn_compressor_kv.weight": ("copy", "attn.compressor.wkv.weight"),
    "indexer_compressor_norm.weight": ("bf16_f32", "attn.indexer.k_norm.weight"),
    "indexer.proj.weight": ("copy", "attn.indexer.weights_proj.weight"),
    "indexer_compressor_kv.weight": ("copy", "attn.indexer.wk.weight"),
    "indexer.attn_q_b.weight": ("fp8_q8", "attn.indexer.wq_b"),
    "engram_embed.weight": ("fp8_mxfp4", "engram.embed"),
    "engram_k.weight": ("copy", "engram.k_weight"),
    "engram_q.weight": ("copy", "engram.q_weight"),
    "engram_wkv.weight": ("fp8_q8", "engram.wkv"),
}
MAIN_RULES = {
    "token_embd.weight": ("copy", "embed.weight"),
    "output.weight": ("copy", "head.weight"),
    "output_norm.weight": ("bf16_f32", "norm.weight"),
}
# the DSpark sidecar: projection and its norm from stage 0, the markov and confidence heads and the final norm from
# stage 2 (as the template's README describes)
DSPARK_RULES = {
    "enc.output_norm.weight": ("bf16_f32", "mtp.0.main_norm.weight"),
    "fc.weight": ("fp8_q8", "mtp.0.main_proj"),
    "conf_proj.weight": ("copy", "mtp.2.confidence_head.proj.weight"),
    "markov_w1.weight": ("copy", "mtp.2.markov_head.embed.weight"),
    "markov_w2.weight": ("copy", "mtp.2.markov_head.head.weight"),
    "output_norm.weight": ("bf16_f32", "mtp.2.norm.weight"),
}


def rule_for(name, dspark):
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if m:
        kind, src = LAYER_RULES[m.group(2)]
        return kind, ("mtp." if dspark else "layers.") + m.group(1) + "." + src
    return (DSPARK_RULES if dspark else MAIN_RULES)[name]


def sources(kind, src, n_expert=0):
    """Checkpoint tensor names a rule reads."""
    if kind in ("copy", "bf16_f32"):
        return [src]
    if kind == "experts":
        return [src.format(e=e) + s for e in range(n_expert) for s in (".weight", ".scale")]
    return [src + ".weight", src + ".scale"]


def plan(ck, tensor, kind, src):
    """The pieces of one GGUF tensor: (offset in the tensor, expected size, spec)."""
    dims, nbytes = tensor["dims"], tensor["nbytes"]
    if kind == "copy":
        dtype, shape, n = ck.info(src)
        assert n == nbytes and v41io.GGUF_TYPE_NAMES[tensor["type"]] == {"BF16": "BF16", "F32": "F32"}[dtype], (tensor["name"], src)
        return [(off, min(COPY_PIECE, n - off), {"kind": kind, "src": src, "start": off, "size": min(COPY_PIECE, n - off)})
                for off in range(0, n, COPY_PIECE)]
    if kind == "bf16_f32":
        dtype, shape, n = ck.info(src)
        assert dtype == "BF16" and tensor["type"] == F32 and n * 2 == nbytes, (tensor["name"], src)
        return [(0, nbytes, {"kind": kind, "src": src})]
    if kind == "experts":
        n_expert = dims[2]
        per = nbytes // n_expert
        dtype, (rows, half), _ = ck.info(src.format(e=0) + ".weight")
        assert dtype == "I8" and [half * 2, rows] == dims[:2] and per == rows * half * 2 // 32 * 17, (tensor["name"], src)
        return [(e * per, per, {"kind": kind, "src": src, "expert": e}) for e in range(n_expert)]
    # fp8_q8 / fp8_mxfp4
    dtype, (rows, cols), _ = ck.info(src + ".weight")
    sdtype, (srows, scols), _ = ck.info(src + ".scale")
    assert dtype == "F8_E4M3" and sdtype == "F8_E8M0" and dims == [cols, rows], (tensor["name"], src)
    out_type = Q8_0 if kind == "fp8_q8" else MXFP4
    assert tensor["type"] == out_type
    row_bytes = v41io.tensor_nbytes([cols], out_type)
    step = max(ROWS_PIECE[kind] // (rows // srows), 1) * (rows // srows)
    return [(r0 * row_bytes, (min(r0 + step, rows) - r0) * row_bytes,
             {"kind": kind, "src": src, "rows": [r0, min(r0 + step, rows)]}) for r0 in range(0, rows, step)]


_lib = None


def ggml():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(GGML_LIB)
        _lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        _lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                             ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        _lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    return _lib


def quantize(x, ggml_type):
    rows, n = x.shape
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty(v41io.tensor_nbytes([n, rows], ggml_type), dtype=np.uint8)
    lib = ggml()
    lib.ggml_quantize_init(ggml_type)
    got = lib.ggml_quantize_chunk(ggml_type, x.ctypes.data, out.ctypes.data, 0, rows, n, None)
    assert got == out.nbytes, (got, out.nbytes)
    return out.tobytes()


def dequant_fp8(w, rows, cols, s, srows, scols):
    """E4M3 [rows, cols] times its E8M0 block scales [srows, scols], as float32."""
    br, bc = rows // srows, cols // scols
    x = torch.frombuffer(bytearray(w), dtype=torch.float8_e4m3fn).reshape(rows, cols).to(torch.float32)
    e = np.frombuffer(s, dtype=np.uint8).reshape(srows, scols).astype(np.int32)
    scale = torch.from_numpy(np.ldexp(np.float32(1), e - 127).astype(np.float32))
    x = x.reshape(srows, br, scols, bc).mul_(scale[:, None, :, None]).reshape(rows, cols)
    return x.numpy()


def repack_fp4(w, rows, cols, s, low_first=True):
    """FP4 codes packed two per byte, element 2k in the low nibble, with one E8M0 per 32 -> ggml block_mxfp4, which keeps
    the scale byte and puts elements j and j + 16 of a block into the low and high nibble of byte j."""
    nb = cols // 32
    packed = np.frombuffer(w, dtype=np.uint8).reshape(rows, nb, 16)
    lo, hi = packed & 0x0F, packed >> 4
    if not low_first:
        lo, hi = hi, lo
    codes = np.empty((rows, nb, 32), dtype=np.uint8)
    codes[..., 0::2] = lo
    codes[..., 1::2] = hi
    out = np.empty((rows, nb, 17), dtype=np.uint8)
    out[..., 0] = np.frombuffer(s, dtype=np.uint8).reshape(rows, nb)
    out[..., 1:] = codes[..., :16] | (codes[..., 16:] << 4)
    return out.tobytes()


def run(ck, spec, low_first=True):
    """The bytes of one piece."""
    kind, src = spec["kind"], spec["src"]
    if kind == "copy":
        return ck.read(src, spec["start"], spec["size"])
    if kind == "bf16_f32":
        return torch.frombuffer(bytearray(ck.read(src)), dtype=torch.bfloat16).to(torch.float32).numpy().tobytes()
    if kind == "experts":
        name = src.format(e=spec["expert"])
        _, (rows, half), _ = ck.info(name + ".weight")
        return repack_fp4(ck.read(name + ".weight"), rows, half * 2, ck.read(name + ".scale"), low_first)
    _, (rows, cols), _ = ck.info(src + ".weight")
    _, (srows, scols), _ = ck.info(src + ".scale")
    r0, r1 = spec["rows"]
    br = rows // srows
    w = ck.read(src + ".weight", r0 * cols, (r1 - r0) * cols)
    s = ck.read(src + ".scale", r0 // br * scols, (r1 - r0) // br * scols)
    x = dequant_fp8(w, r1 - r0, cols, s, (r1 - r0) // br, scols)
    return quantize(x, Q8_0 if kind == "fp8_q8" else MXFP4)


def template_files(template_dir):
    files = sorted(glob.glob(os.path.join(template_dir, "*.gguf")))
    return [(path, v41io.read_gguf(path), "DSpark" in os.path.basename(path)) for path in files]


# ---------------------------------------------------------------------------------------------------------------- verify

def verify(template_dir, repo, revision):
    ck = v41io.Checkpoint(repo, revision)
    files = template_files(template_dir)

    # coverage: every template tensor maps to checkpoint tensors that exist, with matching sizes
    n = 0
    used = set()
    for path, g, dspark in files:
        for t in g["tensors"]:
            kind, src = rule_for(t["name"], dspark)
            names = sources(kind, src, t["dims"][2] if kind == "experts" else 0)
            missing = [x for x in names if x not in ck.weight_map]
            assert not missing, (t["name"], missing[:3])
            plan(ck, t, kind, src)
            used.update(names)
            n += 1
    unused = sorted(x for x in ck.weight_map if x not in used)
    print(f"coverage: {n} template tensors map to {len(used)} checkpoint tensors; unused {len(unused)}:",
          sorted({re.sub(r"(?<=\.)\d+(?=\.)", "N", x) for x in unused}))

    # samples: for each tensor pattern a few instances, and of each a few pieces, compared byte for byte
    by_pattern = {}
    for path, g, dspark in files:
        for t in g["tensors"]:
            key = ("dspark:" if dspark else "") + re.sub(r"^blk\.\d+\.", "blk.N.", t["name"])
            by_pattern.setdefault(key, []).append((path, g, dspark, t))
    failures = 0
    for key, items in by_pattern.items():
        picks = sorted({0, len(items) // 2, len(items) - 1})
        results = []
        for i in picks:
            path, g, dspark, t = items[i]
            kind, src = rule_for(t["name"], dspark)
            pieces = plan(ck, t, kind, src)
            for j in sorted({0, len(pieces) // 2, len(pieces) - 1}):
                off, size, spec = pieces[j]
                got = run(ck, spec)
                with open(path, "rb") as f:
                    f.seek(g["data_start"] + t["offset"] + off)
                    want = f.read(size)
                same = got == want
                note = ""
                if not same and kind == "experts" and run(ck, spec, low_first=False) == want:
                    note = " (matches with the high nibble first)"
                if not same:
                    a, b = np.frombuffer(got, np.uint8), np.frombuffer(want, np.uint8)
                    diff = np.flatnonzero(a != b) if len(a) == len(b) else []
                    note += f" lengths {len(got)}/{len(want)}, {len(diff)} bytes differ, first at {diff[0] if len(diff) else '-'}"
                results.append((t["name"], j, same, note))
        bad = [r for r in results if not r[2]]
        failures += len(bad)
        print(f"{'OK  ' if not bad else 'FAIL'} {key:45s} {kind:9s} {len(results)} pieces" +
              ("" if not bad else "  " + "; ".join(f"{n}[{j}]{note}" for n, j, _, note in bad[:2])))
    print("all samples identical" if not failures else f"{failures} pieces differ")
    return 1 if failures else 0


# --------------------------------------------------------------------------------------------------------------- convert

_ck = None


def _init(checkpoint_dir):
    global _ck
    os.nice(19)
    torch.set_num_threads(1)
    _ck = v41io.Checkpoint(checkpoint_dir)


def _work(task):
    out_path, offset, size, spec = task
    t0 = time.time()
    data = run(_ck, spec)
    assert len(data) == size, (out_path, offset, len(data), size)
    fd = os.open(out_path, os.O_WRONLY)
    try:
        done = 0
        while done < size:
            done += os.pwrite(fd, data[done:], offset + done)
        os.fdatasync(fd)
        os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return out_path, offset, size, time.time() - t0


def out_name(path, renames):
    name = os.path.basename(path)
    for old, new in renames:
        name = name.replace(old, new)
    return name


def all_tasks(ck, template_dir, out_dir, renames, skip_missing=False):
    """Every piece of every output file: (output path, file offset, size, spec); with skip_missing, without the
    tensors whose checkpoint shards are not there yet (and how many tensors that leaves out)."""
    tasks, skipped = [], 0
    for path, g, dspark in template_files(template_dir):
        out_path = os.path.join(out_dir, out_name(path, renames))
        for t in g["tensors"]:
            kind, src = rule_for(t["name"], dspark)
            if skip_missing:
                files = {ck.weight_map[x] for x in sources(kind, src, t["dims"][2] if kind == "experts" else 0)}
                if not all(os.path.exists(os.path.join(ck.root, f)) for f in files):
                    skipped += 1
                    continue
            for off, size, spec in plan(ck, t, kind, src):
                tasks.append((out_path, g["data_start"] + t["offset"] + off, size, spec))
    return tasks, skipped


def convert(template_dir, checkpoint_dir, out_dir, jobs, renames, skip_missing):
    os.makedirs(out_dir, exist_ok=True)
    ck = v41io.Checkpoint(checkpoint_dir)
    for path, g, dspark in template_files(template_dir):
        fd = os.open(os.path.join(out_dir, out_name(path, renames)), os.O_RDWR | os.O_CREAT, 0o644)
        with open(path, "rb") as f:
            header = f.read(g["data_start"])
        os.pwrite(fd, header, 0)
        os.ftruncate(fd, g["size"])
        os.close(fd)
    tasks, skipped = all_tasks(ck, template_dir, out_dir, renames, skip_missing)
    if skipped:
        print(f"{skipped} tensors left out: their shards are not downloaded yet")
    journal_path = os.path.join(out_dir, "convert-journal.jsonl")
    done = set()
    if os.path.exists(journal_path):
        for line in open(journal_path):
            j = json.loads(line)
            done.add((j["file"], j["offset"]))
    todo = [t for t in tasks if (t[0], t[1]) not in done]
    # the longest pieces first, so the pool does not end on one big table
    todo.sort(key=lambda t: -t[2] * (8 if t[3]["kind"] == "fp8_mxfp4" else 1))
    total = sum(t[2] for t in tasks)
    left = sum(t[2] for t in todo)
    print(f"{len(tasks)} pieces, {total / 2**30:.1f} GiB; {len(todo)} pieces, {left / 2**30:.1f} GiB to do, {jobs} jobs",
          flush=True)
    t0, written, last = time.time(), 0, 0.0
    with open(journal_path, "a") as journal, mp.Pool(jobs, initializer=_init, initargs=(checkpoint_dir,)) as pool:
        for out_path, offset, size, secs in pool.imap_unordered(_work, todo, chunksize=1):
            journal.write(json.dumps({"file": out_path, "offset": offset, "size": size}) + "\n")
            journal.flush()
            written += size
            now = time.time()
            if now - last > 30 or written == left:
                rate = written / max(now - t0, 1e-9)
                print(f"{written / 2**30:7.1f} / {left / 2**30:.1f} GiB, {rate / 2**20:5.0f} MiB/s, "
                      f"ETA {(left - written) / max(rate, 1) / 60:5.1f} min", flush=True)
                last = now
    print(f"done in {(time.time() - t0) / 60:.1f} min" + ("" if not skipped else ", run again once the download completes"))
    return 0


def check(template_dir, checkpoint_dir, out_dir, renames, samples):
    import random
    bad = 0
    for path, g, dspark in template_files(template_dir):
        out_path = os.path.join(out_dir, out_name(path, renames))
        with open(path, "rb") as a, open(out_path, "rb") as b:
            same_header = a.read(g["data_start"]) == b.read(g["data_start"])
        same_size = os.path.getsize(out_path) == g["size"]
        bad += not (same_header and same_size)
        print(f"{'OK  ' if same_header and same_size else 'FAIL'} {os.path.basename(out_path)}: header "
              f"{'same' if same_header else 'DIFFERS'}, size {'same' if same_size else 'DIFFERS'}")
    ck = v41io.Checkpoint(checkpoint_dir)
    tasks, skipped = all_tasks(ck, template_dir, out_dir, renames, skip_missing=True)
    journal = os.path.join(out_dir, "convert-journal.jsonl")
    done = {(j["file"], j["offset"]) for j in map(json.loads, open(journal))} if os.path.exists(journal) else set()
    missing = [t for t in tasks if (t[0], t[1]) not in done]
    bad += len(missing) > 0
    print(f"journal: {len(tasks) - len(missing)} of {len(tasks)} pieces written" +
          (f"; {skipped} tensors not converted yet, their shards are still downloading" if skipped else ""))
    written = [t for t in tasks if (t[0], t[1]) in done]
    random.seed(0)
    for out_path, offset, size, spec in random.sample(written, min(samples, len(written))):
        with open(out_path, "rb") as f:
            f.seek(offset)
            same = f.read(size) == run(ck, spec)
        bad += not same
        if not same:
            print(f"FAIL {os.path.basename(out_path)} @ {offset}: {spec}")
    print(f"{min(samples, len(written))} random written pieces converted again: " +
          ("all identical" if not bad else f"{bad} problems") + ("" if not skipped else " (partial: conversion not finished)"))
    return 1 if bad or skipped else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("template_dir")
    v.add_argument("repo")
    v.add_argument("revision", nargs="?", default="main")
    c = sub.add_parser("convert")
    c.add_argument("template_dir")
    c.add_argument("checkpoint_dir")
    c.add_argument("out_dir")
    c.add_argument("--jobs", type=int, default=6)
    c.add_argument("--rename", action="append", default=[], help="OLD=NEW applied to output file names")
    c.add_argument("--skip-missing", action="store_true")
    k = sub.add_parser("check")
    k.add_argument("template_dir")
    k.add_argument("checkpoint_dir")
    k.add_argument("out_dir")
    k.add_argument("--rename", action="append", default=[])
    k.add_argument("--samples", type=int, default=200)
    a = p.parse_args()
    if a.cmd == "verify":
        return verify(a.template_dir, a.repo, a.revision)
    renames = [tuple(r.split("=", 1)) for r in a.rename]
    if a.cmd == "check":
        return check(a.template_dir, a.checkpoint_dir, a.out_dir, renames, a.samples)
    return convert(a.template_dir, a.checkpoint_dir, a.out_dir, a.jobs, renames, a.skip_missing)


if __name__ == "__main__":
    sys.exit(main())
