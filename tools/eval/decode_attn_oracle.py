#!/usr/bin/env python3
"""Check single-token attention of the compressed layers against float64, for ABYSS_DSV41_DECODE_GATHER.

llama-abyss-dump with ABYSS_DUMP_GEN=N saves, for N greedily decoded tokens, the query (hca_q), the keys and mask
flash attention gets (hca_k_all and hca_kq_mask without the gather, hca_k_top and hca_kq_mask_top with it) and the
output (attn_hca). This recomputes the attention in float64, with the layer's sinks from the GGUF, and reports how far
the kernel's output is from it, and whether the gathered rows are exactly the rows the mask leaves visible.

    PYTHONPATH=<mx-llama.cpp>/gguf-py python decode_attn_oracle.py <dump without> <dump with> <gguf dir> --layers 2 20 30
"""

import argparse
import glob
import os

import numpy as np
from gguf import GGUFReader

DTYPES = {0: np.float32, 1: np.float16}


def load(d, name):
    out = []
    while os.path.exists(path := f"{d}/{name}.{len(out)}.bin"):
        with open(path, "rb") as f:
            assert f.read(4) == b"ADMP"
            typ = int(np.frombuffer(f.read(4), np.int32)[0])
            ne = np.frombuffer(f.read(32), np.int64)
            out.append(np.frombuffer(f.read(), DTYPES[typ]).reshape(tuple(int(x) for x in ne[::-1])))
    return out


def sinks(gguf_dir, layer):
    for p in sorted(glob.glob(f"{gguf_dir}/*-of-*.gguf")):
        for t in GGUFReader(p).tensors:
            if t.name == f"blk.{layer}.attn_sinks.weight":
                return np.asarray(t.data, np.float64)
    raise SystemExit(f"no sinks for layer {layer}")


def attend(q, k, mask, sink, scale):
    """q [n_head, d], k [n_kv, d] (keys are the values), mask [n_kv]; float64 with the sink in the denominator"""
    s = q @ k.T * scale + mask[None, :]
    m = np.maximum(s.max(axis=1), sink)
    p = np.exp(s - m[:, None])
    return (p @ k) / (p.sum(axis=1) + np.exp(sink - m))[:, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("without")
    ap.add_argument("with_")
    ap.add_argument("gguf")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 20, 30])
    args = ap.parse_args()

    for layer in args.layers:
        sink = sinks(args.gguf, layer)
        for label, d, kname, mname in (("without", args.without, "hca_k_all", "hca_kq_mask"),
                                       ("with   ", args.with_, "hca_k_top", "hca_kq_mask_top")):
            qs, ks, ms, outs = (load(d, f"{n}-{layer}") for n in ("hca_q", kname, mname, "attn_hca"))
            if not qs:
                print(f"layer {layer:2d} {label}: nothing dumped")
                continue
            errs, n_vis = [], []
            for q, k, m, o in zip(qs, ks, ms, outs):
                q = q.reshape(-1, q.shape[-1]).astype(np.float64)        # [n_head, d]
                k = k.reshape(-1, k.shape[-1]).astype(np.float64)        # [n_kv, d]
                m = m.reshape(-1, m.shape[-1])[0].astype(np.float64)     # query row 0, [n_kv]
                ref = attend(q, k, m, sink, 1 / np.sqrt(q.shape[-1]))
                out = o.reshape(ref.shape).astype(np.float64)
                errs.append(np.abs(out - ref).max() / np.abs(ref).max())
                n_vis.append(int(np.isfinite(m).sum()))
            print(f"layer {layer:2d} {label}: {len(errs)} steps, visible rows {n_vis}, "
                  f"max rel err vs float64 {', '.join(f'{e:.1e}' for e in errs)}")
        # the first step has the same inputs either way, so the two references must agree up to float64 rounding
        r = []
        for d, kname, mname in ((args.without, "hca_k_all", "hca_kq_mask"), (args.with_, "hca_k_top", "hca_kq_mask_top")):
            q, k, m = (load(d, f"{n}-{layer}")[0] for n in ("hca_q", kname, mname))
            q = q.reshape(-1, q.shape[-1]).astype(np.float64)
            k = k.reshape(-1, k.shape[-1]).astype(np.float64)
            m = m.reshape(-1, m.shape[-1])[0].astype(np.float64)
            r.append(attend(q, k, m, sink, 1 / np.sqrt(q.shape[-1])))
        print(f"layer {layer:2d} step 0: float64 references of the two paths differ by "
              f"{np.abs(r[0] - r[1]).max() / np.abs(r[0]).max():.1e}")


if __name__ == "__main__":
    main()
