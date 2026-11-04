#!/usr/bin/env python3
"""How long the host memory takes for the misses of the cached layers that defer their experts (22-38) on a routing
trace of a decode (expert-trace.sh), under the production cache plan (ABYSS_MOE_ADMIT=0.3 with the admission gate,
evicting the coldest by decayed heat 0.99, SLOTS slots shared by layers 22-39): per layer step the uploads and the
misses the CPU computes, and the time of both reading the host memory, as is and with up to K of the CPU's misses
uploaded to spare slots instead (computed by the GPU, as with mx-llama.cpp/0031). The rates are those measured at
5200 MT/s: the CPU alone 50 GB/s, the CPU 32.3 and the uploads 28.7 GB/s while both read.

    ~/src/comfyui/.venv/bin/python cpu_dma_balance.py <dump dir> [--slots 1140] [--warmup 500]
"""
import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_policy as cp  # noqa: E402

EXPERT = 18.8e6                            # bytes of an expert: gate, up and down in MXFP4
CPU, CPU_SHARED, DMA = 50e9, 32.3e9, 28.7e9


def mem_time(cpu_bytes, dma_bytes):
    """the CPU and the uploads read at the same time, the one left over then alone"""
    if dma_bytes == 0:
        return cpu_bytes / CPU
    if cpu_bytes == 0:
        return dma_bytes / DMA
    tc, td = cpu_bytes / CPU_SHARED, dma_bytes / DMA
    if tc > td:
        return td + (cpu_bytes - CPU_SHARED * td) / CPU
    return td


def plan(steps, top, slots, warmup):
    """the production plan: (uploads, CPU misses) of each deferred layer step after the warm-up"""
    layers = sorted(top)
    heat = np.zeros((cp.LAST + 1, cp.N_EXPERT))
    slot_key = np.full(slots, -1, dtype=np.int64)
    slot_of = {}
    acc = 0.0
    rec = []
    for i in range(steps):
        for layer in layers:
            heat[layer] *= 0.99
            ids = [int(e) for e in top[layer][i]]
            for e in ids:
                heat[layer, e] += 1
            miss = [e for e in ids if layer * cp.N_EXPERT + e not in slot_of]
            acc += 0.3 * len(miss)
            n = min(len(miss), int(acc))
            acc -= n
            protect = {slot_of[layer * cp.N_EXPERT + x] for x in ids if layer * cp.N_EXPERT + x in slot_of}
            up = 0
            for e in sorted(miss, key=lambda e: -heat[layer, e])[:n]:
                h = np.where(slot_key >= 0, heat.reshape(-1)[np.maximum(slot_key, 0)], -1.0)
                h[list(protect)] = np.inf
                victim = int(np.argmin(h))
                if slot_key[victim] >= 0 and heat.reshape(-1)[slot_key[victim]] >= heat[layer, e]:
                    break
                if slot_key[victim] >= 0:
                    del slot_of[int(slot_key[victim])]
                slot_key[victim] = layer * cp.N_EXPERT + e
                slot_of[layer * cp.N_EXPERT + e] = victim
                protect.add(victim)
                up += 1
            if i >= warmup and layer < cp.LAST:
                rec.append((up, len(miss) - up))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--slots", type=int, default=1140)
    ap.add_argument("--warmup", type=int, default=500)
    a = ap.parse_args()
    steps, top = cp.load_topk(a.dump)
    rec = plan(steps, top, a.slots, a.warmup)
    n_tok = steps - a.warmup
    print(f"{n_tok} tokens, {len(rec) / n_tok:.0f} deferred layers per token; per token "
          f"{sum(u for u, c in rec) / n_tok:.2f} uploads, {sum(c for u, c in rec) / n_tok:.2f} CPU misses")
    print("layer steps per token by uploads and CPU misses, and the host memory time of one:")
    for (u, c), k in sorted(collections.Counter(rec).items()):
        if k / n_tok >= 0.05:
            print(f"  {u} up {c} cpu: {k / n_tok:5.2f}  {mem_time(c * EXPERT, u * EXPERT) * 1e3:.2f} ms")
    base = sum(mem_time(c * EXPERT, u * EXPERT) for u, c in rec)
    print(f"host memory time per token as is: {base / n_tok * 1e3:.2f} ms")
    for k_max in (1, 2, 3):
        total, moved = 0.0, 0
        for u, c in rec:
            best, best_k = mem_time(c * EXPERT, u * EXPERT), 0
            for k in range(1, min(k_max, c) + 1):
                t = mem_time((c - k) * EXPERT, (u + k) * EXPERT)
                if t < best - 1e-12:
                    best, best_k = t, k
            total += best
            moved += best_k
        print(f"up to {k_max} CPU misses to spare slots: {total / n_tok * 1e3:.2f} ms per token "
              f"({(total - base) / n_tok * 1e3:+.2f}), {moved / n_tok:.2f} moved per token")


if __name__ == "__main__":
    main()
