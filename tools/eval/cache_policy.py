#!/usr/bin/env python3
"""Expert cache policies on a routing trace of a decode (expert-trace.sh), with what each costs in uploads: the
production plan (ABYSS_MOE_CPU_MISS: a share ADMIT of the misses, the hottest by decayed heat, is uploaded and evicts
the least recently used slot; the other misses go to the CPU) against evicting the slot whose expert has the lowest
decayed heat instead, with a few heat decays, and against admitting a miss only if it is hotter than the slot it
would evict. SLOTS slots are shared by layers 22-39 as in production. Per policy, after the warm-up: the hit rate, and
per token the misses (the CPU computes those not uploaded) and the uploads (each about 0.67 ms of the cache GPU).

    ~/src/comfyui/.venv/bin/python cache_policy.py <dump dir> [--slots 1140] [--warmup 500]
"""
import argparse
import collections
import glob
import os
import re

import numpy as np

FIRST, LAST = 22, 39
N_EXPERT, N_USED = 384, 6


def load_topk(d):
    names = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(d, "ffn_moe_topk-*.bin")):
        m = re.match(r"ffn_moe_topk-(\d+)\.(\d+)\.bin$", os.path.basename(p))
        if m and FIRST <= int(m.group(1)) <= LAST:
            names[int(m.group(1))][int(m.group(2))] = p
    steps = min(len(v) for v in names.values())
    top = {}
    for layer, files in names.items():
        rows = []
        for i in range(steps):
            with open(files[i], "rb") as f:
                head = f.read(40)
                assert head[:4] == b"ADMP"
                rows.append(np.frombuffer(f.read(), np.int32)[:N_USED])
        top[layer] = np.stack(rows)
    return steps, top


def simulate(steps, top, slots, warmup, admit, decay, evict, gate=False):
    """evict: 'lru' or 'cold'; gate: admit a miss only if hotter than the expert it would evict"""
    layers = sorted(top)
    heat = np.zeros((LAST + 1, N_EXPERT))
    slot_key = np.full(slots, -1, dtype=np.int64)    # layer * N_EXPERT + expert
    slot_of = {}                                     # key -> slot
    used = collections.OrderedDict((s, True) for s in range(slots))  # LRU order of the slots, head = oldest
    acc = 0.0
    hits = misses = uploads = picks = 0
    for i in range(steps):
        for layer in layers:
            heat[layer] *= decay
            ids = [int(e) for e in top[layer][i]]
            for e in ids:
                heat[layer, e] += 1
            miss = []
            for e in ids:
                s = slot_of.get(layer * N_EXPERT + e)
                if s is None:
                    miss.append(e)
                else:
                    used.move_to_end(s)
            acc += admit * len(miss)
            n = min(len(miss), int(acc))
            acc -= n
            up = 0
            # the slots this step reads stay: its hits, and the misses it admits
            protect = {slot_of[layer * N_EXPERT + x] for x in ids if layer * N_EXPERT + x in slot_of}
            for e in sorted(miss, key=lambda e: -heat[layer, e])[:n]:
                if evict == "lru":
                    victim = next(iter(used))
                else:
                    # the slot whose expert is the coldest, empty slots first
                    h = np.where(slot_key >= 0, heat.reshape(-1)[np.maximum(slot_key, 0)], -1.0)
                    h[list(protect)] = np.inf
                    victim = int(np.argmin(h))
                if gate and slot_key[victim] >= 0 and heat.reshape(-1)[slot_key[victim]] >= heat[layer, e]:
                    continue
                if slot_key[victim] >= 0:
                    del slot_of[int(slot_key[victim])]
                slot_key[victim] = layer * N_EXPERT + e
                slot_of[layer * N_EXPERT + e] = victim
                used.move_to_end(victim)
                protect.add(victim)
                up += 1
            if i >= warmup:
                picks += N_USED
                hits += N_USED - len(miss)
                misses += len(miss)
                uploads += up
    n = steps - warmup
    return hits / picks, misses / n, uploads / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--slots", type=int, default=1140)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--more", action="store_true", help="the second set of variants")
    a = ap.parse_args()
    steps, top = load_topk(a.dump)
    print(f"{steps} decode steps, layers {min(top)}-{max(top)}, {a.slots} slots")
    print(f"{'policy':52s} {'hit rate':>9s} {'misses/token':>13s} {'uploads/token':>14s}")
    variants = [
        ("production: admit 0.3, evict LRU", 0.3, 0.99, "lru", False),
        ("admit 0.2, evict LRU", 0.2, 0.99, "lru", False),
        ("admit 0.3, evict the coldest (decay 0.99)", 0.3, 0.99, "cold", False),
        ("admit 0.3, evict the coldest (decay 0.995)", 0.3, 0.995, "cold", False),
        ("admit 0.3, evict the coldest (decay 0.999)", 0.3, 0.999, "cold", False),
        ("admit 0.2, evict the coldest (decay 0.995)", 0.2, 0.995, "cold", False),
        ("admit 0.3 if hotter than the coldest (0.995)", 0.3, 0.995, "cold", True),
        ("admit 0.5 if hotter than the coldest (0.995)", 0.5, 0.995, "cold", True),
        ("admit 1.0 if hotter than the coldest (0.995)", 1.0, 0.995, "cold", True)]
    if a.more:
        variants = [
            ("admit 0.3 if hotter than the coldest (0.99)", 0.3, 0.99, "cold", True),
            ("admit 0.5 if hotter than the coldest (0.99)", 0.5, 0.99, "cold", True),
            ("admit 1.0 if hotter than the coldest (0.99)", 1.0, 0.99, "cold", True),
            ("admit 0.5 if hotter than the coldest (0.999)", 0.5, 0.999, "cold", True),
            ("admit 1.0 if hotter than the coldest (0.999)", 1.0, 0.999, "cold", True),
            ("admit 0.3 if hotter than the LRU slot (0.995)", 0.3, 0.995, "lru", True),
            ("admit 0.5 if hotter than the LRU slot (0.995)", 0.5, 0.995, "lru", True),
            ("admit 1.0 if hotter than the LRU slot (0.995)", 1.0, 0.995, "lru", True),
            ("admit 1.0 if hotter than the LRU slot (0.99)", 1.0, 0.99, "lru", True)]
    for name, admit, decay, evict, gate in variants:
        hr, m, u = simulate(steps, top, a.slots, a.warmup, admit, decay, evict, gate)
        print(f"{name:52s} {hr:9.1%} {m:13.1f} {u:14.1f}", flush=True)


if __name__ == "__main__":
    main()
