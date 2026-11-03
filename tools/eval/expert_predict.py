#!/usr/bin/env python3
"""Expert routing analysis of an expert-trace.sh dump (llama-abyss-dump, one file per decode step and tensor).

1. Check: the routing recomputed here (each layer's router weights from the GGUF applied to the dumped ffn_norm,
   sqrt-softplus scores plus the selection bias, top 6) against the dumped ffn_moe_topk.
2. Prediction: layer T's router applied to the router input of layer T-k (k = 1, 2, 3) of the same token, as the share
   of T's six experts among the predicted top K; "same" uses the input as it is (normed with layer T-k's weight),
   "renorm" swaps in layer T's norm weight. The experts of layer T at the previous token are the baseline.
3. Expert cache on the trace of layers 22-39 (SLOTS slots shared by all layers, as in production): hit rates of LRU,
   of LRU admitting a share of the misses by decayed heat (production: ABYSS_MOE_ADMIT, heat decay 0.99), of evicting
   the coldest by decayed heat, and of Belady with bypass (the optimum when a miss may skip the cache, as the CPU path
   allows). Then the production policy with prefetch: before layer T the predicted experts that are not cached are
   uploaded, and what is left are the misses that wait.

  ~/src/comfyui/.venv/bin/python expert_predict.py <dump dir> <model dir> [--slots 1140] [--warmup 500]
"""
import argparse
import collections
import glob
import heapq
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/src/mx-llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

FIRST, LAST = 22, 39            # the layers whose experts live in host memory
N_EXPERT, N_USED = 384, 6
DTYPES = {0: np.float32, 26: np.int32}


def read_dump(path):
    with open(path, "rb") as f:
        head = f.read(40)
        assert head[:4] == b"ADMP", path
        typ = int(np.frombuffer(head[4:8], np.int32)[0])
        ne = np.frombuffer(head[8:40], np.int64)
        data = np.frombuffer(f.read(), DTYPES[typ])
    return data.reshape(int(ne[1]), int(ne[0]))


def load_trace(d):
    names = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(d, "*.bin")):
        m = re.match(r"(ffn_norm|ffn_moe_topk)-(\d+)\.(\d+)\.bin$", os.path.basename(p))
        if m:
            names[(m.group(1), int(m.group(2)))][int(m.group(3))] = p
    steps = min(len(v) for v in names.values())
    x = {l: np.stack([read_dump(names[("ffn_norm", l)][i])[0] for i in range(steps)])
         for (n, l) in names if n == "ffn_norm"}
    top = {l: np.stack([read_dump(names[("ffn_moe_topk", l)][i])[0] for i in range(steps)])
           for (n, l) in names if n == "ffn_moe_topk"}
    return steps, x, top


def load_router(model_dir, layers):
    want = {}
    for l in layers:
        want[f"blk.{l}.ffn_gate_inp.weight"] = ("w", l)
        want[f"blk.{l}.exp_probs_b.bias"] = ("b", l)
        want[f"blk.{l}.ffn_norm.weight"] = ("g", l)
    got = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "*-of-*.gguf"))):
        for t in GGUFReader(shard).tensors:
            if t.name not in want:
                continue
            raw = np.asarray(t.data)
            if t.tensor_type.name == "BF16":
                val = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                val = raw.astype(np.float32)
            got[want[t.name]] = val.reshape(N_EXPERT, -1) if want[t.name][0] == "w" else val.reshape(-1)
    missing = [k for k in want.values() if k not in got]
    assert not missing, missing
    return got


def scores(x, router, layer):
    logits = x @ router[("w", layer)].T
    soft = np.logaddexp(0.0, logits)            # softplus
    return np.sqrt(soft) + router[("b", layer)]


def topk(s, k):
    return np.argpartition(-s, k - 1, axis=1)[:, :k]


def overlap(pred, actual):
    """mean share of the actual experts (rows of N_USED) that the predicted rows contain"""
    return float(np.mean([len(set(a) & set(p)) / N_USED for p, a in zip(pred, actual)]))


class Cache:
    """SLOTS slots shared by all layers; keys are (layer, expert)"""

    def __init__(self, slots, policy, admit=0.3, decay=0.99):
        self.slots, self.policy, self.admit, self.decay = slots, policy, admit, decay
        self.lru = collections.OrderedDict()
        self.heat = np.zeros((LAST + 1, N_EXPERT))
        self.acc = np.zeros(LAST + 1)

    def insert(self, key):
        if key in self.lru:
            self.lru.move_to_end(key)
            return
        if len(self.lru) >= self.slots:
            if self.policy == "lfu":
                victim = min(self.lru, key=lambda k: self.heat[k])
                del self.lru[victim]
            else:
                self.lru.popitem(last=False)
        self.lru[key] = True

    def access(self, layer, experts):
        """returns the misses; updates recency, heat and admission"""
        self.heat[layer] *= self.decay
        self.heat[layer, experts] += 1
        hits = [e for e in experts if (layer, e) in self.lru]
        miss = [e for e in experts if (layer, e) not in self.lru]
        for e in hits:
            self.lru.move_to_end((layer, e))
        if self.policy == "admit":
            self.acc[layer] += self.admit * len(miss)
            n = int(self.acc[layer])
            self.acc[layer] -= n
            for e in sorted(miss, key=lambda e: -self.heat[layer, e])[:n]:
                self.insert((layer, e))
        else:
            for e in miss:
                self.insert((layer, e))
        return miss


def belady_bypass(seq, slots):
    """hits of the optimal policy that may bypass the cache: seq is the list of keys in access order"""
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        nxt[i] = last.get(seq[i], float("inf"))
        last[seq[i]] = i
    cached = {}                 # key -> its next use
    heap = []                   # (-next use, key), lazily invalidated
    hits = 0
    for i, key in enumerate(seq):
        if key in cached:
            hits += 1
            cached[key] = nxt[i]
            heapq.heappush(heap, (-nxt[i], key))
            continue
        if nxt[i] == float("inf"):
            continue            # never used again: bypass
        if len(cached) < slots:
            cached[key] = nxt[i]
            heapq.heappush(heap, (-nxt[i], key))
            continue
        while True:
            far, victim = heap[0]
            if cached.get(victim) == -far:
                break
            heapq.heappop(heap)
        if nxt[i] < -far:       # keep the one used sooner
            heapq.heappop(heap)
            del cached[victim]
            cached[key] = nxt[i]
            heapq.heappush(heap, (-nxt[i], key))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("model")
    ap.add_argument("--slots", type=int, default=1140)
    ap.add_argument("--warmup", type=int, default=500)
    a = ap.parse_args()

    steps, x, top = load_trace(a.dump)
    layers = sorted(x)
    router = load_router(a.model, layers)
    targets = [l for l in range(FIRST, LAST + 1) if l in top]
    print(f"{steps} decode steps, layers {layers[0]}-{layers[-1]}")
    out = {"steps": steps}

    # 1. the routing recomputed from the dumped router input
    exact = {l: float(np.mean([set(p) == set(t) for p, t in zip(topk(scores(x[l], router, l), N_USED), top[l])]))
             for l in targets}
    print(f"recomputed routing: exact top-{N_USED} match {min(exact.values()):.4f}-{max(exact.values()):.4f} "
          f"(worst layer {min(exact, key=exact.get)})")
    out["recompute_exact"] = exact

    # 2. predicting layer T from layer T-k
    w = slice(a.warmup, None)
    preds = {}
    print("\nshare of layer T's experts inside the predicted top K (layers 22-39, after warm-up)")
    print(f"{'source':22s} " + " ".join(f"K={k:<5d}" for k in (6, 8, 12, 16)))
    for k in (1, 2, 3):
        for variant in ("same", "renorm"):
            row = []
            for K in (6, 8, 12, 16):
                vals = []
                for t in targets:
                    s = t - k
                    xs = x[s] if variant == "same" else x[s] / router[("g", s)] * router[("g", t)]
                    p = topk(scores(xs, router, t), K)
                    preds[(k, variant, K, t)] = p
                    vals.append(overlap(p[w], top[t][w]))
                row.append(float(np.mean(vals)))
            out[f"pred_k{k}_{variant}"] = row
            print(f"layer T-{k}, {variant:6s}       " + " ".join(f"{v:7.1%}" for v in row))
    prev = float(np.mean([overlap(top[t][a.warmup - 1:-1], top[t][w]) for t in targets]))
    out["prev_token"] = prev
    print(f"previous token, same layer  {prev:7.1%}  (its six experts)")

    # 3. expert cache policies on the trace of layers 22-39
    print(f"\nexpert cache, {a.slots} slots, {len(targets) * N_USED} picks per token")
    seq = [(t, int(e)) for i in range(steps) for t in targets for e in top[t][i]]
    res = {}
    for policy in ("lru", "admit", "lfu"):
        c = Cache(a.slots, policy)
        miss_after = picks_after = 0
        for i in range(steps):
            for t in targets:
                m = c.access(t, list(map(int, top[t][i])))
                if i >= a.warmup:
                    miss_after += len(m)
                    picks_after += N_USED
        res[policy] = 1 - miss_after / picks_after
    hits = belady_bypass(seq, a.slots)
    # Belady counts every access; report it over the whole trace and the others over the same span for comparison
    res["belady_bypass_all"] = hits / len(seq)
    for policy in ("lru", "admit", "lfu"):
        c = Cache(a.slots, policy)
        miss = sum(len(c.access(t, list(map(int, top[t][i])))) for i in range(steps) for t in targets)
        res[policy + "_all"] = 1 - miss / len(seq)
    out["cache"] = res
    print(f"{'policy':34s} {'whole trace':>12s} {'after warm-up':>14s}")
    for name, key in (("LRU, every miss admitted", "lru"), ("production: admit 0.3 by heat", "admit"),
                      ("evict the coldest by heat", "lfu")):
        print(f"{name:34s} {res[key + '_all']:12.1%} {res[key]:14.1%}")
    print(f"{'Belady with bypass (optimum)':34s} {res['belady_bypass_all']:12.1%}")

    # prefetch on top of the production policy, with the better of the two inputs at k = 1, K = 8. Layer 22 is predicted
    # from layer 21, whose experts stay on the GPU and which has no plan to prefetch with, so ABYSS_MOE_PREFETCH starts
    # at layer 23 (the rows "from 23")
    variant = max(("same", "renorm"), key=lambda v: out[f"pred_k1_{v}"][1])
    print(f"\nproduction policy + prefetch of the predicted experts ({variant}) before layer T (after warm-up)")
    print(f"{'prediction':26s} {'misses/token':>12s} {'uploads/token':>14s} {'useful':>8s}")
    base = Cache(a.slots, "admit")
    m0 = 0
    for i in range(steps):
        for t in targets:
            m = base.access(t, list(map(int, top[t][i])))
            if i >= a.warmup:
                m0 += len(m)
    n_after = steps - a.warmup
    print(f"{'none':26s} {m0 / n_after:12.1f} {'':>14s}")
    out["prefetch"] = {"none": m0 / n_after}
    for first in (FIRST, FIRST + 1):
        for k in (1, 2):
            for K in (2, 3, 4, 6, 8):
                pred = {}
                for t in targets:
                    if t >= first:
                        xs = x[t - k] if variant == "same" else x[t - k] / router[("g", t - k)] * router[("g", t)]
                        pred[t] = preds.get((k, variant, K, t))
                        if pred[t] is None:
                            pred[t] = topk(scores(xs, router, t), K)
                c = Cache(a.slots, "admit")
                miss = up = useful = 0
                for i in range(steps):
                    for t in targets:
                        want = list(map(int, top[t][i]))
                        fetched = [int(e) for e in pred[t][i] if (t, int(e)) not in c.lru] if t in pred else []
                        for e in fetched:
                            c.insert((t, e))
                        m = c.access(t, want)
                        if i >= a.warmup:
                            miss += len(m)
                            up += len(fetched)
                            useful += len(set(fetched) & set(want))
                label = f"layer T-{k}, top {K}" + (f", from {first}" if first > FIRST else "")
                out["prefetch"][label] = {"misses": miss / n_after, "uploads": up / n_after, "useful": useful / max(up, 1)}
                print(f"{label:26s} {miss / n_after:12.1f} {up / n_after:14.1f} {useful / max(up, 1):8.0%}")

    with open(os.path.join(a.dump, "expert_predict.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
