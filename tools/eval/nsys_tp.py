#!/usr/bin/env python3
"""What tensor parallelism of layers 0-21 could save, from an nsys sqlite export of a decode turn with the production
settings (prod-decode-profile.sh, node tracing).

Per token, the stretch before layer 22's plan (the first copy of a layer's FFN input from GPU1 to the host, as in
nsys_layers.py): layers 0-21, GPU0's 0-11 and GPU1's 12-21 one after the other (and the attention of layer 22). In it,
each GPU's kernels by their own time (with PDL a kernel's own time starts at the end of the kernel before), by class:
the selected experts' matrix-vector products (mul_mat_vec* with a grid of 6 rows, one per expert), the other
matrix-vector products (the dense layers, the shared expert, the router), attention (flash attention, its fixup, the
gathered keys, the indexer and its top-k), the rest (norms, hyper-connections, rope, quantization, routing, element-wise);
and the time neither GPU runs a kernel or a copy. Then the stretch as it would be with the dense products (and, at the
upper bound, the attention) divided between the GPUs, GPU0 taking SHARE of them (default 0.57, its share of these
layers' weights now, which keeps GPU1's room for the expert cache; 0.5 for an even split), each of a token's experts on
GPU0 with probability SHARE and each layer waiting for the slower GPU, every other kernel running on both, the idle time
as now, and 44 all-reduces per token (two per layer) of US microseconds each (default 20; p2p_allreduce.cu measures it).
It leaves out the launches that the extra synchronization points add (a CUDA graph cut costs a GPU some 5-12 us).

    python3 nsys_tp.py decode-modes.sqlite [US] [SHARE]
"""

import collections
import math
import sqlite3
import sys

N_USED = 6  # experts per token: their products run with that many rows of the grid
N_LAYERS = 22


def kind(name, grid_y):
    if name.startswith("mul_mat_vec"):
        return "experts" if grid_y == N_USED else "matvec"
    if name.startswith(("flash_attn", "dsv4_gather_kv", "lightning_indexer", "top_k_radix")):
        return "attention"
    return "rest"


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def covered(iv, s, e):
    return sum(max(0, min(b, e) - max(a, s)) for a, b in iv)


def main():
    db = sqlite3.connect(sys.argv[1])
    us = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.57
    kern = db.execute("select k.start, k.end, k.deviceId, s.value, k.gridY from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on k.shortName = s.id").fetchall()
    t0 = max((k[1] for k in kern if k[3] == "mul_mat_q"), default=min(k[0] for k in kern))
    mem = db.execute("select start, end, deviceId, copyKind, bytes, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY "
                     "where start > ?", (t0,)).fetchall()
    rt = db.execute("select r.correlationId, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r "
                    "join StringIds s on r.nameId = s.id where r.start > ?", (t0,)).fetchall()
    peer = {c for c, n in rt if n.startswith("cudaMemcpyPeerAsync")}
    n_vocab_bytes = collections.Counter(m[4] for m in mem if m[3] == 2 and m[4] > 400000).most_common(1)[0][0]
    logits = sorted(m[1] for m in mem if m[3] == 2 and m[4] == n_vocab_bytes)
    marks = sorted(m[0] for m in mem if m[3] == 2 and m[4] == 20480 and m[2] == 1 and m[5] not in peer)

    # each kernel's own time, per GPU
    own = []
    for d in (0, 1):
        ks = sorted((s, e, n, gy) for s, e, dev, n, gy in kern if dev == d and e > t0)
        prev = 0
        for s, e, n, gy in ks:
            own.append((max(s, prev), e, d, kind(n, gy)))
            prev = max(prev, e)
    own.sort()
    busy = union([(s, e) for s, e, d, k in own] + [(m[0], m[1]) for m in mem])

    totals = collections.defaultdict(float)  # (gpu, kind) -> ns
    n_tok = 0
    length = idle = 0.0
    prev_end = None
    j = 0
    for t_end in logits:
        if prev_end is None:
            prev_end = t_end
            continue
        m = [x for x in marks if prev_end < x < t_end]
        if not m:
            prev_end = t_end
            continue
        a, b = prev_end, m[0]
        n_tok += 1
        length += b - a
        idle += (b - a) - covered(busy, a, b)
        while j < len(own) and own[j][1] <= a:
            j += 1
        i = j
        while i < len(own) and own[i][0] < b:
            s, e, d, k = own[i]
            totals[(d, k)] += max(0, min(e, b) - max(s, a))
            i += 1
        prev_end = t_end

    ms = lambda ns: ns / n_tok / 1e6
    kinds = ("experts", "matvec", "attention", "rest")
    print(f"{n_tok} tokens; layers 0-21 (to layer 22's plan): {ms(length):.2f} ms per token, both GPUs idle {ms(idle):.2f}")
    print(f"{'kernels, ms/token':18s} " + " ".join(f"{k:>9s}" for k in kinds) + f" {'all':>9s}")
    for d in (0, 1):
        row = [totals[(d, k)] for k in kinds]
        print(f"{'GPU' + str(d):18s} " + " ".join(f"{ms(x):9.2f}" for x in row) + f" {ms(sum(row)):9.2f}")
    tot = {k: totals[(0, k)] + totals[(1, k)] for k in kinds}
    ar = 2 * N_LAYERS * us * 1e3 * n_tok
    for label, split, whole in (("dense matvec split, attention on both", tot["matvec"], tot["attention"] + tot["rest"]),
                                ("dense matvec and attention split", tot["matvec"] + tot["attention"], tot["rest"])):
        for f in sorted({share, 0.5}, reverse=True):
            # per layer the slower GPU: its share of the split work plus the experts of the token that it holds, the
            # holder of each of the token's experts drawn with probability f
            sl, el = split / N_LAYERS, tot["experts"] / N_LAYERS
            per_layer = sum(math.comb(N_USED, x) * f ** x * (1 - f) ** (N_USED - x) *
                            max(f * sl + x / N_USED * el, (1 - f) * sl + (N_USED - x) / N_USED * el)
                            for x in range(N_USED + 1))
            t = N_LAYERS * per_layer + whole + idle + ar
            print(f"{label:38s} GPU0 share {f:.2f}: {ms(t):.2f} ms per token, saves {ms(length - t):.2f} "
                  f"(after all-reduces of {ms(ar):.2f})")

if __name__ == "__main__":
    sys.exit(main())
