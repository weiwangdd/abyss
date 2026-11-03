#!/usr/bin/env python3
"""Per-layer view of the decode of an nsys sqlite export (decode_modes.sh CAPTURE=1): who waits for whom in the layers
whose experts the MoE cache serves (22-39), the place of the expert deferral (ABYSS_MOE_DEFER).

Each plan of such a layer copies the layer's FFN input from GPU1 to the host for the CPU's experts, n_embd floats
(MARK bytes, default 20480; the six selected experts, 24 bytes, are no mark when the expert stats read every layer's
selection too), so the time from one of these copies to the next is one layer; a token ends with the copy of its
logits (n_vocab floats). Per
layer: its length, how much of it GPU1 computed or copied, how much of it the driving thread waited in
cudaStreamSynchronize (the host waiting for the GPU), and how much it spent outside CUDA calls (the host computing:
the plan and the CPU's experts). Per token: the part before the first such layer (layers 0-21), the layers, and the
tail after the last plan (the last layer's rest, the output head, the logits copy).

    python3 nsys_layers.py decode-modes.sqlite [MARK]
"""

import bisect
import collections
import sqlite3
import sys


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(iv, s, e):
    # total length of the sorted disjoint intervals iv inside [s, e)
    total = 0
    i = bisect.bisect_right([x[1] for x in iv], s)
    while i < len(iv) and iv[i][0] < e:
        total += min(iv[i][1], e) - max(iv[i][0], s)
        i += 1
    return total


def main():
    db = sqlite3.connect(sys.argv[1])
    kern = db.execute("select k.start, k.end, k.deviceId, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on k.shortName = s.id").fetchall()
    t0 = max((k[1] for k in kern if k[3] == "mul_mat_q"), default=min(k[0] for k in kern))
    mem = db.execute("select start, end, deviceId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start > ?",
                     (t0,)).fetchall()
    rt = db.execute("select r.start, r.end, r.globalTid, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r "
                    "join StringIds s on r.nameId = s.id where r.start > ?", (t0,)).fetchall()
    # the driving thread is the one launching the graphs
    tid = collections.Counter(r[2] for r in rt if r[3].startswith("cudaGraphLaunch")).most_common(1)[0][0]
    api = sorted((s, e, n.split("_v")[0]) for s, e, t, n in rt if t == tid)
    api_all = union([(s, e) for s, e, n in api])
    sync = union([(s, e) for s, e, n in api if n == "cudaStreamSynchronize"])

    g1 = union([(s, e) for s, e, d, n in kern if d == 1 and e > t0] + [(s, e) for s, e, d, k, b in mem if d == 1])
    g0 = union([(s, e) for s, e, d, n in kern if d == 0 and e > t0] + [(s, e) for s, e, d, k, b in mem if d == 0])

    n_vocab_bytes = collections.Counter(b for s, e, d, k, b in mem if k == 2 and b > 400000).most_common(1)[0][0]
    logits = sorted(e for s, e, d, k, b in mem if k == 2 and b == n_vocab_bytes)
    mark = int(sys.argv[2]) if len(sys.argv) > 2 else 20480
    marks = sorted(s for s, e, d, k, b in mem if k == 2 and b == mark and d == 1)

    layers, heads, tails, walls = [], [], [], []
    prev_end = None
    for t_end in logits:
        if prev_end is None:
            prev_end = t_end
            continue
        m = [x for x in marks if prev_end < x < t_end]
        if len(m) < 2:
            prev_end = t_end
            continue
        walls.append(t_end - prev_end)
        heads.append(m[0] - prev_end)
        tails.append(t_end - m[-1])
        for a, b in zip(m, m[1:]):
            d = b - a
            layers.append((d, overlap(g1, a, b), overlap(g0, a, b), overlap(sync, a, b), d - overlap(api_all, a, b)))
        prev_end = t_end

    n_tok = len(walls)
    ms = lambda ns: ns / 1e6
    print(f"{n_tok} tokens: {ms(sum(walls)) / n_tok:.2f} ms/token; before the first cached layer {ms(sum(heads)) / n_tok:.2f}, "
          f"cached layers {ms(sum(l[0] for l in layers)) / n_tok:.2f} ({len(layers) / n_tok:.0f} per token), "
          f"tail after the last plan {ms(sum(tails)) / n_tok:.2f}")
    n = len(layers)
    tot = [sum(l[i] for l in layers) / n for i in range(5)]
    print(f"per cached layer {ms(tot[0]):.3f} ms: GPU1 busy {ms(tot[1]):.3f} ({100 * tot[1] / tot[0]:.0f}%), "
          f"host waiting in cudaStreamSynchronize {ms(tot[3]):.3f}, host outside CUDA calls {ms(tot[4]):.3f}, "
          f"GPU1 idle while the host is outside CUDA calls ~{ms(max(0, tot[0] - tot[1] - tot[3])):.3f}")
    lens = sorted(l[0] for l in layers)
    print(f"layer length: p10 {ms(lens[n // 10]):.3f}, median {ms(lens[n // 2]):.3f}, p90 {ms(lens[9 * n // 10]):.3f} ms")


if __name__ == "__main__":
    sys.exit(main())
