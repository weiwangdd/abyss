#!/usr/bin/env python3
"""What pipelining a long prefill's ubatches over the two GPUs could save, from an nsys sqlite export of one prefill
(prefill-pipe.sh).

Now the GPUs take turns in each ubatch: GPU0 computes layers 0-11, GPU1 layer 12's attention, GPU0 layer 12's gate and
up experts (they live on GPU0), GPU1 the rest of layers 12-39, and the host prepares the next ubatch while both wait.
A ubatch starts where GPU0 starts after both GPUs waited at least GAP ms (default 0.5) with GPU1 the last one busy, and
runs to the next start; the histogram of the gaps is printed to check GAP. A ubatch whose host-to-GPU0 copies exceed
BIG MiB (default 4096) is a tail (the decoder layers' FFN of a llama_decode call's last tokens: the big batches' experts
are uploaded to GPU0) and stays as it is, and so does one shorter than 100 ms. In the other ones, the encoder ubatches,
GPU0's busy runs (gaps under 0.1 ms joined) are cut before its last run of at least 1 ms: before the cut, a = GPU0's
part (layers 0-11, and the inputs it copies) and p = GPU1's part after GPU0's run before the cut (layer 12's
attention); from the cut on, e = GPU0's part (layer 12's experts) and q = GPU1's part (the rest). With GPU0 computing
the next ubatch's a while GPU1 computes this one's q, a ubatch takes max(a, q) + p + e, the host's preparation hidden
behind the GPUs (it must stay shorter than that); with layer 12's experts on GPU1 instead, max(a, p + e + q); and no
schedule beats max(GPU0's, GPU1's busy time). Now it takes a + p + e + q plus the waits.

    python3 nsys_pipe.py prefill.sqlite [GAP_MS] [BIG_MIB]
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


class Busy:
    """The union of a device's kernels and copies, with the busy time inside any stretch."""

    def __init__(self, iv):
        self.iv = union(iv)
        self.starts = [s for s, e in self.iv]
        self.cum = [0]
        for s, e in self.iv:
            self.cum.append(self.cum[-1] + e - s)

    def before(self, t):  # busy time before t
        j = bisect.bisect_right(self.starts, t) - 1
        if j < 0:
            return 0
        s, e = self.iv[j]
        return self.cum[j] + min(t, e) - s

    def covered(self, s, e):
        return self.before(e) - self.before(s)

    def busy_at(self, t):
        j = bisect.bisect_right(self.starts, t) - 1
        return j >= 0 and self.iv[j][1] > t

    def last_end(self, t):  # end of the last stretch that ends by t (t itself if one covers t)
        j = bisect.bisect_right(self.starts, t) - 1
        if j < 0:
            return None
        return min(self.iv[j][1], t)


def ms(ns):
    return ns / 1e6


def main():
    db = sqlite3.connect(sys.argv[1])
    gap_min = float(sys.argv[2]) * 1e6 if len(sys.argv) > 2 else 0.5e6
    big = float(sys.argv[3]) * 2**20 if len(sys.argv) > 3 else 4096 * 2**20
    kern = db.execute("select start, end, deviceId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    mem = db.execute("select start, end, deviceId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    dev = [Busy([(s, e) for s, e, d in kern if d == g] + [(s, e) for s, e, d, k, b in mem if d == g]) for g in (0, 1)]
    both = Busy(dev[0].iv + dev[1].iv)
    t0 = both.iv[0][0]
    t1 = both.iv[-1][1]
    print("span %.2f s: GPU0 busy %.2f s, GPU1 busy %.2f s, both busy %.2f s, both idle %.2f s" % (
        (t1 - t0) / 1e9, dev[0].covered(t0, t1) / 1e9, dev[1].covered(t0, t1) / 1e9,
        (dev[0].covered(t0, t1) + dev[1].covered(t0, t1) - both.covered(t0, t1)) / 1e9,
        (t1 - t0 - both.covered(t0, t1)) / 1e9))

    # where GPU0 starts again after both GPUs waited with GPU1 the last one busy: a ubatch's start if GPU1 worked at
    # least 30 ms since GPU0's last run of 1 ms or more (layers 12-39), else the hand-over to layer 12's experts
    restarts = []
    hist = {True: [0] * 8, False: [0] * 8}
    edges = [0.05, 0.1, 0.5, 1, 2, 5, 20]
    runs = []  # GPU0's runs, gaps under 0.1 ms joined; the ends of those with 1 ms or more of work
    for s, e in dev[0].iv:
        if runs and s - runs[-1][1] < 0.1e6:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    long_ends = [e for s, e in runs if dev[0].covered(s, e) >= 1e6]
    for k in range(1, len(dev[0].iv)):
        s = dev[0].iv[k][0]
        if dev[1].busy_at(s):
            continue
        prev0 = dev[0].iv[k - 1][1]
        last1 = dev[1].last_end(s)
        last1 = last1 if last1 is not None else 0
        g = s - max(prev0, last1)
        after1 = last1 > prev0
        hist[after1][bisect.bisect_right(edges, ms(g))] += 1
        if g >= gap_min and after1:
            j = bisect.bisect_left(long_ends, s) - 1
            restarts.append((s, dev[1].covered(long_ends[j] if j >= 0 else t0, s) >= 30e6))
    labels = ["<0.05", "0.05-0.1", "0.1-0.5", "0.5-1", "1-2", "2-5", "5-20", ">=20"]
    print("both GPUs waiting before GPU0 starts again (ms): " + ", ".join(
        "%s %d/%d" % (l, hist[True][i], hist[False][i]) for i, l in enumerate(labels)) +
        "  (GPU1 the last one busy / GPU0)")

    starts = [dev[0].iv[0][0]] + [s for s, ub in restarts if ub]
    handovers = [s for s, ub in restarts if not ub]
    h2d = sorted((s, b, d) for s, e, d, k, b in mem if k == 1)
    h2d_starts = [x[0] for x in h2d]
    bounds = starts + [t1]
    rows = []
    for i in range(len(starts)):
        S, T = bounds[i], bounds[i + 1]
        lo, hi = bisect.bisect_left(h2d_starts, S), bisect.bisect_left(h2d_starts, T)
        up = [sum(b for s, b, d in h2d[lo:hi] if d == g) for g in (0, 1)]
        row = dict(S=S, T=T, dur=T - S, b0=dev[0].covered(S, T), b1=dev[1].covered(S, T),
                   wait=(T - S) - both.covered(S, T), up0=up[0], up1=up[1])
        hs = [h for h in handovers[bisect.bisect_left(handovers, S):bisect.bisect_left(handovers, T)]]
        row["tail"] = up[0] > big or T - S < 100e6 or len(hs) != 1
        if len(hs) == 1:
            h = hs[0]
            row.update(a=dev[0].covered(S, h), p=dev[1].covered(S, h), e=dev[0].covered(h, T), q=dev[1].covered(h, T),
                       w1=(h - S) - both.covered(S, h), w2=(T - h) - both.covered(h, T))
        rows.append(row)

    enc = [r for r in rows if not r["tail"]]
    tails = [r for r in rows if r["tail"]]
    print("%d ubatches: %d encoder ubatches, %d tails or other" % (len(rows), len(enc), len(tails)))
    print("%4s %8s %7s %7s %7s %7s %7s %7s %7s %7s %7s %8s %7s" % (
        "#", "start s", "ms", "GPU0", "GPU1", "a", "p", "e", "q", "wait1", "wait2", "GPU0 MiB", "GPU1"))
    for i, r in enumerate(rows):
        if len(rows) > 24 and 6 <= i < len(rows) - 6 and not r["tail"]:
            continue
        f = lambda k: "%.1f" % ms(r[k]) if k in r else "-"
        print("%4d %8.3f %7.1f %7.1f %7.1f %7s %7s %7s %7s %7s %7s %8.0f %7.0f%s" % (
            i, (r["S"] - t0) / 1e9, ms(r["dur"]), ms(r["b0"]), ms(r["b1"]), f("a"), f("p"), f("e"), f("q"), f("w1"),
            f("w2"), r["up0"] / 2**20, r["up1"] / 2**20, "  tail" if r["tail"] else ""))

    span = t1 - t0
    if enc:
        tot = {k: sum(r[k] for r in enc) for k in ("dur", "a", "p", "e", "q", "w1", "w2", "b0", "b1")}
        n = len(enc)
        piped = sum(max(r["a"], r["q"]) + r["p"] + r["e"] for r in enc)
        moved = sum(max(r["a"], r["p"] + r["e"] + r["q"]) for r in enc)
        ideal = sum(max(r["b0"], r["b1"]) for r in enc)
        rest = span - tot["dur"]
        print("encoder ubatches: %.2f s, per ubatch %.0f ms: a %.0f, p %.1f, e %.1f, q %.0f, both waiting %.1f + %.1f" % (
            tot["dur"] / 1e9, ms(tot["dur"]) / n, ms(tot["a"]) / n, ms(tot["p"]) / n, ms(tot["e"]) / n,
            ms(tot["q"]) / n, ms(tot["w1"]) / n, ms(tot["w2"]) / n))
        print("tails and the rest: %.2f s (%d)" % (rest / 1e9, len(tails)))
        print("pipelined: encoder ubatches %.2f s, prefill %.2f s instead of %.2f (%.0f%%); with layer 12's experts "
              "on GPU1, %.2f and %.2f s (%.0f%%)" % (
                  piped / 1e9, (rest + piped) / 1e9, span / 1e9, 100 * (rest + piped - span) / span,
                  moved / 1e9, (rest + moved) / 1e9, 100 * (rest + moved - span) / span))
        print("no schedule beats: encoder ubatches %.2f s, prefill %.2f s (%.0f%%)" % (
            ideal / 1e9, (rest + ideal) / 1e9, 100 * (rest + ideal - span) / span))

    # what the host does while both GPUs wait: CUDA calls of the thread that makes the most of them
    try:
        rt = db.execute("select r.start, r.end, r.globalTid, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r "
                        "join StringIds s on r.nameId = s.id").fetchall()
    except sqlite3.Error:
        rt = []
    if rt:
        main_tid = collections.Counter(r[2] for r in rt).most_common(1)[0][0]
        idle = []
        prev = t0
        for s, e in both.iv:
            if s > prev:
                idle.append((prev, s))
            prev = max(prev, e)
        idle_total = sum(e - s for s, e in idle)
        istarts = [s for s, e in idle]
        by = collections.Counter()
        for s, e, tid, name in rt:
            if tid != main_tid:
                continue
            j = bisect.bisect_right(istarts, e) - 1
            while j >= 0 and idle[j][1] > s:
                by[name.split("_v")[0]] += min(e, idle[j][1]) - max(s, idle[j][0])
                j -= 1
        print("both GPUs waiting %.2f s; the main thread in CUDA calls meanwhile: %s; outside them %.2f s" % (
            idle_total / 1e9, ", ".join("%s %.2f" % (k, v / 1e9) for k, v in by.most_common(5)),
            (idle_total - sum(by.values())) / 1e9))


if __name__ == "__main__":
    main()
