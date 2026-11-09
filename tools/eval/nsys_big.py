#!/usr/bin/env python3
"""Where a turn's prefill spends the decoder layers' large batches, from an nsys sqlite export of one agent turn
(big-pipe.sh): whether GPU0 computing the uploaded experts while it uploads them could pay.

With ABYSS_MOE_BIG_SPLIT each decoder layer of a large batch splits its experts three ways: the ones in the MoE cache
are computed on GPU1 (0023), the ones with few tokens by the CPU, the rest uploaded to GPU0 and computed there, the
uploads queued ahead of the computation on the same stream. A layer starts at a burst of host-to-GPU0 copies of at
least BIG MiB (default 64; copies less than GAP ms apart, default 0.2, are one burst) and runs to the next burst. Per
layer: the burst (its time, GiB and GB/s), GPU0's kernels after it, GPU1's kernels, the time both GPUs wait, and the
host's waits in cudaStreamSynchronize, which is how much the GPUs' paths outlast the CPU's share. Overlapping GPU0's
computation with its upload shortens a layer by at most min(upload, GPU0 kernels), and by no more than the host's wait
once the CPU's share is the longer path: the estimate takes the smallest of the three.

    python3 nsys_big.py turn.sqlite [BIG_MIB] [GAP_MS]
"""

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


def covered(iv, s, e):  # time of the union iv inside [s, e)
    return sum(max(0, min(e, b) - max(s, a)) for a, b in iv if b > s and a < e)


def main():
    db = sqlite3.connect(sys.argv[1])
    big = float(sys.argv[2]) * 2**20 if len(sys.argv) > 2 else 64 * 2**20
    gap = float(sys.argv[3]) * 1e6 if len(sys.argv) > 3 else 0.2e6
    kern = db.execute("select start, end, deviceId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    mem = db.execute("select start, end, deviceId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    syncs = db.execute("select r.start, r.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId = s.id "
                       "where s.value like 'cudaStreamSynchronize%' or s.value like 'cudaDeviceSynchronize%' "
                       "or s.value like 'cudaEventSynchronize%'").fetchall()
    k = [union([(s, e) for s, e, d in kern if d == g]) for g in (0, 1)]
    busy = [union([(s, e) for s, e, d in kern if d == g] + [(s, e) for s, e, d, c, b in mem if d == g]) for g in (0, 1)]
    both = union(busy[0] + busy[1])
    host = union(syncs)

    # bursts of host-to-GPU0 copies
    h2d = sorted((s, e, b) for s, e, d, c, b in mem if d == 0 and c == 1)
    bursts = []
    for s, e, b in h2d:
        if bursts and s - bursts[-1][1] < gap:
            bursts[-1][1] = max(bursts[-1][1], e)
            bursts[-1][2] += b
            bursts[-1][3].append((s, e))
        else:
            bursts.append([s, e, b, [(s, e)]])
    layers = [x for x in bursts if x[2] >= big]
    if not layers:
        sys.exit(f"no burst of host-to-GPU0 copies of at least {big / 2**20:.0f} MiB")
    t_end = max(e for s, e in both)
    t0 = both[0][0]
    print(f"GPU activity window {(t_end - t0) / 1e9:.2f} s; {len(layers)} layers with an upload burst of at least "
          f"{big / 2**20:.0f} MiB (bursts of host-to-GPU0 copies under {gap / 1e6:.1f} ms apart)")
    print(f"{'layer':>5} {'span':>7} {'upload':>7} {'GiB':>5} {'GB/s':>5} {'GPU0 k':>7} {'during':>7} {'GPU1 k':>7} "
          f"{'idle':>6} {'host':>6} {'save':>6}   (ms)")
    tot = dict(span=0, up=0, gb=0, k0=0, during=0, k1=0, idle=0, host=0, save=0, bound=0)
    for i, (s, e, b, cps) in enumerate(layers):
        nxt = layers[i + 1][0] if i + 1 < len(layers) else t_end
        span = nxt - s
        up = covered(union(cps), s, e)
        k0 = covered(k[0], e, nxt)                # GPU0's kernels after the upload
        during = covered(k[0], s, e)              # GPU0's kernels while it uploads
        k1 = covered(k[1], s, nxt)
        idle = span - covered(both, s, nxt)
        hw = covered(host, s, nxt)
        bound = min(up, k0)
        save = min(bound, hw)
        for key, v in (("span", span), ("up", up), ("gb", b / 2**30), ("k0", k0), ("during", during), ("k1", k1),
                       ("idle", idle), ("host", hw), ("save", save), ("bound", bound)):
            tot[key] += v
        print(f"{i:5d} {span / 1e6:7.1f} {up / 1e6:7.1f} {b / 2**30:5.2f} {b / up:5.1f} {k0 / 1e6:7.1f} "
              f"{during / 1e6:7.1f} {k1 / 1e6:7.1f} {idle / 1e6:6.1f} {hw / 1e6:6.1f} {save / 1e6:6.1f}")
    n = len(layers)
    print(f"{'sum':>5} {tot['span'] / 1e6:7.1f} {tot['up'] / 1e6:7.1f} {tot['gb']:5.2f} "
          f"{tot['gb'] * 2**30 / tot['up']:5.1f} {tot['k0'] / 1e6:7.1f} {tot['during'] / 1e6:7.1f} "
          f"{tot['k1'] / 1e6:7.1f} {tot['idle'] / 1e6:6.1f} {tot['host'] / 1e6:6.1f} {tot['save'] / 1e6:6.1f}")
    print(f"the layers take {tot['span'] / 1e9:.2f} s of the turn ({tot['span'] / n / 1e6:.1f} ms each); overlapping "
          f"GPU0's computation with its upload could save at most {tot['bound'] / 1e9:.2f} s, about "
          f"{tot['save'] / 1e9:.2f} s with the CPU's share as the floor")


if __name__ == "__main__":
    main()
