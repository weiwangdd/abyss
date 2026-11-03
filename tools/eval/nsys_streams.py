#!/usr/bin/env python3
"""Per-stream view of one device in the decode part of an nsys sqlite export (decode_modes.sh CAPTURE=1), for the
expert prefetch of ABYSS_MOE_PREFETCH, which uploads on a stream of its own: per generated token, the host-to-device
copies of each stream and how much of them ran while the device computed, the device's kernel time and idle time,
and the CUDA runtime calls of the driving thread that block (stream synchronization) or order streams (events).

    python3 nsys_streams.py decode-modes.sqlite <generated tokens> [device, default 1]
"""

import sqlite3
import sys


def union(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def length(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):
    # a, b: sorted disjoint intervals
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def main():
    db = sqlite3.connect(sys.argv[1])
    n_tok = int(sys.argv[2])
    dev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    kern = db.execute("select k.deviceId, k.streamId, k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on k.shortName = s.id").fetchall()
    mem = db.execute("select deviceId, streamId, copyKind, bytes, start, end from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()

    # the decode window as in nsys_decode.py: after the last quantized matmul of the prefill
    t0 = max((k[3] for k in kern if k[4] == "mul_mat_q"), default=min(k[2] for k in kern))
    t1 = max([k[3] for k in kern] + [m[5] for m in mem])
    clip = lambda iv: [(max(s, t0), min(e, t1)) for s, e in iv if e > t0 and s < t1]
    ms = lambda ns: ns / n_tok / 1e6
    wall = t1 - t0
    print(f"decode window {wall / 1e9:.2f} s for {n_tok} tokens: {ms(wall):.2f} ms/token, device {dev}")

    k_dev = union(clip([(s, e) for d, st, s, e, _ in kern if d == dev]))
    print(f"  kernels {ms(length(k_dev)):6.2f} ms/token")
    streams = sorted({st for d, st, kind, b, s, e in mem if d == dev and kind == 1 and e > t0})
    all_h2d = []
    for st in streams:
        cs = [(s, e) for d, sid, kind, b, s, e in mem if d == dev and sid == st and kind == 1]
        nb = sum(b for d, sid, kind, b, s, e in mem if d == dev and sid == st and kind == 1 and e > t0)
        cu = union(clip(cs))
        all_h2d += cu
        with_k = intersect(cu, k_dev)
        print(f"  stream {st:3d} host->device: {len(cu) / n_tok:6.1f} copies, {nb / n_tok / 2**20:7.1f} MiB, "
              f"{ms(length(cu)):6.2f} ms/token, {ms(length(with_k)):6.2f} of it while kernels ran "
              f"({100 * length(with_k) / max(length(cu), 1):.0f}%)")
    h2d = union(all_h2d)
    busy = union(k_dev + h2d)
    print(f"  all host->device {ms(length(h2d)):6.2f} ms/token; device busy {ms(length(busy)):6.2f}, "
          f"idle {ms(wall - length(busy)):6.2f}")
    if len(streams) > 1:
        a = union(clip([(s, e) for d, sid, kind, b, s, e in mem if d == dev and sid == streams[0] and kind == 1]))
        b = union(clip([(s, e) for d, sid, kind, b, s, e in mem if d == dev and sid != streams[0] and kind == 1]))
        print(f"  host->device of stream {streams[0]} and of the others at the same time: {ms(length(intersect(a, b))):.2f} ms/token")

    tables = {r[0] for r in db.execute("select name from sqlite_master where type = 'table'")}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        print("  CUDA runtime calls per token (all threads):")
        rows = db.execute("select s.value, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r "
                          "join StringIds s on r.nameId = s.id where r.start >= ? and r.start < ? and "
                          "(s.value like '%Synchronize%' or s.value like '%Event%' or s.value like '%Memcpy%' or "
                          "s.value like '%GraphLaunch%') group by s.value order by 3 desc", (t0, t1)).fetchall()
        for name, c, ns in rows:
            print(f"    {c / n_tok:7.1f} calls {ms(ns):6.2f} ms  {name}")


if __name__ == "__main__":
    sys.exit(main())
