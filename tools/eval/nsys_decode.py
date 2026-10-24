#!/usr/bin/env python3
"""Per-token breakdown of the decode part of an nsys sqlite export (see profile-decode.sh).

The decode window starts after the last quantized-matmul (MMQ) kernel, which only batches of more than 8 tokens use, and
ends at the last GPU activity. Times are per generated token: busy and idle per device, kernels by name, host-device
copies by size (expert cache uploads are the 5-7 MiB ones), and CUDA runtime calls.

    python3 nsys_decode.py decode-48k.sqlite <generated tokens>
"""

import collections
import sqlite3
import sys

COPY_KINDS = {1: "host->device", 2: "device->host", 8: "device->device", 10: "peer->peer"}


def union(intervals):
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def main():
    db = sqlite3.connect(sys.argv[1])
    n_tok = int(sys.argv[2])
    kern = db.execute("select k.deviceId, k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on k.shortName = s.id").fetchall()
    mem = db.execute("select deviceId, copyKind, bytes, start, end from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()

    t0 = max((k[2] for k in kern if k[3] == "mul_mat_q"), default=min(k[1] for k in kern))
    t1 = max([k[2] for k in kern] + [m[4] for m in mem])
    # the first token after the prefill also pays for sampling the prompt's last logits; counting it is fine
    wall = (t1 - t0) / n_tok
    clip = lambda iv: [(max(s, t0), min(e, t1)) for s, e in iv if e > t0 and s < t1]
    ms = lambda ns: ns / n_tok / 1e6
    print(f"decode window {(t1 - t0) / 1e9:.2f} s for {n_tok} tokens: {wall / 1e6:.1f} ms/token")

    every = []
    for d in sorted({k[0] for k in kern}):
        ks = clip([(k[1], k[2]) for k in kern if k[0] == d])
        cs = clip([(m[3], m[4]) for m in mem if m[0] == d])
        every += ks + cs
        print(f"device {d}: kernels {ms(union(ks)):6.2f} ms/token, copies {ms(union(cs)):6.2f}, "
              f"busy {ms(union(ks + cs)):6.2f}, idle {wall / 1e6 - ms(union(ks + cs)):6.2f}")
    print(f"both devices idle: {wall / 1e6 - ms(union(every)):.2f} ms/token")

    for d in sorted({k[0] for k in kern}):
        by, cnt = collections.Counter(), collections.Counter()
        for dev, s, e, name in kern:
            if dev == d and s >= t0:
                by[name] += e - s
                cnt[name] += 1
        print(f"\ndevice {d} kernels: {sum(cnt.values()) / n_tok:.0f} per token")
        for name, ns in by.most_common(10):
            print(f"  {ms(ns):6.2f} ms/token {cnt[name] / n_tok:6.1f}/token  {name[:80]}")

    print("\ncopies per token:")
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for d, kind, nbytes, s, e in mem:
        if s < t0:
            continue
        size = "5-7 MiB" if 5 * 2**20 <= nbytes < 7 * 2**20 else ("< 1 MiB" if nbytes < 2**20 else "other")
        a = agg[(d, COPY_KINDS.get(kind, str(kind)), size)]
        a[0] += 1
        a[1] += nbytes
        a[2] += e - s
    for (d, kind, size), (n, nbytes, ns) in sorted(agg.items()):
        print(f"  device {d} {kind:15s} {size:8s} {n / n_tok:7.1f} copies {nbytes / n_tok / 2**20:8.1f} MiB "
              f"{ms(ns):6.2f} ms  {nbytes / max(ns, 1):5.1f} GB/s")

    tables = {r[0] for r in db.execute("select name from sqlite_master where type = 'table'")}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        print("\nCUDA runtime calls per token:")
        rows = db.execute("select s.value, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r "
                          "join StringIds s on r.nameId = s.id where r.start >= ? group by s.value order by 3 desc "
                          "limit 8", (t0,)).fetchall()
        for name, c, ns in rows:
            print(f"  {c / n_tok:7.1f} calls {ms(ns):6.2f} ms  {name}")


if __name__ == "__main__":
    sys.exit(main())
