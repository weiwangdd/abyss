#!/usr/bin/env python3
"""The decode's kernels by their own time per token, per GPU, from an nsys sqlite export (decode_modes.sh CAPTURE=1).

With PDL a kernel starts before the one before it on the GPU has ended and waits for it, so the time between its start
and end overstates it; its own time here runs from the end of the kernel before (or its start, if later) to its end.
Grouped by kernel name and launch shape (grid x, block x), the top N (default 30) by own time, with how many run per
token and the mean of both times.

    python3 nsys_kernels.py decode-modes.sqlite [N]
"""

import collections
import sqlite3
import sys


def main():
    db = sqlite3.connect(sys.argv[1])
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rows = db.execute("select k.deviceId, k.start, k.end, s.value, k.gridX, k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on k.shortName = s.id").fetchall()
    # the decode starts after the prefill's last batched matrix multiplication
    t0 = max((r[2] for r in rows if r[3] == "mul_mat_q"), default=min(r[1] for r in rows))
    mem = db.execute("select end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start > ?", (t0,)).fetchall()
    # a token ends with the copy of its logits (n_vocab floats, the most frequent large device to host copy)
    n_vocab_bytes = collections.Counter(b for e, b, k in mem if k == 2 and b > 400000).most_common(1)[0][0]
    n_tok = sum(1 for e, b, k in mem if k == 2 and b == n_vocab_bytes)

    by_dev = collections.defaultdict(list)
    for r in rows:
        if r[1] > t0:
            by_dev[r[0]].append(r)
    agg = collections.defaultdict(lambda: [0, 0, 0])  # count, duration, own time
    for d, ks in by_dev.items():
        ks.sort(key=lambda r: r[1])
        prev_end = 0
        for _, s, e, name, gx, bx in ks:
            a = agg[(d, name, gx, bx)]
            a[0] += 1
            a[1] += e - s
            a[2] += max(0, e - max(s, prev_end))
            prev_end = max(prev_end, e)

    print(f"{n_tok} tokens")
    for d in sorted(by_dev):
        own = sum(v[2] for k, v in agg.items() if k[0] == d)
        dur = sum(v[1] for k, v in agg.items() if k[0] == d)
        print(f"GPU{d}: kernels {own / n_tok / 1e6:.2f} ms/token own time ({dur / n_tok / 1e6:.2f} start to end)")
    print("GPU kernel                              grid x  block  per token  mean us  own us  own ms/token")
    for (d, name, gx, bx), (c, dur, own) in sorted(agg.items(), key=lambda kv: -kv[1][2])[:top]:
        print(f"{d:3d} {name[:36]:36s} {gx:7d} {bx:6d} {c / n_tok:10.1f} {dur / c / 1e3:8.2f} {own / c / 1e3:7.2f} "
              f"{own / n_tok / 1e6:13.3f}")


if __name__ == "__main__":
    sys.exit(main())
