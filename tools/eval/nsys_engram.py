#!/usr/bin/env python3
"""How long the GPUs wait for the Engram rows in a decode, from an nsys sqlite export (decode_modes.sh CAPTURE=1).

The Engram layers (1 and 14 of DeepSeek V4.1) project the gathered rows with a matrix-vector product of 25,600 rows,
the only one of that size, one per token on each GPU; the CPU gathers the rows just before it (a get_rows of its own
split), so the largest gap on that GPU among the 3 kernels before the projection is the wait for the rows (the
quantization of the gathered rows sits between). Per GPU: the median, p10, p90 and mean of that gap over the tokens.

    python3 nsys_engram.py decode-modes.sqlite [rows of the projection, default 25600]
"""

import sqlite3
import sys


def main():
    db = sqlite3.connect(sys.argv[1])
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 25600
    kern = db.execute("select k.deviceId, k.start, k.end, s.value, k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on k.shortName = s.id").fetchall()
    # the decode starts after the prefill's last batched matrix multiplication
    t0 = max((k[2] for k in kern if k[3] == "mul_mat_q"), default=0)
    total = 0.0
    for d in sorted({k[0] for k in kern}):
        ks = sorted((s, e, n, gx) for dd, s, e, n, gx in kern if dd == d and s > t0)
        ends = []
        m = 0
        for s, e, n, gx in ks:
            m = max(m, e)
            ends.append(m)
        gaps = []
        for i, (s, e, n, gx) in enumerate(ks):
            if n == "mul_mat_vec_q" and gx == rows and i >= 4:
                gaps.append(max(0, max(ks[j][0] - ends[j - 1] for j in range(i - 3, i + 1))) / 1e3)
        if not gaps:
            continue
        gaps.sort()
        n = len(gaps)
        mean = sum(gaps) / n
        total += mean
        print(f"GPU{d}: {n} tokens, wait before the Engram projection: median {gaps[n // 2]:.0f} us, p10 {gaps[n // 10]:.0f}, "
              f"p90 {gaps[9 * n // 10]:.0f}, mean {mean:.0f}")
    print(f"both GPUs: {total / 1e3:.2f} ms per token")


if __name__ == "__main__":
    sys.exit(main())
