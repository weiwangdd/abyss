#!/usr/bin/env python3
"""Compares two expert stats files (ABYSS_EXPERT_STATS) of the same requests: per phase, the layers whose token counts
agree, and how many of the expert picks differ (half the summed absolute difference of the per-expert counts, over
the picks).

    python3 stats_compare.py a.json b.json
"""

import json
import sys


def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    for phase in ("prefill", "decode"):
        ta, tb = a[phase]["tokens"], b[phase]["tokens"]
        n = max(len(ta), len(tb))
        ta, tb = ta + [0] * (n - len(ta)), tb + [0] * (n - len(tb))
        same = sum(x == y for x, y in zip(ta, tb))
        ha, hb = a[phase]["hits"], b[phase]["hits"]
        picks = diff = 0
        worst = (0.0, -1)
        for il in range(n):
            ra = ha[il] if il < len(ha) else []
            rb = hb[il] if il < len(hb) else []
            m = max(len(ra), len(rb))
            ra, rb = ra + [0] * (m - len(ra)), rb + [0] * (m - len(rb))
            d = sum(abs(x - y) for x, y in zip(ra, rb)) / 2
            p = max(sum(ra), sum(rb))
            picks += p
            diff += d
            if p and d / p > worst[0]:
                worst = (d / p, il)
        print(f"{phase}: tokens agree in {same}/{n} layers ({sum(ta)} vs {sum(tb)} layer tokens); "
              f"{diff:.0f} of {picks} picks differ ({100 * diff / max(picks, 1):.3f}%)"
              + (f", most in layer {worst[1]} ({100 * worst[0]:.2f}%)" if worst[1] >= 0 else ""))


if __name__ == "__main__":
    sys.exit(main())
