#!/usr/bin/env python3
"""Compares two runs of dist_check: how many of the draws selected another token, and how far the probabilities of
the first draw are apart (largest relative difference among the tokens with p > 1e-6, and the summed absolute one).

    python3 dist_check.py off.txt on.txt
"""

import sys


def load(path):
    sel, p = [], []
    for line in open(path):
        kind, value = line.split()
        (sel if kind == "s" else p).append(int(value) if kind == "s" else float(value))
    return sel, p


def main():
    (sa, pa), (sb, pb) = load(sys.argv[1]), load(sys.argv[2])
    same = sum(x == y for x, y in zip(sa, sb))
    rel = max((abs(x - y) / x for x, y in zip(pa, pb) if x > 1e-6), default=0.0)
    l1 = sum(abs(x - y) for x, y in zip(pa, pb))
    print(f"draws selecting the same token: {same}/{len(sa)}; first draw's probabilities: largest relative "
          f"difference {rel:.2e} (p > 1e-6), summed absolute difference {l1:.2e}, sums {sum(pa):.7f} and {sum(pb):.7f}")


if __name__ == "__main__":
    sys.exit(main())
