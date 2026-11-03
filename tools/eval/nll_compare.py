#!/usr/bin/env python3
"""Paired comparison of two per-token NLL files of llama-abyss-dump's forced decoding (ABYSS_DUMP_FORCE), the same
tokens under two configurations: the mean NLL and perplexity of each, and the mean difference B - A with its standard
error, as a relative perplexity change.

    python3 nll_compare.py A/nll.txt B/nll.txt
"""

import math
import sys


def main():
    a, b = ([float(x) for x in open(p)] for p in sys.argv[1:3])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = [y - x for x, y in zip(a, b)]
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n)
    ma, mb = sum(a) / n, sum(b) / n
    print(f"{n} tokens: A mean NLL {ma:.5f} (PPL {math.exp(ma):.4f}), B mean NLL {mb:.5f} (PPL {math.exp(mb):.4f})")
    print(f"B - A: {mean:+.5f} +/- {se:.5f} nats per token, PPL {100 * (math.exp(mean) - 1):+.2f}% +/- "
          f"{100 * se:.2f}%; tokens with |difference| > 0.01: {sum(abs(x) > 0.01 for x in d)}")


if __name__ == "__main__":
    sys.exit(main())
