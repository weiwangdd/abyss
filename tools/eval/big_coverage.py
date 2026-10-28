#!/usr/bin/env python3
"""Per request of a server log with ABYSS_MOE_BIG_STATS=1: the large batches of layers 22-39 (the prefill's batches of
more than 32 tokens) and how much of them the expert cache already held, by experts and by rows (token-expert pairs),
with the share of those cached experts the plan sent to the CPU; the tokens the request prefilled and their time.

    big_coverage.py LOG
"""
import re
import sys

BIG = re.compile(r"abyss: big batch il=(\d+) tokens=(\d+) experts=(\d+) cached=(\d+) rows=(\d+) cached_rows=(\d+) "
                 r"cpu=(\d+) cached_cpu=(\d+)")
PROMPT = re.compile(r"prompt eval time = +([\d.]+) ms / +(\d+) tokens")


def main():
    reqs, cur = [], []
    for line in open(sys.argv[1], errors="replace"):
        if m := BIG.search(line):
            cur.append(tuple(map(int, m.groups())))
        elif m := PROMPT.search(line):
            reqs.append((float(m.group(1)), int(m.group(2)), cur))
            cur = []
    print(f"{'req':>3} {'prefill':>8} {'s':>7} {'batches':>7} {'tok/batch':>9} {'exp/batch':>9} {'cached exp':>10} "
          f"{'cached rows':>11} {'of cached on CPU':>16}")
    tot = [0] * 8
    for i, (ms, n, bs) in enumerate(reqs, 1):
        if not bs:
            print(f"{i:>3} {n:>8} {ms / 1000:>7.2f} {0:>7}")
            continue
        s = [sum(b[k] for b in bs) for k in range(8)]
        if i > 1:  # the first request starts from an empty cache
            tot = [a + b for a, b in zip(tot, s)]
        print(f"{i:>3} {n:>8} {ms / 1000:>7.2f} {len(bs):>7} {s[1] / len(bs):>9.0f} {s[2] / len(bs):>9.0f} "
              f"{100 * s[3] / s[2]:>9.1f}% {100 * s[5] / s[4]:>10.1f}% {100 * s[7] / max(1, s[3]):>15.1f}%")
    if tot[2]:
        print(f"after the first request: experts {100 * tot[3] / tot[2]:.1f}% cached, rows {100 * tot[5] / tot[4]:.1f}% "
              f"cached, {100 * tot[7] / max(1, tot[3]):.1f}% of the cached experts sent to the CPU")


if __name__ == "__main__":
    main()
