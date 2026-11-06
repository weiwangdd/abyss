#!/usr/bin/env python3
"""How many pages of the Engram tables the decode steps of a server read from the disk: the counts of
engram-pagecache.bt (pages entering the page cache per 10 ms) set against the requests of the server's log, each
decoding over [release - eval time, release] and prefilling just before. Per decoded token: the pages read, and the
share of tokens that read any (a 10 ms bucket with pages, a token being about 19 ms); a token with cold rows waits for
the disk once, about 75 us while the drive is busy and about 600 us after it idled for 15 ms or more
(ABYSS_ENGRAM_KEEPALIVE_MS keeps it busy).

    engram_pagecache.py <engram-pagecache.log> <server.log> "<server start, YYYY-MM-DD HH:MM:SS>"
"""

import datetime
import re
import sys


def main():
    bt, log, start = sys.argv[1], sys.argv[2], datetime.datetime.strptime(sys.argv[3], "%Y-%m-%d %H:%M:%S")
    ref_ns = ref_wall = None
    buckets = {}
    for line in open(bt):
        m = re.match(r"start (\d+) (\S+ \S+)", line)
        if m:
            ref_ns, ref_wall = int(m.group(1)), datetime.datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S.%f")
            continue
        m = re.match(r"@pages\[(\d+)\]: (\d+)", line)
        if m:
            t = ref_wall + datetime.timedelta(seconds=(int(m.group(1)) * 10_000_000 - ref_ns) / 1e9)
            buckets[t] = buckets.get(t, 0) + int(m.group(2))
    if ref_ns is None:
        sys.exit("no start line in " + bt)
    end = max(buckets) if buckets else ref_wall

    # the requests: relative times M.SS.mmm.uuu since the server started
    def when(stamp):
        mins, secs, ms, us = stamp.split(".")
        return start + datetime.timedelta(minutes=int(mins), seconds=int(secs), milliseconds=int(ms), microseconds=int(us))

    reqs, prompt_ms, eval_ms, eval_n = [], None, None, None
    for line in open(log, errors="replace"):
        m = re.match(r"(\d+\.\d+\.\d+\.\d+) I slot print_timing: .*prompt eval time = +([\d.]+) ms", line)
        if m:
            prompt_ms = float(m.group(2))
            continue
        m = re.match(r"(\d+\.\d+\.\d+\.\d+) I slot print_timing: .*       eval time = +([\d.]+) ms / +(\d+) tokens", line)
        if m:
            eval_ms, eval_n = float(m.group(2)), int(m.group(3))
            continue
        m = re.match(r"(\d+\.\d+\.\d+\.\d+) I slot      release: .*n_tokens = (\d+)", line)
        if m and eval_ms is not None:
            t1 = when(m.group(1))
            t0 = t1 - datetime.timedelta(milliseconds=eval_ms)
            p0 = t0 - datetime.timedelta(milliseconds=prompt_ms or 0)
            if t0 >= ref_wall and t1 <= end + datetime.timedelta(seconds=10):
                reqs.append((p0, t0, t1, eval_n, int(m.group(2))))
            prompt_ms = eval_ms = eval_n = None

    times = sorted(buckets)
    tok = pages = hot = pre_pages = 0
    print(f"{len(reqs)} requests from {ref_wall:%m-%d %H:%M} to {end:%m-%d %H:%M}")
    for p0, t0, t1, n, ctx in reqs:
        d = [buckets[t] for t in times if t0 <= t < t1]
        pp = sum(buckets[t] for t in times if p0 <= t < t0)
        tok, pages, hot, pre_pages = tok + n, pages + sum(d), hot + len(d), pre_pages + pp
        print(f"  {t0:%H:%M:%S} context {ctx / 1e3:5.0f}k: decode {n:5d} tokens, {sum(d):6d} pages in {len(d):5d} buckets "
              f"({len(d) / max(n, 1):.0%} of tokens), prefill before it {pp:7d} pages")
    if tok:
        print(f"decode: {tok} tokens, {pages / tok:.2f} pages per token, {hot / tok:.1%} of tokens read any; "
              f"prefill: {pre_pages} pages")


if __name__ == "__main__":
    main()
