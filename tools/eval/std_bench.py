#!/usr/bin/env python3
"""The AIPerf runs of std-bench.sh, one row per input:output shape: the time to first token (p50 and p99), the time per
output token after the first (AIPerf's inter-token latency) and the output tokens/s per user, means over the requests;
the share of prompt tokens the server took from its prompt cache shows the prefills were cold.

    std_bench.py DIR
"""
import json
import pathlib
import re
import sys


def main():
    runs = []
    for p in pathlib.Path(sys.argv[1]).glob("aiperf-*-*/profile_export_aiperf.json"):
        isl, osl = (int(x) for x in re.findall(r"\d+", p.parent.name))
        runs.append((isl, osl, json.load(open(p))))
    if not runs:
        sys.exit("no AIPerf results in " + sys.argv[1])
    print("| input | output | requests | TTFT p50 (s) | TTFT p99 (s) | TPOT (ms) | tok/s per user | prompt cache |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for isl, osl, j in sorted(runs):
        ttft = j["time_to_first_token"]
        cache = j.get("overall_usage_prompt_cache_read_pct", {})
        cache = cache.get("avg") if isinstance(cache, dict) else cache
        print(f"| {isl} | {osl} | {j['request_count']['avg']:.0f} | {ttft['p50'] / 1000:.2f} | {ttft['p99'] / 1000:.2f} "
              f"| {j['inter_token_latency']['avg']:.1f} | {j['output_token_throughput_per_user']['avg']:.1f} "
              f"| {'' if cache is None else f'{cache:.1f}%'} |")


if __name__ == "__main__":
    main()
