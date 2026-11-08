#!/usr/bin/env python3
"""The numbers of one AgentX run: summary.py ARTIFACT_DIR [--json].

From AIPerf's profile_export_aiperf.json (times in ms, each metric a dict of avg/p50/p90/p99/...): requests completed
and failed, TTFT, inter-token latency and the output tokens/s per user, request latency, input and output lengths,
the system's output tokens/s, the prompt-cache read share the server reported
(usage.prompt_tokens_details.cached_tokens) and the theoretical prefix-cache hit of the traces, and whether the run is a
valid AgentX submission.
"""

import glob
import json
import os
import sys


def stat(j, key, q="avg"):
    v = j.get(key)
    if isinstance(v, dict):
        return v.get(q)
    return v if q == "avg" else None


def load(art):
    paths = glob.glob(os.path.join(art, "**", "profile_export_aiperf.json"), recursive=True)
    if not paths:
        return None
    j = json.load(open(sorted(paths)[0]))
    errors = j.get("error_summary") or []
    out = {
        "requests": stat(j, "completed_request_count") or stat(j, "request_count"),
        "errors": sum(e.get("count", 0) for e in errors) if isinstance(errors, list) else None,
        "duration_s": stat(j, "benchmark_duration"),
        "ttft_ms_median": stat(j, "time_to_first_token", "p50"), "ttft_ms_mean": stat(j, "time_to_first_token"),
        "ttft_ms_p90": stat(j, "time_to_first_token", "p90"), "ttft_ms_p99": stat(j, "time_to_first_token", "p99"),
        "itl_ms_mean": stat(j, "inter_token_latency"), "itl_ms_median": stat(j, "inter_token_latency", "p50"),
        "user_tok_s_mean": stat(j, "output_token_throughput_per_user"),
        "user_tok_s_median": stat(j, "output_token_throughput_per_user", "p50"),
        "latency_s_mean": (stat(j, "request_latency") or 0) / 1e3,
        "isl_mean": stat(j, "input_sequence_length"), "isl_max": stat(j, "input_sequence_length", "max"),
        "osl_mean": stat(j, "output_sequence_length"),
        "system_tok_s": stat(j, "output_token_throughput"),
        "cache_read_pct": stat(j, "overall_usage_prompt_cache_read_pct"),
        "theoretical_hit_pct": next((stat(j, k) for k in j if "theoretical" in k and "prefix" in k), None),
        "submission_valid": j.get("submission_valid", (j.get("metadata") or {}).get("submission_valid")),
    }
    return out


def main():
    r = load(sys.argv[1])
    if r is None:
        sys.exit(f"no profile_export_aiperf.json under {sys.argv[1]}")
    if "--json" in sys.argv:
        print(json.dumps(r))
        return
    f = lambda v, d=1: "-" if v is None else f"{v:,.{d}f}"
    s = lambda v: None if v is None else v / 1e3
    print(f"requests {f(r['requests'], 0)} ({f(r['errors'], 0)} errors) in {f(r['duration_s'], 0)} s; "
          f"system {f(r['system_tok_s'], 2)} output tok/s; submission_valid {r['submission_valid']}")
    print(f"TTFT median {f(s(r['ttft_ms_median']), 2)} s, mean {f(s(r['ttft_ms_mean']), 2)} s, "
          f"p90 {f(s(r['ttft_ms_p90']), 2)} s; ITL mean {f(r['itl_ms_mean'])} ms, "
          f"per user {f(r['user_tok_s_mean'])} tok/s (median {f(r['user_tok_s_median'])}); "
          f"latency mean {f(r['latency_s_mean'])} s")
    print(f"ISL mean {f(r['isl_mean'], 0)}, max {f(r['isl_max'], 0)}; OSL mean {f(r['osl_mean'], 0)}; "
          f"prompt cache read {f(r['cache_read_pct'])}% (theoretical prefix hit {f(r['theoretical_hit_pct'])}%)")


if __name__ == "__main__":
    main()
