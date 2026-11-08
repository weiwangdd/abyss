#!/usr/bin/env python3
"""Two AgentX runs request by request: pairs.py ARTIFACT_DIR_A ARTIFACT_DIR_B.

With the same seed both runs replay the same trace positions in the same order, but a faster server gets further in
the fixed window, so its percentiles take in requests (a new trace's cold first prefill, say) the slower one never
sent. This compares only the profiling requests both runs sent, matched by trace, outer and inner position and agent
depth: TTFT, inter-token latency (decode time over output tokens after the first), request latency, prompt tokens and
the share the server took from its prompt cache; it also prints the warmup's first request (the primer that prefills a
trace's whole prefix cold).
"""

import json
import os
import statistics
import sys


def records(art):
    out = {}
    for line in open(os.path.join(art, "profile_export.jsonl")):
        r = json.loads(line)
        m, v = r["metadata"], {k: x.get("value") for k, x in r.get("metrics", {}).items()}
        # a subagent group's requests share the outer index and differ in the inner one
        key = (m.get("source_trace_id"), m.get("source_outer_idx"), m.get("source_inner_idx"), m.get("agent_depth"),
               m.get("source_kind"))
        out.setdefault(m.get("benchmark_phase"), {})[key] = v
    return out


def itl(v):
    n = v.get("output_sequence_length") or 0
    return (v["request_latency"] - v["time_to_first_token"]) / (n - 1) if n > 1 else None


def main():
    a, b = (records(p) for p in sys.argv[1:3])
    na, nb = (os.path.basename(os.path.normpath(p)) for p in sys.argv[1:3])
    for name, recs in ((na, a), (nb, b)):
        w = recs.get("warmup", {})
        if w:
            first = next(iter(w.values()))
            print(f"{name}: warmup primer {first.get('usage_prompt_tokens')} prompt tokens, TTFT "
                  f"{first['time_to_first_token'] / 1e3:.1f} s")
    pa, pb = a.get("profiling", {}), b.get("profiling", {})
    common = [k for k in pa if k in pb]
    print(f"profiling requests: {na} {len(pa)}, {nb} {len(pb)}, both {len(common)}")
    if not common:
        return
    rows = [
        ("TTFT (s)", lambda v: v["time_to_first_token"] / 1e3),
        ("inter-token latency (ms)", itl),
        ("request latency (s)", lambda v: v["request_latency"] / 1e3),
        ("prompt tokens", lambda v: v.get("usage_prompt_tokens")),
        ("prompt cache read (%)", lambda v: 100 * v["usage_prompt_cache_read_tokens"] / v["usage_prompt_tokens"]
         if v.get("usage_prompt_tokens") else None),
        ("output tokens", lambda v: v.get("output_sequence_length")),
    ]
    print(f"| over the {len(common)} requests both sent | {na} mean | {nb} mean | {na} median | {nb} median |")
    print("|---|---:|---:|---:|---:|")
    for label, get in rows:
        xa = [x for x in (get(pa[k]) for k in common) if x is not None]
        xb = [x for x in (get(pb[k]) for k in common) if x is not None]
        print(f"| {label} | {statistics.mean(xa):,.2f} | {statistics.mean(xb):,.2f} | "
              f"{statistics.median(xa):,.2f} | {statistics.median(xb):,.2f} |")
    sa = sum(pa[k]["request_latency"] for k in common) / 1e3
    sb = sum(pb[k]["request_latency"] for k in common) / 1e3
    out = sum(pa[k].get("output_sequence_length") or 0 for k in common)
    print(f"summed request time: {na} {sa:,.0f} s, {nb} {sb:,.0f} s ({sb / sa:.2f}x); output tokens over it: "
          f"{out / sa:.1f} / {out / sb:.1f} tok/s")


if __name__ == "__main__":
    main()
