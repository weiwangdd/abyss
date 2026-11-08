#!/usr/bin/env python3
"""The numbers of one MLPerf Agentic Inference run: summary.py REPORT_DIR [--json].

From performance/result_summary.json (times in ns) and scores.json: turns issued, completed and failed, TTFT and
time per output token (median, mean, p90, p99), end-to-end latency, input and output lengths, the system's output
tokens/s, the end-to-end interactivity (output tokens over the summed turn times) and the inline accuracy score.
"""

import json
import os
import sys


def load(report):
    s = json.load(open(os.path.join(report, "performance", "result_summary.json")))
    ms = lambda v: None if v is None else v / 1e6
    p = lambda m, q: ms((s.get(m) or {}).get("percentiles", {}).get(q))
    out = {
        "turns": s.get("n_samples_issued"), "completed": s.get("n_samples_completed"),
        "failed": s.get("n_samples_failed"), "duration_s": s["duration_ns"] / 1e9,
        "ttft_ms_median": ms(s["ttft"]["median"]), "ttft_ms_mean": ms(s["ttft"]["avg"]),
        "ttft_ms_p90": p("ttft", "90.0"), "ttft_ms_p99": p("ttft", "99.0"),
        "tpot_ms_median": ms(s["tpot"]["median"]), "tpot_ms_mean": ms(s["tpot"]["avg"]),
        "tpot_ms_p90": p("tpot", "90.0"),
        "latency_s_mean": s["latency"]["avg"] / 1e9,
        "isl_mean": s["input_sequence_lengths"]["avg"], "isl_max": s["input_sequence_lengths"]["max"],
        "osl_mean": (s.get("output_sequence_lengths_full_run") or {}).get("output_sequence_lengths", {}).get("avg"),
        "system_tok_s": s.get("tps"), "interactivity_tok_s": s.get("e2e_avg_interactivity"),
        "complete": s.get("complete"),
    }
    sc = os.path.join(report, "scores.json")
    if os.path.exists(sc):
        j = json.load(open(sc))
        out["accuracy"] = j.get("score")
        out["accuracy_turns"] = (j.get("turns") or {}).get("scored")
    return out


def main():
    r = load(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(r))
        return
    f = lambda v, d=1: "-" if v is None else f"{v:,.{d}f}"
    print(f"turns {r['turns']} ({r['completed']} completed, {r['failed']} failed) in {f(r['duration_s'], 0)} s; "
          f"system {f(r['system_tok_s'], 2)} output tok/s, interactivity {f(r['interactivity_tok_s'], 2)} tok/s")
    print(f"TTFT median {f(r['ttft_ms_median'] / 1e3, 2)} s, mean {f(r['ttft_ms_mean'] / 1e3, 2)} s, "
          f"p90 {f(r['ttft_ms_p90'] / 1e3 if r['ttft_ms_p90'] else None, 2)} s; "
          f"TPOT median {f(r['tpot_ms_median'])} ms ({f(1e3 / r['tpot_ms_median'])} tok/s), "
          f"mean {f(r['tpot_ms_mean'])} ms; "
          f"latency mean {f(r['latency_s_mean'], 1)} s")
    print(f"ISL mean {f(r['isl_mean'], 0)}, max {f(r['isl_max'], 0)}; OSL mean {f(r['osl_mean'], 0)}"
          + (f"; inline accuracy {f(r['accuracy'], 3)} over {r['accuracy_turns']} turns" if "accuracy" in r else ""))


if __name__ == "__main__":
    main()
