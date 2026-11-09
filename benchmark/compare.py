#!/usr/bin/env python3
"""The servers of one run.sh side by side: compare.py OUT_DIR SERVER [SERVER ...] prints a Markdown table per
benchmark (mlperf-SERVER, agentx-SERVER, tracelab-SERVER under OUT_DIR, read by each benchmark's summary.py; TraceLab
by AgentX's), with the first server over the second where there are two."""

import importlib.util
import os
import sys

B = os.path.dirname(os.path.abspath(__file__))


def module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(B, name, "summary.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ROWS = {
    "mlperf": [
        ("turns completed / failed", lambda r: f"{r['completed']} / {r['failed']}", None),
        ("TTFT median (s)", lambda r: r["ttft_ms_median"] / 1e3, "low"),
        ("TTFT mean (s)", lambda r: r["ttft_ms_mean"] / 1e3, "low"),
        ("TTFT p90 (s)", lambda r: r["ttft_ms_p90"] / 1e3, "low"),
        ("time per output token, median (ms)", lambda r: r["tpot_ms_median"], "low"),
        ("output tok/s per user (1 / TPOT median)", lambda r: 1e3 / r["tpot_ms_median"], "high"),
        ("turn latency mean (s)", lambda r: r["latency_s_mean"], "low"),
        ("system output tok/s", lambda r: r["system_tok_s"], "high"),
        ("E2E interactivity (tok/s)", lambda r: r["interactivity_tok_s"], "high"),
        ("input tokens per turn, mean", lambda r: r["isl_mean"], None),
        ("output tokens per turn, mean", lambda r: r["osl_mean"], None),
        ("run time (s)", lambda r: r["duration_s"], "low"),
        ("inline accuracy", lambda r: r.get("accuracy"), None),
    ],
    "agentx": [
        ("requests in the window / errors", lambda r: f"{r['requests']:.0f} / {r['errors']}", None),
        ("TTFT median (s)", lambda r: r["ttft_ms_median"] / 1e3, "low"),
        ("TTFT p90 (s)", lambda r: r["ttft_ms_p90"] / 1e3, "low"),
        ("TTFT p99 (s)", lambda r: r["ttft_ms_p99"] / 1e3, "low"),
        ("output tok/s per user, mean", lambda r: r["user_tok_s_mean"], "high"),
        ("inter-token latency mean (ms)", lambda r: r["itl_ms_mean"], "low"),
        ("request latency mean (s)", lambda r: r["latency_s_mean"], "low"),
        ("system output tok/s", lambda r: r["system_tok_s"], "high"),
        ("prompt cache read (%)", lambda r: r["cache_read_pct"], None),
        ("input tokens per request, mean", lambda r: r["isl_mean"], None),
        ("output tokens per request, mean", lambda r: r["osl_mean"], None),
    ],
}
# TraceLab is replayed by the AgentX harness, the same artifacts
ROWS["tracelab"] = ROWS["agentx"]
SUMMARY = {"tracelab": "agentx"}


def cell(v):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}" if abs(v) < 100 else f"{v:,.1f}"


def main():
    out, servers = sys.argv[1], sys.argv[2:]
    for bench, rows in ROWS.items():
        dirs = {s: os.path.join(out, f"{bench}-{s}") for s in servers}
        results = {s: module(SUMMARY.get(bench, bench)).load(d) if os.path.isdir(d) else None for s, d in dirs.items()}
        if not any(results.values()):
            continue
        print(f"\n### {bench}\n")
        ratio = len(servers) == 2 and all(results.values())
        print("| | " + " | ".join(servers) + (f" | {servers[0]} / {servers[1]} |" if ratio else " |"))
        print("|---|" + "---:|" * (len(servers) + ratio))
        for label, get, _ in rows:
            vals = []
            for s in servers:
                try:
                    vals.append(get(results[s]) if results[s] else None)
                except (TypeError, KeyError, ZeroDivisionError):
                    vals.append(None)
            line = f"| {label} | " + " | ".join(cell(v) for v in vals)
            if ratio:
                a, b = vals
                both = isinstance(a, (int, float)) and isinstance(b, (int, float)) and b
                line += " | " + (f"{a / b:.2f}" if both else "")
            print(line + " |")


if __name__ == "__main__":
    main()
