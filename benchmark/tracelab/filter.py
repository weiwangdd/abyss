#!/usr/bin/env python3
"""The TraceLab corpus without the session trees that do not fit the server: filter.py IN OUT [--max N].

AIPerf's TraceLab reader can drop traces whose recorded peak context exceeds --max-context-length, but its command line
still takes that flag for Weka traces only. This runs the reader's own steps (aiperf.dataset.loader: rounds grouped by
session_id, subagent sessions joined to their parents by its timing-containment rule, one trace per root with its nested
children, the peak of _trace_peak_context_length) and writes the rows of the trees whose peak is at most N (default
190,000) to OUT in the original order, so the unmodified client, reading OUT, replays the traces it would have kept.
Dropping whole trees keeps every remaining link as it was. Run it with the agx venv's python.
"""

import argparse
import collections
import gzip
import json

from aiperf.common.environment import Environment
from aiperf.dataset.loader._tracelab_convert import build_join_index, build_trace, group_children_by_parent, safe_trace_id
from aiperf.dataset.loader.tracelab_trace import DEFAULT_BLOCK_SIZE
from aiperf.dataset.loader.weka_trace import _trace_peak_context_length
from aiperf.dataset.loader.weka_trace_models import WekaTrace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--max", type=int, default=190000)
    a = ap.parse_args()

    sessions = collections.defaultdict(list)
    order = []
    with gzip.open(a.input, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if sid := row.get("session_id"):
                    sessions[sid].append(row)
                    order.append((sid, line))
    sessions = dict(sessions)

    children = {}
    if Environment.DATASET.TRACELAB_SUBAGENT_JOIN:
        links, stats = build_join_index(sessions, min_spawn_ms=Environment.DATASET.TRACELAB_MIN_SPAWN_MS,
                                        enable_codex=Environment.DATASET.TRACELAB_CODEX_SUBAGENT_JOIN)
        children = group_children_by_parent(sessions, links, stats)
        print(f"subagent join: {stats.summary()}")
    nested = {c for kids in children.values() for c in kids}

    # as TraceLabTraceDatasetLoader._build_traces: roots with their children, then the children that failed to nest
    placed = set()
    trees = []
    for sid, rows in sessions.items():
        if sid not in nested:
            before = set(placed)
            blob = build_trace(sid, rows, DEFAULT_BLOCK_SIZE, children.get(sid), placed_sids=placed)
            if blob and blob["requests"]:
                trees.append((sid, blob, {sid} | (placed - before)))
    for sid in sessions:
        if sid in nested - placed:
            blob = build_trace(sid, sessions[sid], DEFAULT_BLOCK_SIZE)
            if blob and blob["requests"]:
                trees.append((sid, blob, {sid}))

    keep = set()
    peaks = collections.Counter()
    kept = collections.Counter()
    for sid, blob, members in trees:
        blob["id"] = safe_trace_id(sid)
        peak = _trace_peak_context_length(WekaTrace.model_validate(blob))
        provider = sid.split(":", 1)[0]
        peaks[provider] += 1
        if peak <= a.max:
            keep |= members
            kept[provider] += 1

    n = 0
    with gzip.open(a.output, "wt", encoding="utf-8") as f:
        for sid, line in order:
            if sid in keep:
                f.write(line)
                n += 1
    print(f"{len(sessions)} sessions in {len(trees)} traces; kept {sum(kept.values())} traces with a peak of at most "
          f"{a.max:,} tokens ({', '.join(f'{p} {kept[p]} of {peaks[p]}' for p in sorted(peaks))}), "
          f"{len(keep)} sessions, {n} of {len(order)} rounds")


if __name__ == "__main__":
    main()
