#!/usr/bin/env python3
"""The CUDA graph launches of a decode capture (decode_modes.sh CAPTURE=1, with --cuda-graph-trace node or graph):
per token over the window from the first graph launch to the last (the prefill before it runs no graphs), the launches
and the host time inside cudaGraphLaunch, the time from a launch call to its graph starting on a GPU that had nothing
else to do, and how long both GPUs are idle. With node tracing, also the nodes per graph and a fit of the host time of
a launch on them, since a launch whose cost grows with its nodes shrinks with fewer kernels.

    nsys_graphs.py <sqlite> [tokens, default 256]
"""

import collections
import sqlite3
import sys


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def union(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


db = sqlite3.connect(sys.argv[1])
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
names = dict(db.execute("select id, value from StringIds"))
launch_ids = [i for i, v in names.items() if v.startswith("cudaGraphLaunch")]
calls = db.execute(f"select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME "
                   f"where nameId in ({','.join(map(str, launch_ids))}) order by start").fetchall()
w0, w1 = calls[0][0], calls[-1][1]
span = (w1 - w0) / 1e6
corr_calls = {c: (s, e) for s, e, c in calls}

# the device activities in the window: graphs as a whole (graph tracing) or their nodes (node tracing), and the rest
busy = collections.defaultdict(list)
execs = {}
nodes = collections.Counter()
for tab in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET",
            "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
    if tab not in tables:
        continue
    for s, e, dev, corr in db.execute(f"select start, end, deviceId, correlationId from {tab} where end > ? and start < ?",
                                      (w0, w1)):
        busy[dev].append((s, e))
        if corr in corr_calls:
            g = execs.get(corr)
            execs[corr] = (min(g[0], s) if g else s, max(g[1], e) if g else e, dev)
            if tab != "CUPTI_ACTIVITY_KIND_GRAPH_TRACE":
                nodes[corr] += 1
mode = "graph" if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables else "node"

host = [(e - s) / 1e3 for s, e, _ in calls]
print(f"{mode} tracing: {len(calls)} graph launches over {span:.1f} ms, {span / tokens:.2f} ms per token "
      f"({tokens} tokens), {len(calls) / tokens:.1f} launches per token")
print(f"  host time in cudaGraphLaunch: mean {sum(host) / len(host):.1f} us, median {pct(host, .5):.1f}, "
      f"p90 {pct(host, .9):.1f}; {sum(host) / 1e3 / tokens:.2f} ms per token")

# from the call to the graph's first activity, for graphs whose device had nothing running between the two
lat_end, lat_start = [], []
by_dev = {d: sorted(v) for d, v in busy.items()}
for c, (gs, ge, dev) in execs.items():
    cs, ce = corr_calls[c]
    if any(s < gs and e > ce for s, e in by_dev[dev] if s < gs):
        continue
    lat_start.append((gs - cs) / 1e3)
    lat_end.append((gs - ce) / 1e3)
if lat_start:
    print(f"  on an idle GPU ({len(lat_start)} of {len(execs)} graphs): first activity {pct(lat_start, .5):.1f} us after "
          f"the call starts (p90 {pct(lat_start, .9):.1f}), {pct(lat_end, .5):.1f} us after it returns (p90 "
          f"{pct(lat_end, .9):.1f})")

for dev in sorted(busy):
    print(f"  GPU{dev}: busy {union(busy[dev]) / 1e6 / tokens:.2f} ms per token")
both = (w1 - w0) - union([iv for v in busy.values() for iv in v])
print(f"  both GPUs idle: {both / 1e6 / tokens:.2f} ms per token")

if mode == "node" and nodes:
    pts = [(nodes[c], (corr_calls[c][1] - corr_calls[c][0]) / 1e3) for c in nodes]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx else 0
    a = my - b * mx
    print(f"  nodes per graph: median {pct([x for x, _ in pts], .5)}, max {max(x for x, _ in pts)}, "
          f"{sum(x for x, _ in pts) / tokens:.0f} per token; host time of a launch = {a:.1f} us + {b:.3f} us x nodes")
    for lo, hi in ((0, 20), (20, 60), (60, 150), (150, 400), (400, 10**6)):
        g = [y for x, y in pts if lo <= x < hi]
        if g:
            print(f"    {lo:>3}-{hi if hi < 10**6 else 'more':>4} nodes: {len(g) / tokens:5.1f} launches per token, "
                  f"median {pct(g, .5):.1f} us")
