#!/usr/bin/env python3
"""What a graph's compute buffers hold at their peak, from a server log with GGML_ALLOC_TRACE=1 (the abyss trace in
ggml-alloc.c): the buffer sizes of each reserve, and for the one with the most nodes (or the RESERVE-th, from 0) the
tensors live when each buffer's live bytes peak, grouped by name without the layer suffix.

    alloc_peak.py LOG [TOP [RESERVE]]
"""
import collections
import re
import sys

RESERVE = re.compile(r"galloc: reserve (\d+) nodes (\d+) leafs:(.*)")
EVENT = re.compile(r"galloc: (alloc|free) (\d+) (\d+) (\d+) (\d+) (.*)")
BUFFER = re.compile(r"galloc: buffer (\d+) (\S+) max (\d+)")


def segments(lines):
    seg = None
    for line in lines:
        m = RESERVE.search(line)
        if m:
            seg = {"nodes": int(m.group(1)), "names": dict(x.split("=", 1) for x in m.group(3).split()), "events": [],
                   "max": {}}
            yield seg
            continue
        if seg is None:
            continue
        m = EVENT.search(line)
        if m:
            seg["events"].append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)),
                                  m.group(6).strip()))
            continue
        m = BUFFER.search(line)
        if m:
            seg["max"][m.group(1)] = int(m.group(3))
            if len(seg["max"]) == len(seg["names"]):
                seg = None  # the allocations after a reserve belong to graph computes, not to it


def group(name):
    return re.sub(r"-\d+$", "", name) or "(unnamed)"


def main():
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    segs = list(segments(open(sys.argv[1], errors="replace")))
    if not segs:
        sys.exit("no GGML_ALLOC_TRACE reserve in the log")
    print(f"{len(segs)} reserves (nodes: buffer sizes in MiB):")
    for s in segs:
        print(f"  {s['nodes']:6d}: " + " ".join(f"{s['names'][b]} {m / 2**20:.0f}" for b, m in sorted(s["max"].items())))
    if len(sys.argv) > 3:
        seg = segs[int(sys.argv[3])]
        print(f"reserve {sys.argv[3]} has {seg['nodes']} nodes")
    else:
        seg = max(segs, key=lambda s: s["nodes"])
        print(f"the largest has {seg['nodes']} nodes")
    live = collections.defaultdict(dict)  # buffer -> (chunk, offset) -> (size, name)
    cur = collections.Counter()
    peak = collections.Counter()
    at_peak = {}
    for kind, buf, chunk, off, size, name in seg["events"]:
        key = (chunk, off)
        if kind == "alloc":
            live[buf][key] = (size, name)
            cur[buf] += size
            if cur[buf] > peak[buf]:
                peak[buf] = cur[buf]
                at_peak[buf] = dict(live[buf])
        elif key in live[buf]:
            cur[buf] -= live[buf].pop(key)[0]
    for buf_s, name in sorted(seg["names"].items(), key=lambda x: int(x[0])):
        buf = int(buf_s)
        size = seg["max"].get(buf_s, 0)
        print(f"\nbuffer {buf} {name}: size {size / 2**20:.0f} MiB, live peak {peak[buf] / 2**20:.0f} MiB")
        groups = collections.Counter()
        count = collections.Counter()
        for sz, nm in at_peak.get(buf, {}).values():
            groups[group(nm)] += sz
            count[group(nm)] += 1
        for g, sz in groups.most_common(top):
            print(f"  {sz / 2**20:8.0f} MiB  {count[g]:4d} x  {g}")


if __name__ == "__main__":
    main()
