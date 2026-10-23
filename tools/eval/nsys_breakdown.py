#!/usr/bin/env python3
"""Summarise an nsys sqlite export: GPU busy and idle time per device, kernel time by category, copies by kind."""

import collections
import re
import sqlite3
import sys

CATEGORIES = [
    ("matmul MMQ (quantized)", r"mul_mat_q|mmq"),
    ("matmul vector (MMVQ/MMVF)", r"mul_mat_vec|mmvq|mmvf"),
    ("cuBLAS GEMM", r"gemm|cutlass|cublas|xmma|Kernel2"),
    ("attention", r"flash_attn|fattn|attn|lightning|indexer"),
    ("softmax", r"soft_max"),
    ("get_rows / set_rows", r"get_rows|set_rows"),
    ("quantize activations", r"quantize"),
    ("copy / convert", r"cpy|convert|dup|cont\b|concat"),
    ("norm", r"norm"),
    ("rope", r"rope"),
    ("routing", r"argsort|top_k|topk"),
    ("moe reduce", r"moe"),
    ("elementwise", r"add|mul|scale|silu|swiglu|glu|sigmoid|sqrt|clamp|fill|unary|bin_bcast|sum|sgn|abs|repeat"),
]
COPY_KINDS = {1: "host->device", 2: "device->host", 8: "device->device", 10: "peer->peer"}


def category(name):
    for cat, pat in CATEGORIES:
        if re.search(pat, name, re.I):
            return cat
    return "other"


def union(intervals):
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def main():
    db = sqlite3.connect(sys.argv[1])
    tables = {r[0] for r in db.execute("select name from sqlite_master where type = 'table'")}
    kern = db.execute("select k.deviceId, k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on k.shortName = s.id").fetchall()
    mem = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        mem = db.execute("select deviceId, copyKind, bytes, start, end from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()

    t0 = min([k[1] for k in kern] + [m[3] for m in mem])
    t1 = max([k[2] for k in kern] + [m[4] for m in mem])
    wall = (t1 - t0) / 1e9
    print(f"GPU activity window: {wall:.2f} s")

    devices = sorted({k[0] for k in kern} | {m[0] for m in mem})
    every = []
    for d in devices:
        ks = [(k[1], k[2]) for k in kern if k[0] == d]
        ms = [(m[3], m[4]) for m in mem if m[0] == d]
        every += ks + ms
        print(f"device {d}: kernels {union(ks) / 1e9:6.2f} s, copies {union(ms) / 1e9:6.2f} s, "
              f"busy {union(ks + ms) / 1e9:6.2f} s, idle {wall - union(ks + ms) / 1e9:6.2f} s")
    print(f"both devices idle: {wall - union(every) / 1e9:.2f} s")

    for d in devices:
        cats = collections.Counter()
        names = collections.Counter()
        for dev, s, e, name in kern:
            if dev == d:
                cats[category(name)] += e - s
                names[name] += e - s
        print(f"\ndevice {d} kernel time by category:")
        for cat, ns in cats.most_common():
            print(f"  {cat:28s} {ns / 1e9:7.2f} s")
        print(f"device {d} top kernels:")
        for name, ns in names.most_common(12):
            print(f"  {ns / 1e9:7.2f} s  {name[:90]}")

    if mem:
        print("\ncopies:")
        agg = collections.defaultdict(lambda: [0, 0, 0])
        for d, kind, nbytes, s, e in mem:
            a = agg[(d, COPY_KINDS.get(kind, str(kind)))]
            a[0] += 1
            a[1] += nbytes
            a[2] += e - s
        for (d, kind), (n, nbytes, ns) in sorted(agg.items()):
            print(f"  device {d} {kind:15s} {n:7d} copies {nbytes / 2**30:8.2f} GiB {ns / 1e9:7.2f} s")


if __name__ == "__main__":
    sys.exit(main())
