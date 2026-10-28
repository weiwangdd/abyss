#!/usr/bin/env python3
"""Where a decode token waits, from an nsys sqlite export of a decode turn (decode_modes.sh CAPTURE=1): the gaps when
both GPUs are idle, grouped by the GPU events around them, with each gap split into the time the CPU spent before it
issued the CUDA call that ends the gap and the time from that call to the GPU starting; then the timeline of one layer
of one token from the first expert upload of that token (a layer of 22-39, whose experts are in host memory).

    nsys_layer.py decode.sqlite [TOKEN]

The decode window is as in nsys_decode.py; a token starts where GPU0 starts working again after an idle stretch.
"""
import bisect
import collections
import sqlite3
import statistics
import sys

GAP_NS = 5_000  # gaps shorter than this are launch jitter
COPY = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def label(kind, nbytes):
    if kind == "K":
        return "kernel"
    size = "big" if nbytes >= 1 << 20 else "small"
    return f"{kind} {size}"


def main():
    db = sqlite3.connect(sys.argv[1])
    names = dict(db.execute("select id, value from StringIds"))
    ev = []  # (start, end, device, kind, name or bytes, correlation id)
    for s, e, d, n, c in db.execute("select start, end, deviceId, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL"):
        ev.append((s, e, d, "K", names.get(n, "?"), c))
    for s, e, d, b, k, c in db.execute(
            "select start, end, deviceId, bytes, copyKind, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY"):
        ev.append((s, e, d, COPY.get(k, str(k)), b, c))
    ev.sort()
    calls = {c: (s, names.get(n, "?")) for s, n, c in db.execute(
        "select start, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}

    # the decode window: after the last quantized matmul (batches of more than 8 tokens), up to the last GPU activity
    t0 = max(e for s, e, d, k, x, c in ev if k == "K" and "mul_mat_q" == x)
    ev = [x for x in ev if x[0] >= t0]
    t1 = max(x[1] for x in ev)

    # tokens: GPU0 starts again after an idle stretch
    gpu0 = [x for x in ev if x[2] == 0]
    starts = [gpu0[0][0]]
    last_end = gpu0[0][1]
    for s, e, *_ in gpu0[1:]:
        if s - last_end > 3_000_000:
            starts.append(s)
        last_end = max(last_end, e)
    n_tok = len(starts) - 1
    print(f"decode window {(t1 - t0) / 1e9:.2f} s, {n_tok} whole tokens, {(starts[-1] - starts[0]) / n_tok / 1e6:.1f} ms each")

    # both GPUs idle: gaps in the union of all GPU activity
    by_pair = collections.defaultdict(lambda: [0, 0, 0, 0])  # count, total, cpu part, launch part
    end_so_far = None
    prev = None
    for x in ev:
        if x[0] > starts[-1]:
            break
        if end_so_far is not None and x[0] - end_so_far > GAP_NS and x[0] > starts[0]:
            gap = x[0] - end_so_far
            issued = calls.get(x[5], (None, "?"))[0]
            cpu = max(0, min(gap, (issued or x[0]) - end_so_far))
            key = (label(prev[3], prev[4]), label(x[3], x[4]), calls.get(x[5], (None, "?"))[1].split("_v")[0])
            r = by_pair[key]
            r[0] += 1
            r[1] += gap
            r[2] += cpu
            r[3] += gap - cpu
        if end_so_far is None or x[1] > end_so_far:
            end_so_far = x[1]
            prev = x
    total = sum(r[1] for r in by_pair.values())
    print(f"\nboth GPUs idle: {total / n_tok / 1e6:.2f} ms per token, by the events around the gap "
          f"(CPU = before the CUDA call that ends it, launch = from that call to the GPU)")
    print(f"  {'before':12s} {'after':12s} {'call':24s} {'per token':>9s} {'ms/token':>9s} {'CPU':>6s} {'launch':>7s}")
    for key, (n, t, c, l) in sorted(by_pair.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {key[0]:12s} {key[1]:12s} {key[2][:24]:24s} {n / n_tok:9.1f} {t / n_tok / 1e6:9.2f} "
              f"{c / n_tok / 1e6:6.2f} {l / n_tok / 1e6:7.2f}")

    # one layer of one token: from the first expert upload of the token to the next one
    k = int(sys.argv[2]) if len(sys.argv) > 2 else n_tok // 2
    lo, hi = starts[k], starts[k + 1]
    tok = [x for x in ev if lo <= x[0] < hi]
    ups = [i for i, x in enumerate(tok) if x[3] == "H2D" and x[4] >= 1 << 20]
    if not ups:
        return
    first = ups[0]
    # the next layer begins with the readback of its experts' ids; show up to its first upload
    back = next((i for i in range(first, len(tok)) if tok[i][3] == "D2H"), len(tok) - 1)
    nxt = next((i for i in ups if i > back), len(tok) - 1)
    # back to the experts' ids being read back before the first upload
    a = first
    while a > 0 and not (tok[a][3] == "D2H" and tok[first][0] - tok[a][0] < 2_000_000):
        a -= 1
    a = max(0, a - 6)
    print(f"\ntoken {k}, from the readback before its first expert upload to the next layer's upload "
          f"(times from the token's start, GPU0 then GPU1):")
    run = None
    out = []
    for x in tok[a:nxt + 1]:
        if x[3] == "K" and run and run[2] == x[2] and x[0] - run[1] < GAP_NS:
            run[1] = max(run[1], x[1])
            run[3][x[4]] += 1
            continue
        if run:
            out.append(run)
        run = [x[0], x[1], x[2], collections.Counter({x[4]: 1})] if x[3] == "K" else None
        if x[3] != "K":
            out.append([x[0], x[1], x[2], f"{x[3]} {x[4] / 2**20:.2f} MiB"])
    if run:
        out.append(run)
    last = None
    for s, e, d, what in out:
        if last is not None and s - last > GAP_NS:
            print(f"  {'':>9s} {'':>7s}       idle {(s - last) / 1e3:7.0f} us")
        if isinstance(what, collections.Counter):
            what = "kernels " + ", ".join(f"{n}x{c}" if c > 1 else n for n, c in what.most_common(4))
        print(f"  {(s - lo) / 1e6:8.3f}ms {(e - s) / 1e3:6.0f}us  GPU{d} {what[:90]}")
        last = e if last is None else max(last, e)


if __name__ == "__main__":
    main()
