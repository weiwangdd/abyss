#!/usr/bin/env python3
"""Two nsys sqlite exports of a decode turn (decode_modes.sh CAPTURE=1, e.g. prod-decode-profile.sh without and with a
switch) compared kernel by kernel: per generated token, the count and the own time (the end minus the later of its start
and the previous kernel's end on the device, as in nsys_kernels.py, since PDL starts a kernel early) of each kernel by
device, name and grid x, the largest differences first.

    python3 nsys_kernel_diff.py A.sqlite B.sqlite [ROWS, default 16]
"""
import collections
import sqlite3
import sys


def load(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    kern = db.execute("select start, end, deviceId, shortName, gridX from CUPTI_ACTIVITY_KIND_KERNEL "
                      "order by deviceId, start").fetchall()
    # the decode window: after the last quantized matmul (batches of more than 8 tokens)
    t0 = max(e for s, e, d, n, g in kern if names.get(n) == "mul_mat_q")
    mem = db.execute("select copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start > ?", (t0,)).fetchall()
    # a token ends with the copy of its logits, the most common large device-to-host copy
    n_vocab_bytes = collections.Counter(b for k, b in mem if k == 2 and b > 400000).most_common(1)[0][0]
    n_tok = sum(1 for k, b in mem if k == 2 and b == n_vocab_bytes)
    count, own = collections.Counter(), collections.Counter()
    prev_end = {}
    for s, e, d, n, g in kern:
        if s < t0:
            continue
        pe = prev_end.get(d, 0)
        prev_end[d] = max(pe, e)
        key = (d, names.get(n, "?"), g)
        count[key] += 1
        own[key] += max(0, e - max(s, pe))
    return count, own, n_tok


def main():
    ca, oa, na = load(sys.argv[1])
    cb, ob, nb = load(sys.argv[2])
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    diff = []
    for key in set(ca) | set(cb):
        ta, tb = oa[key] / na / 1e3, ob[key] / nb / 1e3
        diff.append((tb - ta, key, ca[key] / na, cb[key] / nb, ta, tb))
    diff.sort(key=lambda r: -abs(r[0]))
    print(f"{na} and {nb} tokens; per token:")
    print(f"{'dev':>3s} {'kernel':34s} {'grid':>6s} {'count A':>8s} {'count B':>8s} {'us A':>9s} {'us B':>9s} {'B - A':>8s}")
    for d, key, na_, nb_, ta, tb in diff[:rows]:
        print(f"{key[0]:3d} {key[1][:34]:34s} {key[2]:6d} {na_:8.1f} {nb_:8.1f} {ta:9.1f} {tb:9.1f} {d:8.1f}")
    print(f"own kernel time per token: A {sum(oa.values()) / na / 1e3:.1f} us, B {sum(ob.values()) / nb / 1e3:.1f} us")


if __name__ == "__main__":
    sys.exit(main())
