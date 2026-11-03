#!/usr/bin/env python3
"""Achieved memory bandwidth of the decode's matrix-vector kernels, from the model's tensors and an nsys sqlite export
(decode_modes.sh CAPTURE=1) of the production layout.

Per token, each GPU reads the weights of its layers once: the quantized ones (Q8_0, MXFP4: mul_mat_vec_q) and the
float ones (BF16, F16, F32: mul_mat_vec_f), and of the routed experts six per layer - from VRAM for layers 0-21, from
the expert cache's slots for 22-39 (six slots per token, the zero slots of the CPU's experts included). Layers 0-11
are on GPU0 with the gate and up experts of layer 12, the rest and the output head on GPU1. The bytes over the
kernels' time per token give the bandwidth each kernel class reaches, against the GPU's peak.

    ~/src/comfyui/.venv/bin/python kernel_bw.py <model dir> <sqlite> [tokens, default 256] [peak GB/s, default 1792]
"""

import collections
import glob
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.expanduser("~/src/mx-llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

N_EXPERT, N_USED = 384, 6
QUANT = {"Q8_0", "MXFP4", "Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ4_XS"}
FLOAT = {"BF16", "F16", "F32"}


def main():
    model_dir, sqlite_path = sys.argv[1:3]
    n_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    peak = float(sys.argv[4]) if len(sys.argv) > 4 else 1792.0

    # bytes read per token, per (device, class, part)
    per = collections.defaultdict(float)
    for shard in sorted(glob.glob(os.path.join(model_dir, "*-of-*.gguf"))):
        for t in GGUFReader(shard).tensors:
            name, typ, nbytes = t.name, t.tensor_type.name, int(t.n_bytes)
            m = re.match(r"blk\.(\d+)\.(.*)", name)
            if m is None:
                if name.startswith("output.") or name.startswith("output_norm"):
                    dev, part = 1, "head"
                else:
                    continue  # token embedding (host), others not read per token
                il = -1
            else:
                il, rest = int(m.group(1)), m.group(2)
                part = "dense"
                if re.match(r"ffn_(gate|up|down)_exps\.", rest):
                    part = "experts"
                    nbytes = nbytes * N_USED // N_EXPERT
                if rest.startswith("engram"):
                    continue  # the engram tables are host side
                dev = 0 if il < 12 else 1
                if il == 12 and re.match(r"ffn_(gate|up)_exps\.", rest):
                    dev = 0
            if len(t.shape) < 2:
                cls = "vector"
            else:
                cls = "q" if typ in QUANT else "f" if typ in FLOAT else "other:" + typ
            per[(dev, cls, part)] += nbytes

    db = sqlite3.connect(sqlite_path)
    kern = db.execute("select k.deviceId, k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on k.shortName = s.id").fetchall()
    t0 = max(k[2] for k in kern if k[3] == "mul_mat_q")
    t_kind = collections.defaultdict(float)
    for d, s, e, name in kern:
        if s > t0:
            kind = "q" if name.startswith("mul_mat_vec_q") else "f" if name.startswith("mul_mat_vec_f") else "rest"
            t_kind[(d, kind)] += (e - s) / n_tok / 1e6  # ms per token

    for dev in (0, 1):
        print(f"GPU{dev}:")
        for cls in ("q", "f"):
            b = {part: per[(dev, cls, part)] for part in ("dense", "experts", "head")}
            tot = sum(b.values())
            ms = t_kind[(dev, cls)]
            bw = tot / (ms / 1e3) / 1e9 if ms else 0
            print(f"  mul_mat_vec_{cls}: {tot / 2**20:8.1f} MiB/token (dense {b['dense'] / 2**20:.0f}, experts "
                  f"{b['experts'] / 2**20:.0f}, head {b['head'] / 2**20:.0f}) in {ms:.2f} ms/token: {bw:6.0f} GB/s, "
                  f"{100 * bw / peak:.0f}% of {peak:.0f}; at 80% of peak {tot / (0.8 * peak * 1e9) * 1e3:.2f} ms")
        others = {k: v for k, v in per.items() if k[0] == dev and k[1] not in ("q", "f")}
        if others:
            print("  not matrix-vector: " + ", ".join(f"{k[1]} {k[2]} {v / 2**20:.1f} MiB" for k, v in sorted(others.items())))
        print(f"  other kernels {t_kind[(dev, 'rest')]:.2f} ms/token")


if __name__ == "__main__":
    sys.exit(main())
