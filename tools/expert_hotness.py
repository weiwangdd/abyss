#!/usr/bin/env python3
"""Analyse routed-expert hit counts written by llama-server with ABYSS_EXPERT_STATS.

Reports how concentrated each layer's routing is, then compares two ways of filling VRAM for decode:
  layer   - whole layers on GPU (plan B: layers 0..gpu_layers-1), the rest computed by the CPU
  expert  - the encoder layers whole on GPU, the remaining budget spent on the hottest decoder experts
Both use the same VRAM budget, counted in experts.
"""

import argparse
import json
import sys


def coverage(hits, frac):
    """Number of experts that together take `frac` of a layer's hits."""
    total = sum(hits)
    if total == 0:
        return 0
    acc = 0
    for n, h in enumerate(sorted(hits, reverse=True), 1):
        acc += h
        if acc >= frac * total:
            return n
    return len(hits)


def cpu_hits_per_token(decode, on_gpu):
    """Expert hits per decode token that fall outside `on_gpu` (a set of (layer, expert))."""
    n_tok = max(decode["tokens"]) if decode["tokens"] else 0
    if n_tok == 0:
        return 0.0
    miss = 0
    for il, hits in enumerate(decode["hits"]):
        for e, h in enumerate(hits):
            if (il, e) not in on_gpu:
                miss += h
    return miss / n_tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", help="JSON file written by llama-server")
    ap.add_argument("--n-layer", type=int, default=40)
    ap.add_argument("--n-expert", type=int, default=384)
    ap.add_argument("--n-encoder", type=int, default=20, help="encoder layers, kept whole on GPU")
    ap.add_argument("--gpu-layers", type=int, default=22, help="plan B: layers whose experts sit on GPU")
    ap.add_argument("--extra-experts", type=int, default=0, help="VRAM budget beyond --gpu-layers, in experts")
    ap.add_argument("--expert-mb", type=float, default=18.8, help="bytes of one expert (MXFP4), MB")
    ap.add_argument("--cpu-gbs", type=float, default=37.0, help="effective CPU read bandwidth, GB/s")
    ap.add_argument("--gpu-ms", type=float, default=15.0, help="GPU time per decode token, ms")
    args = ap.parse_args()

    with open(args.stats) as f:
        stats = json.load(f)
    decode = stats["decode"]

    print(f"prefill tokens {max(stats['prefill']['tokens'] or [0])}, decode tokens {max(decode['tokens'] or [0])}")
    print()
    print("layer  phase    tokens   experts for 50% / 80% / 95% of hits   unused")
    for phase in ("prefill", "decode"):
        p = stats[phase]
        for il, hits in enumerate(p["hits"]):
            if not p["tokens"][il]:
                continue
            hits = hits + [0] * (args.n_expert - len(hits))
            unused = sum(1 for h in hits if h == 0)
            print(f"{il:5d}  {phase:7s} {p['tokens'][il]:8d}   "
                  f"{coverage(hits, .5):4d} / {coverage(hits, .8):4d} / {coverage(hits, .95):4d}"
                  f"{'':21s}{unused:4d}")
    print()

    budget = args.gpu_layers * args.n_expert + args.extra_experts

    layer_gpu = {(il, e) for il in range(args.gpu_layers) for e in range(args.n_expert)}
    layer_gpu |= set(
        sorted(((il, e) for il in range(args.gpu_layers, args.n_layer) for e in range(args.n_expert)),
               key=lambda k: -_hit(decode, *k))[: args.extra_experts])

    expert_gpu = {(il, e) for il in range(args.n_encoder) for e in range(args.n_expert)}
    rest = budget - len(expert_gpu)
    ranked = sorted(((il, e) for il in range(args.n_encoder, args.n_layer) for e in range(args.n_expert)),
                    key=lambda k: -_hit(decode, *k))
    expert_gpu |= set(ranked[: max(rest, 0)])

    print(f"VRAM budget {budget} experts ({budget * args.expert_mb / 1024:.1f} GB), "
          f"CPU {args.cpu_gbs:g} GB/s, GPU {args.gpu_ms:g} ms/token")
    for name, on_gpu in (("layer", layer_gpu), ("expert", expert_gpu)):
        miss = cpu_hits_per_token(decode, on_gpu)
        cpu_ms = miss * args.expert_mb / 1000 / args.cpu_gbs * 1000
        tps = 1000 / (args.gpu_ms + cpu_ms) if args.gpu_ms + cpu_ms > 0 else float("inf")
        print(f"  {name:6s}  CPU experts/token {miss:6.1f}   CPU {cpu_ms:5.1f} ms   est. {tps:5.1f} tok/s")


def _hit(p, il, e):
    if il >= len(p["hits"]):
        return 0
    hits = p["hits"][il]
    return hits[e] if e < len(hits) else 0


if __name__ == "__main__":
    sys.exit(main())
