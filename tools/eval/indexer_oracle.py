#!/usr/bin/env python3
"""Check the fork's V4.1 sparse indexer against a reference written from DeepSeek's inference/model.py and kernel.py.

llama-abyss-dump saves, for one prompt, the indexer inputs (qr_norm, attn_norm, idx_latent) and outputs (idx_q, idx_w,
idx_k, idx_score, idx_top_k) of a few layers. This recomputes q, w, k, the scores and the top-k from the dumped inputs
with the GGUF weights three ways and compares the fork with each:

  float     float32 throughout, the algorithm without rounding other than FP4
  int8      the fork's arithmetic: the Q8_0 matmul quantizes its activations to int8 (blocks of 32)
  official  model.py's arithmetic: bf16 activations, quantized to e4m3 with power-of-2 scales for the fp8 GEMM

The fork should match "int8" up to float summation order; "official" vs "float" is the spread the reference
implementation itself has, the yardstick for the fork's distance from "float".

    PYTHONPATH=<mx-llama.cpp>/gguf-py python indexer_oracle.py <dump dir> <gguf dir> --layers 2 20 24
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

# the released config (inference/config.json)
CFG = dict(head_dim=128, n_heads=32, topk=512, rope_dim=64, rope_theta=160000.0, original_seq_len=65536,
           rope_factor=16.0, beta_fast=32, beta_slow=1, kv_sources=[2, 8, 14, 20])
GGML_DTYPES = {0: np.float32, 1: np.float16, 26: np.int32}


def load(d, name):
    """every evaluation of a dumped tensor, in order, shaped like its ggml ne reversed"""
    out = []
    while os.path.exists(path := f"{d}/{name}.{len(out)}.bin"):
        with open(path, "rb") as f:
            assert f.read(4) == b"ADMP", path
            t = int(np.frombuffer(f.read(4), np.int32)[0])
            ne = np.frombuffer(f.read(32), np.int64)
            out.append(np.frombuffer(f.read(), GGML_DTYPES[t]).reshape(tuple(ne[::-1])))
    return out


class Weights:
    def __init__(self, gguf_dir):
        self.tensors = {}
        self.fields = {}
        for p in sorted(glob.glob(f"{gguf_dir}/*-of-*.gguf")):
            r = GGUFReader(p)
            for t in r.tensors:
                self.tensors[t.name] = t
            for k, v in r.fields.items():
                self.fields.setdefault(k, v)

    def find(self, layer, *parts):
        names = [n for n in self.tensors if n.startswith(f"blk.{layer}.") and all(p in n for p in parts)]
        if len(names) != 1:
            sys.exit(f"layer {layer} {parts}: {names}")
        return names[0]

    def get(self, name):
        t = self.tensors[name]
        a = dequantize(t.data, t.tensor_type).astype(np.float32)
        return torch.from_numpy(a.reshape(tuple(int(x) for x in reversed(t.shape))).copy())

    def scalar(self, key):
        f = self.fields[key]
        return f.parts[f.data[0]][0].item()


# ---- reference pieces, as in inference/model.py and kernel.py

def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope(x, freqs):
    """rotate the last dim of x [n, ..., d] by freqs [n, d/2], adjacent pairs as complex numbers"""
    c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.view(freqs.shape[0], *([1] * (c.ndim - 2)), freqs.shape[-1])
    return torch.view_as_real(c * f).flatten(-2)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp4_qdq(x):
    """fp4_act_quant(x, 32, inplace=True): power-of-two scale 2^ceil(log2(amax/6)), e2m1 round to nearest even"""
    shape = x.shape
    b = x.reshape(-1, 32).float()
    amax = b.abs().amax(dim=1, keepdim=True).clamp(min=6 * 2.0 ** -126)
    m, e = torch.frexp(amax / 6)  # amax/6 = m * 2^e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)  # an exact power of two keeps its exponent
    s = torch.ldexp(torch.ones_like(amax), e)
    y = (b / s).clamp(-6, 6)
    a = y.abs().unsqueeze(-1)
    d = (a - E2M1).abs()
    best = d.min(dim=-1, keepdim=True).values
    # ties go to the even code, which on this grid is the even index
    cand = (d == best) & (torch.arange(8) % 2 == 0) | (d == best) & ((d == best).sum(-1, keepdim=True) == 1)
    idx = torch.where(cand, torch.arange(8), 8).min(dim=-1).values
    q = E2M1[idx] * torch.sign(y)
    return (q * s).reshape(shape)


def q8_1(x):
    """the int8 activations of the fork's quantized matmuls: blocks of 32, scale amax/127, round to nearest"""
    shape = x.shape
    b = x.reshape(-1, 32).float()
    d = b.abs().amax(dim=1, keepdim=True) / 127
    q = torch.where(d > 0, torch.round(b / torch.where(d > 0, d, 1)), 0).clamp(-127, 127)
    return (q * d).reshape(shape)


def act_fp8(x):
    """inference/kernel.py act_quant as model.py's linear() uses it: blocks of 32, e4m3, power-of-2 scales"""
    shape = x.shape
    b = x.reshape(-1, 32).float()
    amax = b.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
    sc = torch.exp2(torch.ceil(torch.log2(amax * (1 / 448))))
    return ((b / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc).reshape(shape)


def bf16(x):
    return x.bfloat16().float()


def rms_norm(x, w, eps):
    return w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


def rel(a, b):
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("gguf")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 20, 24])
    args = ap.parse_args()

    W = Weights(args.gguf)
    eps = W.scalar("deepseek41.attention.layer_norm_rms_epsilon")
    ratios = [int(x) for x in W.fields["deepseek41.attention.compress_ratios"].contents()]
    print(f"rms eps {eps}, ratios {ratios[:4]}..{ratios[18:22]}")

    freqs = precompute_freqs_cis(CFG["rope_dim"], 1 << 17, CFG["original_seq_len"], CFG["rope_theta"],
                                 CFG["rope_factor"], CFG["beta_fast"], CFG["beta_slow"])
    keys = {}
    for layer in sorted(set(args.layers) | {max(s for s in CFG["kv_sources"] if s <= l) for l in args.layers}):
        if layer not in CFG["kv_sources"]:
            continue
        ratio = ratios[layer]
        wk = W.get(W.find(layer, "indexer_compressor_kv"))
        wn = W.get(W.find(layer, "indexer_compressor_norm"))
        k_ref, k_fork = [], []
        for latent, k in zip(load(args.dump, f"idx_latent-{layer}"), load(args.dump, f"idx_k-{layer}")):
            lat = torch.from_numpy(latent.reshape(-1, latent.shape[-1]).copy())
            j0 = sum(x.shape[0] for x in k_ref)
            kk = rms_norm(lat @ wk.T, wn, eps)
            pos = (torch.arange(lat.shape[0]) + j0) * ratio
            kk = torch.cat([kk[:, :-CFG["rope_dim"]], rope(kk[:, -CFG["rope_dim"]:], freqs[pos])], dim=-1)
            k_ref.append(fp4_qdq(kk))
            k_fork.append(torch.from_numpy(k.reshape(-1, k.shape[-1]).copy()))
        keys[layer] = (torch.cat(k_ref), torch.cat(k_fork))
        # a trailing partial group gets a key too, but no query sees it until the group completes
        n = sum(x.reshape(-1, x.shape[-1]).shape[0] for x in load(args.dump, f"attn_norm-{layer}")) // ratio
        kr, kf = keys[layer][0][:n], keys[layer][1][:n]
        print(f"layer {layer:2d} keys:   {n} complete groups, max rel err {rel(kf, kr):.1e}, "
              f"identical values {(kf == kr).float().mean().item():.5f}")

    def qk_rope(q, pos):
        return torch.cat([q[..., :-CFG["rope_dim"]], rope(q[..., -CFG["rope_dim"]:], freqs[pos])], dim=-1)

    w_scale = CFG["head_dim"] ** -0.5 * CFG["n_heads"] ** -0.5
    for layer in args.layers:
        ratio = ratios[layer]
        src = max(s for s in CFG["kv_sources"] if s <= layer)
        wqb = W.get(W.find(layer, "indexer", "attn_q_b"))
        wpr = W.get(W.find(layer, "indexer", "proj"))
        kr_all = keys[src][0]

        st = {}
        def add(key, v):
            st.setdefault(key, []).append(v)

        p0 = 0
        for qr, x, q, w, score, top in zip(*(load(args.dump, f"{n}-{layer}") for n in
                                             ("qr_norm", "attn_norm", "idx_q", "idx_w", "idx_score", "idx_top_k"))):
            qr = torch.from_numpy(qr.reshape(-1, qr.shape[-1]).copy())
            x = torch.from_numpy(x.reshape(-1, x.shape[-1]).copy())
            nt = qr.shape[0]
            pos = torch.arange(p0, p0 + nt)
            shape = (nt, CFG["n_heads"], CFG["head_dim"])

            q_ref = {
                "float": fp4_qdq(qk_rope((qr @ wqb.T).view(shape), pos)),
                "int8": fp4_qdq(qk_rope((q8_1(qr) @ wqb.T).view(shape), pos)),
                "official": bf16(fp4_qdq(bf16(qk_rope(bf16(act_fp8(bf16(qr)) @ wqb.T).view(shape), pos)))),
            }
            w_ref = {"float": (x @ wpr.T) * w_scale, "official": bf16(bf16(x) @ bf16(wpr).T) * w_scale}
            w_ref["int8"] = w_ref["float"]

            q_f = torch.from_numpy(q.reshape(shape).copy())
            w_f = torch.from_numpy(w.reshape(nt, CFG["n_heads"]).copy())
            for r in ("float", "int8"):
                add(f"q_same_{r}", (q_f == q_ref[r]).float().mean().item())
            add("w", rel(w_f, w_ref["float"]))

            visible = (pos + 1) // ratio
            n_vis = int(visible.max())
            kr = kr_all[:n_vis]
            vis_mask = torch.arange(n_vis)[None, :] < visible[:, None]
            s_f = torch.from_numpy(score.reshape(nt, -1)[:, :n_vis].copy().astype(np.float32))
            s_ref = {}
            for r in q_ref:
                s_ref[r] = (torch.relu(torch.einsum("shd,td->sht", q_ref[r], kr)) * w_ref[r].unsqueeze(-1)).sum(1)
                s_ref[r] = s_ref[r].masked_fill(~vis_mask, -torch.inf)
            add("score", rel(s_f[vis_mask], s_ref["float"][vis_mask]))

            top_f = torch.from_numpy(top.reshape(nt, -1).copy().astype(np.int64))
            for i in range(nt):
                v = int(visible[i])
                if v == 0:
                    continue
                k = min(CFG["topk"], v)
                sets = {r: set(s_ref[r][i].topk(k).indices.tolist()) for r in s_ref}
                sets["fork"] = set(t for t in top_f[i].tolist() if t < v)
                sets["fork scores"] = set(s_f[i, :v].topk(k).indices.tolist())
                for a, b in (("fork", "fork scores"), ("fork", "int8"), ("fork", "float"), ("official", "float"),
                             ("fork", "official")):
                    add(f"{a} / {b}", len(sets[a] & sets[b]) / k)
            p0 += nt

        print(f"layer {layer:2d} (ratio {ratio}, keys of {src}): q identical to float {np.mean(st['q_same_float']):.5f}, "
              f"to int8 {np.mean(st['q_same_int8']):.5f}; w max rel err {max(st['w']):.1e}, "
              f"score max rel err {max(st['score']):.1e}")
        for key in ("fork / fork scores", "fork / int8", "fork / float", "official / float", "fork / official"):
            o = np.array(st[key])
            print(f"         top-k overlap {key:<20s} mean {o.mean():.4f}  p1 {np.percentile(o, 1):.4f}  "
                  f"min {o.min():.4f}  identical sets {(o == 1).mean():.3f}")

if __name__ == "__main__":
    main()
