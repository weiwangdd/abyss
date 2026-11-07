#!/usr/bin/env python3
"""Lay out the production GGUF the way upstream llama.cpp reads DeepSeek-V4.1 (src/models/deepseek41.cpp, e95426f16):
the same weights, renamed and regrouped, so that upstream and the fork can be compared on identical bits.

- indexer_compressor_kv / indexer_compressor_norm (attn.indexer.wk / k_norm) -> indexer.attn_k / indexer.k_norm
- engram_wkv -> engram_kv
- engram_q * engram_k (BF16, so the product is exact in F32) -> engram_gate (F32), as upstream's converter folds them
- the two engram_embed tables (MXFP4) -> one per_layer_token_embd, their rows back to back in layer order
- the V4.1 keys under upstream's names (attention.kv_source_layers, attention.indexer.*, ple.*), the Engram head
  offsets made absolute into the joined table; engram.pad_token_id and engram.token_map already have upstream's names

Upstream's own converter writes the dense FP8 weights and the Engram tables as BF16 (the tables alone ~393 GB); this
keeps the production types (Q8_0, MXFP4), so a comparison is between the runtimes. Shards 2-7 hold experts only and
are symlinked; shard 1 (the key/values), shard 8 (the renamed indexer tensors) and the Engram files are written, about
172 GB. Both sides of the copy use direct I/O, so a running server keeps its hot pages. With --config (the checkpoint's
config.json) the Engram hash constants are also recomputed the way upstream's converter does.

    to_upstream.py SRC_DIR DST_DIR [--config CONFIG_JSON] [--check]
"""

import argparse
import glob
import json
import math
import mmap
import os
import random
import re
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v41io  # noqa: E402

ARCH = "deepseek41"
P = ARCH + "."
F32, BF16, MXFP4 = 0, 30, 39
ALIGN = 4096  # for O_DIRECT
PIECE = 64 << 20

# GGUF value types
T_UINT16, T_UINT32, T_INT32, T_ARRAY, T_UINT64 = 2, 4, 5, 9, 10
FMT = {T_UINT16: "H", T_UINT32: "I", T_INT32: "i", T_UINT64: "Q"}

RENAME = {
    "indexer_compressor_kv.weight": "indexer.attn_k.weight",
    "indexer_compressor_norm.weight": "indexer.k_norm.weight",
    "engram_wkv.weight": "engram_kv.weight",
}
# the fork's V4.1 keys that get upstream's names; split.* is rewritten for the new tensor count
DROP = {P + k for k in ("attention.kv_source_layer_ids", "attention.index_source_layer_ids",
                        "attention.candidate_source_layer_id", "attention.candidate_block_size",
                        "attention.candidate_topk_blocks", "engram.layer_ids", "engram.head_dim", "engram.n_heads",
                        "engram.max_ngram_size", "engram.compressed_vocab_size", "engram.num_embeddings",
                        "engram.primes", "engram.offsets", "engram.multipliers")} | {
    "split.no", "split.count", "split.tensors.count"}


def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_scalar(t, value):
    return struct.pack("<I" + FMT[t], t, value)


def kv_array(t, values):
    return struct.pack("<IIQ", T_ARRAY, t, len(values)) + struct.pack(f"<{len(values)}{FMT[t]}", *values)


def raw_kvs(path):
    """Each key/value of a GGUF in file order, as (key, its serialized type and value)."""
    with open(path, "rb") as f:
        r = v41io._Reader(f)
        assert f.read(4) == b"GGUF", path
        r.get("I")
        r.get("Q")
        out = []
        for _ in range(r.get("Q")):
            key = r.string()
            start = f.tell()
            r.value(r.get("I"))
            end = f.tell()
            f.seek(start)
            out.append((key, f.read(end - start)))
            f.seek(end)
        return out


def pad(n, a=32):
    return (n + a - 1) // a * a


class DirectReader:
    """Byte ranges of a file read with O_DIRECT through an aligned buffer, handed out as views of that buffer."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.buf = mmap.mmap(-1, PIECE + ALIGN)
        self.mv = memoryview(self.buf)

    def chunks(self, offset, nbytes):
        pos, end = offset, offset + nbytes
        while pos < end:
            a = pos // ALIGN * ALIGN
            n = pad(min(end, a + PIECE) - a, ALIGN)
            got = os.preadv(self.fd, [self.mv[:n]], a)
            hi = min(end - a, got)
            assert hi > pos - a, f"short read at {a}"
            yield self.mv[pos - a:hi]  # valid until the next read
            pos = a + hi

    def read(self, offset, nbytes):
        return b"".join(bytes(c) for c in self.chunks(offset, nbytes))


class DirectWriter:
    """A file written front to back with O_DIRECT through an aligned buffer, cut to its true size at the end."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        self.buf = mmap.mmap(-1, PIECE)
        self.mv = memoryview(self.buf)
        self.fill = 0
        self.done = 0

    def write(self, data):
        data = memoryview(data).cast("B")
        while len(data):
            n = min(len(data), PIECE - self.fill)
            self.mv[self.fill:self.fill + n] = data[:n]
            self.fill += n
            data = data[n:]
            if self.fill == PIECE:
                self._flush(PIECE)

    def _flush(self, n):
        w = 0
        while w < n:
            w += os.write(self.fd, self.mv[w:n])
        self.done += n
        self.fill = 0

    def close(self):
        size = self.done + self.fill
        if self.fill:
            n = pad(self.fill, ALIGN)
            self.mv[self.fill:n] = bytes(n - self.fill)
            self._flush(n)
        os.ftruncate(self.fd, size)
        os.fsync(self.fd)
        os.close(self.fd)
        return size


def write_gguf(path, kvs, tensors):
    """kvs: (key, serialized type and value); tensors: dicts with name, dims, type, nbytes and chunks (an iterator)."""
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for key, raw in kvs:
        head += gstr(key) + raw
    off = 0
    for t in tensors:
        head += gstr(t["name"]) + struct.pack("<I", len(t["dims"])) + struct.pack(f"<{len(t['dims'])}Q", *t["dims"])
        head += struct.pack("<IQ", t["type"], off)
        off += pad(t["nbytes"])
    head += bytes(pad(len(head)) - len(head))
    tmp = path + ".tmp"
    w = DirectWriter(tmp)
    w.write(head)
    for t in tensors:
        n = 0
        for c in t["chunks"]:
            w.write(c)
            n += len(c)
        assert n == t["nbytes"], (t["name"], n, t["nbytes"])
        w.write(bytes(pad(n) - n))
    size = w.close()
    assert size == len(head) + off, (path, size, len(head) + off)
    os.rename(tmp, path)
    return size


def bf16_to_f32(b):
    return (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def expected_names(kv):
    """The tensors upstream's load_arch_tensors creates for this model, by its rules for each layer."""
    n_layer = kv[P + "block_count"]
    ratios = kv[P + "attention.compress_ratios"]
    kv_src, idx_src = set(kv[P + "attention.kv_source_layer_ids"]), set(kv[P + "attention.index_source_layer_ids"])
    engram = set(kv[P + "engram.layer_ids"])
    names = {"token_embd.weight", "output_norm.weight", "output.weight", "per_layer_token_embd.weight"}
    for i in range(n_layer):
        b = f"blk.{i}."
        names |= {b + n + ".weight" for n in ("attn_norm", "attn_sinks", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv",
                                              "attn_kv_a_norm", "attn_output_a", "attn_output_b", "hc_attn_fn",
                                              "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
                                              "ffn_gate_inp", "ffn_norm", "ffn_gate_exps", "ffn_down_exps", "ffn_up_exps",
                                              "ffn_gate_shexp", "ffn_down_shexp", "ffn_up_shexp")}
        names.add(b + "exp_probs_b.bias")
        if i in kv_src:
            names |= {b + n + ".weight" for n in ("attn_compressor_kv", "attn_compressor_norm", "indexer.attn_k",
                                                  "indexer.k_norm")}
            if ratios[i] > 1:
                names.add(b + "attn_compressor_gate.weight")
        if i in idx_src:
            names |= {b + "indexer.attn_q_b.weight", b + "indexer.proj.weight"}
        if i in engram:
            names |= {b + "engram_kv.weight", b + "engram_gate.weight"}
    return names


def check_constants(kv, config_path):
    """The Engram primes, multipliers and token map, recomputed as upstream's conversion/deepseek41.py does."""
    hp = json.load(open(config_path))
    hp = hp.get("text_config", hp)
    layer_ids, n_gram, n_heads = kv[P + "engram.layer_ids"], kv[P + "engram.max_ngram_size"], kv[P + "engram.n_heads"]
    assert (list(layer_ids), n_gram, n_heads) == (hp["engram_layer_ids"], hp["engram_max_ngram_size"], hp["engram_n_heads"])

    def is_prime(n):
        if n < 2 or n % 2 == 0:
            return n == 2
        return all(n % d for d in range(3, math.isqrt(n) + 1, 2))

    seen, primes = set(), []
    for _ in layer_ids:
        for _ in range(n_gram - 1):
            cur = hp["engram_vocab_size"] - 1
            for _ in range(n_heads):
                cand = cur + 1
                while not is_prime(cand) or cand in seen:
                    cand += 1
                cur = cand
                seen.add(cur)
                primes.append(cur)
    assert primes == list(kv[P + "engram.primes"]), "primes differ from upstream's"
    cvocab = kv[P + "engram.compressed_vocab_size"]
    bound = max(1, (np.iinfo(np.int64).max // cvocab) // 2)
    mult = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        mult += [int(v) * 2 + 1 for v in rng.integers(low=0, high=bound, size=(n_gram,), dtype=np.int64)]
    assert mult == list(kv[P + "engram.multipliers"]), "multipliers differ from upstream's"
    print("engram primes and multipliers: as upstream computes them")
    try:
        from tokenizers import Regex, normalizers
        from transformers import AutoTokenizer
    except ImportError:
        print("engram token map: not checked (no tokenizers/transformers here)")
        return
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(config_path))
    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    backend = tokenizer.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    assert len(key_to_new) == cvocab and lookup == list(kv[P + "engram.token_map"]), "token map differs from upstream's"
    print("engram token map: as upstream computes it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--config")
    ap.add_argument("--check", action="store_true", help="only check an existing DST against SRC")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*-of-*.gguf")))
    g = [v41io.read_gguf(p) for p in files]
    n_split = len(files)
    kv = g[0]["kv"]
    assert kv["general.architecture"] == ARCH and kv["split.count"] == n_split
    if args.config:
        check_constants(kv, args.config)

    where = {}  # tensor name -> (file index, tensor info)
    for i, h in enumerate(g):
        for t in h["tensors"]:
            where[t["name"]] = (i, t)
    engram_files = [i for i, h in enumerate(g) if all("engram" in t["name"] for t in h["tensors"])]
    layer_ids = list(kv[P + "engram.layer_ids"])
    # this model: shard 1 and 2-7 experts, 8 the rest, then per Engram layer its table and its small tensors
    assert len(engram_files) == 2 * len(layer_ids) == 4 and engram_files == list(range(n_split - 4, n_split))
    renamed = {i for i, h in enumerate(g) for t in h["tensors"] if "engram" in t["name"] or "indexer_compressor" in t["name"]}
    assert 0 not in renamed

    def new_name(name):
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        return f"blk.{m.group(1)}.{RENAME[m.group(2)]}" if m and m.group(2) in RENAME else name

    readers = {}

    def chunks(name):
        i, t = where[name]
        if files[i] not in readers:
            readers[files[i]] = DirectReader(files[i])
        return readers[files[i]].chunks(g[i]["data_start"] + t["offset"], t["nbytes"])

    def copy(name):
        i, t = where[name]
        return {"name": new_name(name), "dims": t["dims"], "type": t["type"], "nbytes": t["nbytes"], "chunks": chunks(name)}

    def gate(il):
        q, k = (readers_small(f"blk.{il}.engram_{w}.weight") for w in ("q", "k"))
        data = (bf16_to_f32(q) * bf16_to_f32(k)).astype(np.float32).tobytes()
        dims = where[f"blk.{il}.engram_q.weight"][1]["dims"]
        return {"name": f"blk.{il}.engram_gate.weight", "dims": dims, "type": F32, "nbytes": len(data), "chunks": iter([data])}

    def readers_small(name):
        i, t = where[name]
        assert t["type"] == BF16
        fd = os.open(files[i], os.O_RDONLY)
        b = os.pread(fd, t["nbytes"], g[i]["data_start"] + t["offset"])
        os.close(fd)
        return b

    tables = [where[f"blk.{il}.engram_embed.weight"][1] for il in layer_ids]
    assert all(t["type"] == MXFP4 and t["dims"][0] == tables[0]["dims"][0] for t in tables)
    assert [t["dims"][1] for t in tables] == list(kv[P + "engram.num_embeddings"])

    def table():
        for il in layer_ids:
            yield from chunks(f"blk.{il}.engram_embed.weight")

    joined = {"name": "per_layer_token_embd.weight", "dims": [tables[0]["dims"][0], sum(t["dims"][1] for t in tables)],
              "type": MXFP4, "nbytes": sum(t["nbytes"] for t in tables), "chunks": table()}

    # each output shard by position: None is a symlink to the source's
    plan = [None] * n_split
    plan[0] = [copy(t["name"]) for t in g[0]["tensors"]]
    for i in sorted(renamed - set(engram_files)):
        plan[i] = [copy(t["name"]) for t in g[i]["tensors"]]
    e0 = engram_files[0]
    plan[e0] = [joined]
    plan[e0 + 1] = [copy(f"blk.{layer_ids[0]}.engram_wkv.weight"), gate(layer_ids[0])]
    plan[e0 + 2] = [copy(f"blk.{layer_ids[1]}.engram_wkv.weight")]
    plan[e0 + 3] = [gate(layer_ids[1])]
    n_tensors = sum(len(p) if p is not None else len(g[i]["tensors"]) for i, p in enumerate(plan))

    # the key/values: the source's, less the renamed ones, plus upstream's
    n_gram, n_heads = kv[P + "engram.max_ngram_size"], kv[P + "engram.n_heads"]
    primes, offsets, num = list(kv[P + "engram.primes"]), list(kv[P + "engram.offsets"]), list(kv[P + "engram.num_embeddings"])
    n_cols = (n_gram - 1) * n_heads
    assert len(primes) == len(offsets) == len(layer_ids) * n_cols
    for e in range(len(layer_ids)):
        per = primes[e * n_cols:(e + 1) * n_cols]
        # the fork's offsets start at 0 again in each layer's own table
        assert offsets[e * n_cols:(e + 1) * n_cols] == [sum(per[:c]) for c in range(n_cols)] and sum(per) == num[e]
    abs_offsets = [sum(primes[:c]) for c in range(len(primes))]
    new_kv = [(k, raw) for k, raw in raw_kvs(files[0]) if k not in DROP] + [
        (P + "attention.kv_source_layers", kv_array(T_UINT32, kv[P + "attention.kv_source_layer_ids"])),
        (P + "attention.indexer.source_layers", kv_array(T_UINT32, kv[P + "attention.index_source_layer_ids"])),
        (P + "attention.indexer.candidate_source_layer", kv_scalar(T_INT32, kv[P + "attention.candidate_source_layer_id"])),
        (P + "attention.indexer.candidate_top_k_blocks", kv_scalar(T_UINT32, kv[P + "attention.candidate_topk_blocks"])),
        (P + "attention.indexer.candidate_block_size", kv_scalar(T_UINT32, kv[P + "attention.candidate_block_size"])),
        (P + "ple.layers", kv_array(T_UINT32, layer_ids)),
        (P + "ple.ngram_size", kv_scalar(T_UINT32, n_gram)),
        (P + "ple.heads_per_ngram", kv_scalar(T_UINT32, n_heads)),
        (P + "embedding_length_per_layer_input", kv_scalar(T_UINT32, kv[P + "engram.head_dim"])),
        (P + "ple.layer_multipliers", kv_array(T_UINT64, kv[P + "engram.multipliers"])),
        (P + "ple.head_offsets", kv_array(T_UINT64, abs_offsets)),
        (P + "ple.head_vocab_sizes", kv_array(T_UINT64, primes)),
    ]

    def split_kv(i):
        return [("split.no", kv_scalar(T_UINT16, i)), ("split.count", kv_scalar(T_UINT16, n_split)),
                ("split.tensors.count", kv_scalar(T_INT32, n_tensors))]

    os.makedirs(args.dst, exist_ok=True)
    if not args.check:
        need = sum(t["nbytes"] for p in plan if p for t in p)
        free = os.statvfs(args.dst).f_bavail * os.statvfs(args.dst).f_frsize
        assert free > need + (100 << 30), f"{free / 2**30:.0f} GiB free, {need / 2**30:.0f} GiB needed"
        for i, p in enumerate(plan):
            out = os.path.join(args.dst, os.path.basename(files[i]))
            if p is None:
                if not os.path.lexists(out):
                    os.symlink(os.path.realpath(files[i]), out)
                continue
            t0 = time.time()
            size = write_gguf(out, (new_kv if i == 0 else []) + split_kv(i), p)
            print(f"{out}: {len(p)} tensors, {size / 2**30:.2f} GiB in {time.time() - t0:.0f} s", flush=True)

    # check: the headers as upstream's loader reads them, the names upstream creates, and sampled bytes of each tensor
    out_files = [os.path.join(args.dst, os.path.basename(f)) for f in files]
    h = [v41io.read_gguf(p) for p in out_files]
    assert all(x["kv"]["split.no"] == i and x["kv"]["split.count"] == n_split for i, x in enumerate(h))
    got = {t["name"]: (i, t) for i, x in enumerate(h) for t in x["tensors"]}
    assert len(got) == sum(len(x["tensors"]) for x in h) == h[0]["kv"]["split.tensors.count"] == n_tensors
    want = expected_names(kv)
    assert set(got) == want, (sorted(set(got) - want), sorted(want - set(got)))
    for x in h:
        off = 0
        for t in x["tensors"]:
            assert t["offset"] == off, t["name"]
            off += pad(t["nbytes"])
        assert x["data_start"] + off - pad(x["tensors"][-1]["nbytes"]) + x["tensors"][-1]["nbytes"] <= x["size"]
    for k in [k for k, _ in new_kv if k.startswith(P + "ple.") or k.startswith(P + "attention.indexer.c")
              or k.startswith(P + "attention.kv_s") or k.startswith(P + "attention.indexer.s")]:
        v = h[0]["kv"][k]
        print(f"  {k} = {v if not isinstance(v, list) or len(v) <= 8 else str(v[:8])[:-1] + f', ... ({len(v)})]'}")

    rng = random.Random(0)
    src_reader = {}

    def src_bytes(i, offset, n):
        if files[i] not in src_reader:
            src_reader[files[i]] = DirectReader(files[i])
        return src_reader[files[i]].read(offset, n)

    def dst_bytes(name, start, n):
        i, t = got[name]
        fd = os.open(out_files[i], os.O_RDONLY)
        b = os.pread(fd, n, h[i]["data_start"] + t["offset"] + start)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        return b

    n_checked = 0
    for name, (i, t) in sorted(got.items()):
        if name == "per_layer_token_embd.weight":
            row = t["nbytes"] // t["dims"][1]
            base = 0
            for il, tab in zip(layer_ids, tables):
                si, st = where[f"blk.{il}.engram_embed.weight"]
                for r in [0, 1, tab["dims"][1] - 1] + [rng.randrange(tab["dims"][1]) for _ in range(64)]:
                    assert src_bytes(si, g[si]["data_start"] + st["offset"] + r * row, row) == \
                        dst_bytes(name, (base + r) * row, row), (il, r)
                base += tab["dims"][1]
            n_checked += 1
            continue
        if name.endswith("engram_gate.weight"):
            il = int(name.split(".")[1])
            q, k = (readers_small(f"blk.{il}.engram_{w}.weight") for w in ("q", "k"))
            assert dst_bytes(name, 0, t["nbytes"]) == (bf16_to_f32(q) * bf16_to_f32(k)).astype(np.float32).tobytes()
            assert t["type"] == F32 and t["dims"] == where[f"blk.{il}.engram_q.weight"][1]["dims"]
            n_checked += 1
            continue
        old = next(o for o in where if new_name(o) == name)
        si, st = where[old]
        assert (t["dims"], t["type"]) == (st["dims"], st["type"]), name
        if plan[i] is None:
            continue  # a symlinked shard
        for start in {0, max(t["nbytes"] - (1 << 20), 0), rng.randrange(max(t["nbytes"] - (1 << 20), 0) + 1)}:
            n = min(1 << 20, t["nbytes"] - start)
            assert src_bytes(si, g[si]["data_start"] + st["offset"] + start, n) == dst_bytes(name, start, n), name
        n_checked += 1
    print(f"{args.dst}: {n_tensors} tensors in {n_split} files, all the names upstream creates, "
          f"{n_checked} written tensors sampled equal to the source")


if __name__ == "__main__":
    main()
