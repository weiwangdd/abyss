#!/usr/bin/env python3
"""Tokens of slot saves (roundtrip-capture.sh), for finding where a re-rendered answer leaves the generated one.

    slot_tokens.py extract <slot save> <out.tok>    the token ids of a llama_state_seq save (u32 magic, version, count)
    slot_tokens.py compare <dir> [url]              each capture against the next: where their common prefix ends, and
                                                    when that is inside the earlier capture's tail, both sides around it
                                                    (the pieces from the server's /detokenize, default the production
                                                    server at http://127.0.0.1:8080)
"""
import array
import glob
import json
import os
import struct
import sys
import urllib.request


def extract(src, dst):
    with open(src, "rb") as f:
        magic, version, n = struct.unpack("<III", f.read(12))
        a = array.array("i")
        a.frombytes(f.read(4 * n))
    a.tofile(open(dst, "wb"))
    print(f"{dst}: {n} tokens (magic {magic:#x}, version {version})")


def load(path):
    a = array.array("i")
    a.frombytes(open(path, "rb").read())
    return a


def piece(url, t, cache={}):
    if t not in cache:
        req = urllib.request.Request(url + "/detokenize", json.dumps({"tokens": [t]}).encode(),
                                     {"Content-Type": "application/json"})
        cache[t] = json.load(urllib.request.urlopen(req, timeout=10))["content"]
    return cache[t]


def text(url, toks):
    req = urllib.request.Request(url + "/detokenize", json.dumps({"tokens": list(toks)}).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))["content"]


def compare(d, url):
    caps = sorted(glob.glob(os.path.join(d, "*.tok")), key=lambda p: int(os.path.basename(p).split(".")[0]))
    for a_path, b_path in zip(caps, caps[1:]):
        a, b = load(a_path), load(b_path)
        n = min(len(a), len(b))
        lcp = next((i for i in range(n) if a[i] != b[i]), n)
        before_end = len(a) - lcp
        print(f"{os.path.basename(a_path)} ({len(a)}) -> {os.path.basename(b_path)} ({len(b)}): common prefix {lcp}, "
              f"{before_end} before the end of the first")
        if 0 < before_end <= 2000:
            for name, s in (("generated (first)", a), ("re-rendered (second)", b)):
                print(f"  {name}:")
                print("    " + " ".join(f"{t}:{piece(url, t)!r}" for t in s[max(0, lcp - 12):lcp + 16]))
            print(f"  generated tail:   {text(url, a[max(0, lcp - 60):len(a)])!r}")
            print(f"  re-rendered there: {text(url, b[max(0, lcp - 60):min(len(b), lcp + before_end + 40)])!r}")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080")
