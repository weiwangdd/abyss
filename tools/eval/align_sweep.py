#!/usr/bin/env python3
"""Robustness sweep for ABYSS_CED_ALIGN: raw token prompts whose single prefill call (N - 4 tokens, the final 4 go
separately) has the given remainder mod 2048, each followed by a short generation."""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8080"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


corpus = open(sys.argv[1], encoding="utf-8").read()
toks = post("/tokenize", {"content": corpus})["tokens"]
toks = toks * (1 + 70000 // len(toks))

for k in (1, 2, 10):
    for r in (1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 129, 1000, 2047, 0):
        call = 2048 * k + r
        n = call + 4
        res = post("/completion", {"prompt": toks[:n], "n_predict": 8, "temperature": 0, "cache_prompt": False})
        t = res["timings"]
        print(f"call {call:6d} (mod 2048 = {r:4d}): prefill {t['prompt_ms'] / 1000:6.2f} s, gen {res['content']!r}",
              flush=True)
