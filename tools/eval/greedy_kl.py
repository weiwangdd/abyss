#!/usr/bin/env python3
"""Decode-path comparison of two server configurations on several prompts.

    greedy_kl.py run <corpus> OUT [--prompts 8] [--tokens 128] [--chars 6000]
        PROMPTS slices of the corpus (CHARS characters each, from spread-out offsets) with a question, each answered
        greedily for TOKENS tokens (thinking off, end of text ignored) with the top-20 logprobs of every token, saved
        to OUT (one list per prompt)
    greedy_kl.py compare A B
        per prompt the identical prefix of the two outputs, and over all identical tokens the top-20 KL(A || B): mean,
        median, 99th percentile and max (tokens B leaves out of its top 20 get B's smallest logprob)

The server is at GREEDY_URL (default http://127.0.0.1:8080). Every token after the prompt is a single-token decode, so
this measures the decode path, which kl.sh (batched evaluation) does not reach.
"""

import argparse
import json
import math
import os
import sys
import urllib.request

URL = os.environ.get("GREEDY_URL", "http://127.0.0.1:8080")
QUESTIONS = ["上面这段在讲什么？分点说明。", "上面这段里有哪些数字，它们说明了什么？", "根据上面这段，接下来应该做什么？",
             "把上面这段改写成给新同事的说明。"]


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def run(a):
    corpus = open(a.corpus, encoding="utf-8").read()
    step = max(1, (len(corpus) - a.chars) // max(1, a.prompts - 1))
    out = []
    for i in range(a.prompts):
        text = corpus[i * step:i * step + a.chars]
        body = {"messages": [{"role": "user", "content": text + "\n\n" + QUESTIONS[i % len(QUESTIONS)]}],
                "temperature": 0, "max_tokens": a.tokens, "ignore_eos": True, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True, "top_logprobs": 20}
        r = post("/v1/chat/completions", body)
        t = r["timings"]
        print(f"prompt {i}: {t['prompt_n']} tok, gen {t['predicted_n']} tok at {t['predicted_per_second']:.1f} tok/s",
              flush=True)
        out.append(r["choices"][0]["logprobs"]["content"])
    json.dump(out, open(a.out, "w"))


def compare(a):
    xs, ys = (json.load(open(p)) for p in (a.a, a.b))
    # by token id: different tokens can have the same text (ids 223 and 11827 are both " "), and keyed by text the
    # second of them took the place of the first
    key = lambda t: t.get("id", t["token"])
    kls, rows = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        n = min(len(x), len(y))
        same = next((j for j in range(n) if key(x[j]) != key(y[j])), n)
        for j in range(same):
            px = {key(t): t["logprob"] for t in x[j]["top_logprobs"]}
            py = {key(t): t["logprob"] for t in y[j]["top_logprobs"]}
            floor = min(py.values())
            kls.append(sum(math.exp(lp) * (lp - py.get(tok, floor)) for tok, lp in px.items()))
        rows.append(f"{same}/{n}")
    print(f"identical prefix per prompt: {' '.join(rows)}")
    if not kls:
        print("no identical tokens")
        return
    s = sorted(kls)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    print(f"top-20 KL over {len(s)} tokens: mean {sum(s) / len(s):.2e}, median {q(0.5):.2e}, p99 {q(0.99):.2e}, "
          f"max {s[-1]:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("corpus")
    r.add_argument("out")
    r.add_argument("--prompts", type=int, default=8)
    r.add_argument("--tokens", type=int, default=128)
    r.add_argument("--chars", type=int, default=6000)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
