#!/usr/bin/env python3
"""The tokens a server samples for a fixed set of requests with fixed seeds, for comparing two configurations whose
sampling must pick the same tokens (sample-fast.sh).

    seeded_tokens.py run <corpus> OUT [--prompts 4] [--tokens 256] [--chars 6000]
        PROMPTS slices of the corpus (CHARS characters each, from spread-out offsets) with a question, each a chat
        whose tokens /apply-template gives, answered for up to TOKENS tokens through /completion with the server's
        sampling (temperature 1.0 in production), thinking, and a seed of its own, then PROMPTS/2 of them answered
        greedily without thinking, all with the token ids returned and no probabilities (with probabilities, or with
        the end of text ignored, which biases the logits, the server samples on its original path); then two chat
        requests with a tool that they ask to be called, whose lazy grammar checks the sampled tokens from the call on
        (and resamples a token it rejects), the first thinking, the second not. Saves the outputs to OUT
    seeded_tokens.py compare A B
        per request whether the two outputs are the same, or where they part

The server is at SEEDED_URL (default http://127.0.0.1:8080).
"""

import argparse
import json
import os
import sys
import urllib.request

URL = os.environ.get("SEEDED_URL", "http://127.0.0.1:8080")
QUESTIONS = ["上面这段在讲什么？分点说明。", "上面这段里有哪些数字，它们说明了什么？", "根据上面这段，接下来应该做什么？",
             "把上面这段改写成给新同事的说明。"]
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "读取一个文件的内容",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
ASKS = ["读一下 /etc/hostname，告诉我这台机器叫什么。", "看看 ~/abyss-stats/restart.sh 里 server 用的是哪个模型文件。"]


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def run(a):
    corpus = open(a.corpus, encoding="utf-8").read()
    step = max(1, (len(corpus) - a.chars) // max(1, a.prompts - 1))
    texts = [corpus[i * step:i * step + a.chars] + "\n\n" + QUESTIONS[i % len(QUESTIONS)] for i in range(a.prompts)]
    reqs = [(f"sampled {i}", {"seed": 11 + i}, True) for i in range(a.prompts)]
    reqs += [(f"greedy {i}", {"temperature": 0}, False) for i in range(0, a.prompts, 2)]
    out = []
    for name, extra, think in reqs:
        chat = post("/apply-template", {"messages": [{"role": "user", "content": texts[int(name.split()[1])]}],
                                        "chat_template_kwargs": {"enable_thinking": think}, "with_tokens": True})
        body = {"prompt": chat["tokens"], "n_predict": a.tokens, "cache_prompt": False, "return_tokens": True, **extra}
        r = post("/completion", body)
        t = r["timings"]
        print(f"{name}: {t['prompt_n']} tok, gen {t['predicted_n']} tok at {t['predicted_per_second']:.1f} tok/s",
              flush=True)
        out.append({"name": name, "tokens": r["tokens"]})
    for i, ask in enumerate(ASKS):
        body = {"messages": [{"role": "user", "content": ask}], "tools": TOOLS, "max_tokens": 1024, "seed": 21 + i,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": i == 0}}
        r = post("/v1/chat/completions", body)
        m = r["choices"][0]["message"]
        calls = [c["function"] for c in m.get("tool_calls") or []]
        print(f"tools {i}: {r['usage']['prompt_tokens']} tok, gen {r['usage']['completion_tokens']} tok, {len(calls)} "
              f"calls {json.dumps(calls, ensure_ascii=False)[:120]}", flush=True)
        out.append({"name": f"tools {i}", "message": {k: m.get(k) for k in ("content", "reasoning_content")},
                    "calls": calls})
    json.dump(out, open(a.out, "w"), ensure_ascii=False)


def compare(a):
    xs, ys = (json.load(open(p)) for p in (a.a, a.b))
    bad = 0
    for x, y in zip(xs, ys):
        if "tokens" in x:
            n = min(len(x["tokens"]), len(y["tokens"]))
            same = next((j for j in range(n) if x["tokens"][j] != y["tokens"][j]), n)
            ok = same == n and len(x["tokens"]) == len(y["tokens"])
            print(f"  {x['name']:10s} {'same' if ok else 'differ'}: {same} of {len(x['tokens'])} / {len(y['tokens'])}"
                  " tokens identical")
        else:
            ok = x == y
            print(f"  {x['name']:10s} {'same' if ok else 'differ'}: {len(x['calls'])} / {len(y['calls'])} calls")
        bad += not ok
    if len(xs) != len(ys):
        print(f"  {len(xs)} outputs against {len(ys)}")
        bad += 1
    sys.exit(1 if bad else 0)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("corpus")
    r.add_argument("out")
    r.add_argument("--prompts", type=int, default=4)
    r.add_argument("--tokens", type=int, default=256)
    r.add_argument("--chars", type=int, default=6000)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
