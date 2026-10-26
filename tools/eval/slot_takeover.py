#!/usr/bin/env python3
"""Steps of slot-takeover.sh against a running server: a conversation that a request sharing only its first part
takes over, as an agent's review of its own turn does, then the conversation's next turn.

    slot_takeover.py <corpus> grow N       the conversation, one request per round up to round N (one checkpoint each)
    slot_takeover.py <corpus> review K N   rounds 1..K of the conversation and a review question (it diverges after
                                           round K of N), print what the prompt cache reused
    slot_takeover.py <corpus> next N OUT   all N rounds and one more question, 32 greedy tokens with top-20 logprobs
                                           saved to OUT, print what the prompt cache reused
    slot_takeover.py <corpus> compare A B
"""

import json
import math
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080"
ROUNDS = 36  # the corpus is cut into this many user messages of about 1,700 tokens


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def rounds(corpus, n):
    text = corpus * 4
    step = len(text) // ROUNDS
    messages = [{"role": "system", "content": "你是一个代码助手。"}]
    for i in range(n):
        if i:
            messages.append({"role": "assistant", "content": f"收到第 {i} 段，已经读完。"})
        messages.append({"role": "user", "content": f"第 {i + 1} 段材料：\n\n" + text[i * step:(i + 1) * step]})
    return messages


def ask(messages, **extra):
    body = {"messages": messages, "max_tokens": 4, "temperature": 0, "cache_prompt": True}
    body.update(extra)
    r = post("/v1/chat/completions", body)
    return r, r["timings"]


def main():
    corpus = open(sys.argv[1], encoding="utf-8").read()
    cmd = sys.argv[2]
    if cmd == "grow":
        n = int(sys.argv[3])
        t0, total = time.time(), 0
        for i in range(1, n + 1):
            _, t = ask(rounds(corpus, i))
            total += t["prompt_n"]
        print(f"grow: {n} requests, {t['cache_n'] + t['prompt_n']} tokens at the end, processed {total} tok, "
              f"wall {time.time() - t0:.1f} s")
    elif cmd == "review":
        k, n = int(sys.argv[3]), int(sys.argv[4])
        messages = rounds(corpus, k)
        messages.append({"role": "assistant", "content": f"收到第 {k} 段，已经读完。"})
        messages.append({"role": "user", "content": "回顾一下到目前为止的对话：有没有值得记下来的做法？"})
        t0 = time.time()
        _, t = ask(messages)
        print(f"review after round {k} of {n}: reused {t.get('cache_n', 0)}, processed {t['prompt_n']} tok in "
              f"{t['prompt_ms'] / 1000:.1f} s, wall {time.time() - t0:.1f} s")
    elif cmd == "next":
        n = int(sys.argv[3])
        messages = rounds(corpus, n)
        messages.append({"role": "assistant", "content": f"收到第 {n} 段，已经读完。"})
        messages.append({"role": "user", "content": "第 3 段讲了什么？用三句话说。"})
        t0 = time.time()
        r, t = ask(messages, max_tokens=32, logprobs=True, top_logprobs=20)
        print(f"next: reused {t.get('cache_n', 0)}, processed {t['prompt_n']} tok in {t['prompt_ms'] / 1000:.1f} s, "
              f"wall {time.time() - t0:.1f} s")
        json.dump(r["choices"][0]["logprobs"]["content"], open(sys.argv[4], "w"))
    elif cmd == "compare":
        a, b = (json.load(open(p)) for p in sys.argv[3:5])
        n = min(len(a), len(b))
        same = next((i for i in range(n) if a[i]["token"] != b[i]["token"]), n)
        kl = []
        for i in range(same):
            pa = {x["token"]: x["logprob"] for x in a[i]["top_logprobs"]}
            pb = {x["token"]: x["logprob"] for x in b[i]["top_logprobs"]}
            floor = min(pb.values())
            kl.append(sum(math.exp(lp) * (lp - pb.get(tok, floor)) for tok, lp in pa.items()))
        print(f"identical tokens: first {same} of {n}" +
              (f", top-20 KL mean {sum(kl) / same:.2e} max {max(kl):.2e}" if same else ""))


if __name__ == "__main__":
    main()
