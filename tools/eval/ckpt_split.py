#!/usr/bin/env python3
"""Prompts split for checkpoints (ABYSS_CKPT_COUNT_START, mx-llama.cpp/0045) against a fresh server.

    ckpt_split.py <corpus> run OUT --log LOG [--agent-tokens N] [--many-repeat R]
                                               two long conversations with many messages, each from scratch, then with
                                               a user message at about 60% edited (the slot goes back to its checkpoint
                                               before the edit), then one more turn on the edited one; 32 greedy tokens
                                               each with top-20 logprobs. Prints each request's prefill with what the
                                               server log LOG says about it (where it continued from, the llama_decode()
                                               calls it took, the checkpoints they made) and saves it all to OUT. The
                                               second conversation has about N tokens (default 110000), the first
                                               R copies of the corpus (default 6, about 96k tokens); smaller ones
                                               with a smaller --checkpoint-min-step suit a server without the
                                               CED-aware prefill
    ckpt_split.py <corpus> compare A B         two runs request by request: prefill, calls, checkpoints, identical
                                               tokens and top-20 KL
"""

import argparse
import json
import math
import os
import random
import re
import time
import urllib.request

URL = os.environ.get("CKPT_URL", "http://127.0.0.1:8080")


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def many(corpus, repeat=6):
    # restart_prefill.py's conversation: about 96k tokens in 16 user messages of about 6k
    text = corpus * repeat
    n = 16
    step = len(text) // n
    messages = [{"role": "system", "content": "你是一个代码助手。"}]
    for i in range(n):
        messages.append({"role": "user", "content": f"第 {i + 1} 段材料：\n\n" + text[i * step:(i + 1) * step]})
        messages.append({"role": "assistant", "content": f"收到第 {i + 1} 段，已经读完。"})
    messages.append({"role": "user", "content": "以上材料分几部分？"})
    return messages


def agent(corpus, chars_per_token, n_tokens=110000):
    # an agent-like session: requests and tool output of 100-6,000 tokens (most of them short), answers of 50-500;
    # another system prompt than many(), so that it starts from scratch after it
    rng = random.Random(1)
    text = corpus * (int(n_tokens * chars_per_token * 1.2) // len(corpus) + 1)
    messages = [{"role": "system", "content": "你是一个在终端里工作的编程助手，读文件、跑命令，然后回答。"}]
    off = total = i = 0
    while total < n_tokens:
        r = rng.random()
        n_user = rng.randint(100, 600) if r < 0.6 else rng.randint(600, 2500) if r < 0.9 else rng.randint(2500, 6000)
        for role, n in (("user", n_user), ("assistant", rng.randint(50, 500))):
            m = int(n * chars_per_token)
            messages.append({"role": role, "content": f"[{role} {i + 1}]\n" + text[off:off + m]})
            off += m
            total += n
        i += 1
    messages.append({"role": "user", "content": "刚才最后读的那段讲的是什么？用一句话回答。"})
    return messages


def edited(messages, frac=0.6):
    # the user message at about frac of the conversation's text, with a line put in front of it
    sizes = [len(m["content"]) for m in messages]
    acc = 0
    for k, m in enumerate(messages):
        acc += sizes[k]
        if m["role"] == "user" and acc >= frac * sum(sizes):
            break
    out = [dict(m) for m in messages]
    out[k]["content"] = "（这一段后来改过。）\n" + out[k]["content"]
    return out


def next_turn(messages):
    return messages + [{"role": "assistant", "content": "好的，看完了。"},
                       {"role": "user", "content": "再用一句话说说第一部分讲了什么。"}]


def ask(name, messages, log):
    body = {"messages": messages, "max_tokens": 32, "temperature": 0, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True, "top_logprobs": 20}
    start = os.path.getsize(log)
    t0 = time.time()
    r = post("/v1/chat/completions", body)
    wall = time.time() - t0
    rec = {"name": name, "wall": wall, "timings": r["timings"], "tokens": r["choices"][0]["logprobs"]["content"],
           "calls": None, "ckpts": None}
    for _ in range(50):  # the timings, with the calls, can be printed after the answer is sent
        with open(log, encoding="utf-8", errors="replace") as f:
            f.seek(start)
            text = f.read()
        m = re.search(r"abyss: the prompt took (\d+) calls, ([\d +]+) tokens, checkpoints at((?: \d+)+| none)", text)
        if m:
            rec["calls"] = [int(x) for x in m.group(2).split(" + ")]
            rec["ckpts"] = [int(x) for x in m.group(3).split()] if m.group(3) != " none" else []
            break
        time.sleep(0.1)
    t = rec["timings"]
    calls = (f"{len(rec['calls'])} calls: {' + '.join(map(str, rec['calls']))}; checkpoints at "
             f"{' '.join(map(str, rec['ckpts'])) or 'none'}") if rec["calls"] is not None else "no calls line in the log"
    print(f"{name}: {t.get('cache_n', 0) + t['prompt_n']} tokens, from {t.get('cache_n', 0)}, {t['prompt_n']} in "
          f"{t['prompt_ms'] / 1000:.2f} s; {calls}", flush=True)
    return rec


def run(a):
    corpus = open(a.corpus, encoding="utf-8").read()
    chars_per_token = len(corpus) / len(post("/tokenize", {"content": corpus})["tokens"])
    recs = []
    for conv, messages in (("many", many(corpus, a.many_repeat)), ("agent", agent(corpus, chars_per_token, a.agent_tokens))):
        recs.append(ask(f"{conv} fresh", messages, a.log))
        messages = edited(messages)
        recs.append(ask(f"{conv} edit", messages, a.log))
        recs.append(ask(f"{conv} next", next_turn(messages), a.log))
    json.dump(recs, open(a.out, "w"), ensure_ascii=False)


def kl(a, b):
    # by token id, as different tokens can have the same text
    key = lambda t: t.get("id", t["token"])
    n = min(len(a), len(b))
    same = next((i for i in range(n) if key(a[i]) != key(b[i])), n)
    out = []
    for i in range(same):
        pa = {key(x): x["logprob"] for x in a[i]["top_logprobs"]}
        pb = {key(x): x["logprob"] for x in b[i]["top_logprobs"]}
        floor = min(pb.values())
        out.append(sum(math.exp(lp) * (lp - pb.get(tok, floor)) for tok, lp in pa.items()))
    return same, n, out


def compare(a):
    x = {r["name"]: r for r in json.load(open(a.a))}
    y = {r["name"]: r for r in json.load(open(a.b))}
    for name, p in x.items():
        q = y.get(name)
        if q is None:
            continue
        tp, tq = p["timings"], q["timings"]
        same, n, k = kl(p["tokens"], q["tokens"])
        calls = lambda r: len(r["calls"]) if r["calls"] is not None else "?"
        line = (f"{name}: prefill {tp['prompt_ms'] / 1000:.2f} -> {tq['prompt_ms'] / 1000:.2f} s, "
                f"calls {calls(p)} -> {calls(q)}, from {tp.get('cache_n', 0)} / {tq.get('cache_n', 0)}, "
                f"identical tokens {same} of {n}" + (f", top-20 KL mean {sum(k) / same:.2e} max {max(k):.2e}" if same else ""))
        print(line)
        if p["ckpts"] is not None and q["ckpts"] is not None and p["ckpts"] != q["ckpts"]:
            print(f"  checkpoints {' '.join(map(str, p['ckpts'] or []))}")
            print(f"           -> {' '.join(map(str, q['ckpts'] or []))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--log", required=True)
    r.add_argument("--agent-tokens", type=int, default=110000)
    r.add_argument("--many-repeat", type=int, default=6)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
