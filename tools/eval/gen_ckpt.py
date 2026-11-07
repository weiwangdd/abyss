#!/usr/bin/env python3
"""Numerics of a conversation's next turn when the slot keeps the last answer (mx-llama.cpp/0041): per corpus slice,
turn 1 (the slice, the agent tools and a question, answered greedily, thinking on, up to TOKENS tokens), turn 2 (the
answer as the server returned it, then the tool results or a follow-up, answered greedily for 64 tokens) and turn 2b
(the same with the answer changed at 60% of its text, so the prompt leaves the slot inside the answer), each with the
top-20 logprobs of every token, saved to OUT as [turn 1, turn 2, turn 2b] per slice, flattened: compare with
greedy_kl.py compare A B. Turn 1 must come out the same in the two runs (same path); in turn 2 a slot that continues
from its end or from a checkpoint in the answer keeps the answer's memory from the decode instead of computing it
again, and in turn 2b the slot goes back to a checkpoint in the answer or before it.

    gen_ckpt.py <corpus> OUT [--prompts 4] [--tokens 768] [--chars 12000]

The server is at GEN_CKPT_URL (default http://127.0.0.1:8080).
"""

import argparse
import json
import os
import sys
import urllib.request

URL = os.environ.get("GEN_CKPT_URL", "http://127.0.0.1:8080")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "读取仓库里一个文件的内容",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}},
                       "required": ["path"]}}},
]
QUESTIONS = ["先说说上面这段在讲什么，再决定要不要读 README.md。", "上面这段里有哪些数字，它们说明了什么？",
             "根据上面这段，接下来应该做什么？需要的话读 docs/plan.md。", "把上面这段改写成给新同事的说明。"]


def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def turn(messages, max_tokens):
    r = post({"messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": max_tokens, "cache_prompt": True,
              "logprobs": True, "top_logprobs": 20})
    return r, r["choices"][0]["message"], r["choices"][0]["logprobs"]["content"]


def changed(msg):
    """The assistant message with a few characters put in at 60% of its longer text."""
    msg = dict(msg)
    key = "reasoning_content" if len(msg.get("reasoning_content") or "") > len(msg.get("content") or "") else "content"
    text = msg.get(key) or ""
    i = int(len(text) * 0.6)
    msg[key] = text[:i] + "（补一句）" + text[i:]
    return msg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=768)
    ap.add_argument("--chars", type=int, default=12000)
    a = ap.parse_args()
    corpus = open(a.corpus, encoding="utf-8").read()
    step = max(1, (len(corpus) - a.chars) // max(1, a.prompts - 1))
    out = []
    for i in range(a.prompts):
        text = corpus[i * step:i * step + a.chars]
        messages = [{"role": "system", "content": "你是一个代码助手。需要看文件时调用 read_file，一次只调一个工具。"},
                    {"role": "user", "content": text + "\n\n" + QUESTIONS[i % len(QUESTIONS)]}]
        r1, msg, lp1 = turn(messages, a.tokens)
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            assistant["reasoning_content"] = msg["reasoning_content"]
        follow = []
        if msg.get("tool_calls"):
            assistant["tool_calls"] = msg["tool_calls"]
            follow = [{"role": "tool", "tool_call_id": c.get("id", ""), "content": corpus[j * 3000:(j + 1) * 3000]}
                      for j, c in enumerate(msg["tool_calls"])]
        else:
            follow = [{"role": "user", "content": "好，再具体一点：第一步具体怎么做？"}]
        r2, _, lp2 = turn(messages + [assistant] + follow, 64)
        r2b, _, lp2b = turn(messages + [changed(assistant)] + follow, 64)
        t1, t2, t2b = r1["timings"], r2["timings"], r2b["timings"]
        print(f"slice {i}: turn 1 prompt {t1['prompt_n']} gen {t1['predicted_n']} "
              f"({'tool call' if msg.get('tool_calls') else 'answer'}); "
              f"turn 2 reused {t2.get('cache_n', 0)} computed {t2['prompt_n']} in {t2['prompt_ms'] / 1e3:.2f} s; "
              f"turn 2b reused {t2b.get('cache_n', 0)} computed {t2b['prompt_n']} in {t2b['prompt_ms'] / 1e3:.2f} s",
              flush=True)
        out += [lp1, lp2, lp2b]
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    sys.exit(main())
