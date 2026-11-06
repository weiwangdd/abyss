#!/usr/bin/env python3
"""Decode speed of a long agent-like conversation under the ways a client can ask for it.

Turn 1 sends a system prompt, the agent tools and REPEAT copies of the corpus (about 16k tokens each), and generates
one token, which fills the prompt cache. Every later turn appends the previous answer and a new question, so the
prompt cache keeps the conversation and the expert cache sees new text each time, and generates MAX_TOKENS tokens in
one of the modes, round robin: greedy or the server's default sampling, streamed or not. The server log's eval time
of each request is the decode speed (decode_modes.sh reads it). The turns ignore the end of generation, so a sampled
answer that makes a tool call can leave the tool-call grammar nothing to accept (the server answers 500); such a turn
is left out of the conversation and the next one goes on. --seed S samples turn t with the seed S + t, so two servers
whose logits are the same bit for bit hold the same conversation (and their expert caches the same experts).
"""

import argparse
import os
import json
import subprocess
import time
import urllib.error
import urllib.request

URL = os.environ.get("DECODE_URL", "http://127.0.0.1:8080")
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "读取一个文件的内容",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "列出目录下的文件",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
]
MODES = {
    "greedy":         {"temperature": 0, "stream": False},
    "greedy-stream":  {"temperature": 0, "stream": True},
    "default":        {"stream": False},
    "default-stream": {"stream": True},
}
QUESTIONS = ["第 {} 节讲的是什么？详细说明它的设计。", "第 {} 节里的数字是怎么测出来的？", "第 {} 节有哪些风险？",
             "把第 {} 节改写成给新同事看的说明。"]


def request(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        if not body.get("stream"):
            msg = json.load(r)["choices"][0]["message"]
            return msg.get("content") or "", msg.get("reasoning_content") or ""
        content, reasoning = [], []
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
        return "".join(content), "".join(reasoning)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2, help="turns per mode")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--capture", help="nsys report of the last turn")
    ap.add_argument("--seed", type=int, help="sample turn t with seed SEED + t (default: the server picks one)")
    args = ap.parse_args()

    corpus = open(args.corpus, encoding="utf-8").read()
    messages = [
        {"role": "system", "content": "你是一个代码助手。需要看文件时调用 read_file 或 list_dir。"},
        {"role": "user", "content": "下面是项目的设计文档：\n\n" + corpus * args.repeat + "\n\n先说说它分几节。"},
    ]
    t0 = time.time()
    content, _ = request({"messages": messages, "tools": TOOLS, "max_tokens": 1, "temperature": 0,
                          "cache_prompt": True})
    print(f"turn 1: prefill {time.time() - t0:.1f} s", flush=True)
    messages.append({"role": "assistant", "content": content})

    modes = args.modes.split(",")
    n_turns = args.rounds * len(modes)
    for turn in range(n_turns):
        mode = modes[turn % len(modes)]
        capture = args.capture and turn == n_turns - 1
        messages.append({"role": "user", "content": QUESTIONS[turn % len(QUESTIONS)].format(turn + 1)})
        body = {"messages": messages, "tools": TOOLS, "max_tokens": args.max_tokens, "cache_prompt": True,
                "ignore_eos": True, **MODES[mode]}
        if args.seed is not None:
            body["seed"] = args.seed + turn
        if capture:
            subprocess.run(["nsys", "start", "--session=abyss", f"--output={args.capture}", "--force-overwrite=true"],
                           check=True)
        t0 = time.time()
        try:
            content, reasoning = request(body)
        except urllib.error.HTTPError as e:
            print(f"turn {turn + 2}: {mode:15s} failed, HTTP {e.code} after {time.time() - t0:.1f} s; left out", flush=True)
            if capture:
                subprocess.run(["nsys", "stop", "--session=abyss"], check=True)
            messages.pop()
            continue
        print(f"turn {turn + 2}: {mode:15s} {time.time() - t0:5.1f} s", flush=True)
        if capture:
            subprocess.run(["nsys", "stop", "--session=abyss"], check=True)
        answer = {"role": "assistant", "content": content}
        if reasoning:
            answer["reasoning_content"] = reasoning
        messages.append(answer)


if __name__ == "__main__":
    main()
