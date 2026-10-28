#!/usr/bin/env python3
"""Drive a multi-turn agent session against the server and report, per turn, what the prompt cache reused and what
was processed again.

Turn 1 carries a long document (REPEAT copies of the corpus) and two tools. Each tool call gets a file-sized slice of
the corpus as its result; a reply without a tool call gets a follow-up user message. For each turn: the prompt length,
the tokens reused from the cache (cache_n), the tokens processed (prompt_n) and their time, and how many of the
processed tokens were already in the slot (the previous prompt and generation) - those are what a checkpoint rollback
or a re-rendered history costs.

    python3 agent_turns.py corpus.txt [--turns 6] [--repeat 3] [--max-tokens 512] [--capture REPORT]

With --capture the server runs in the nsys session "abyss" (see agent-turns.sh): the last turn's prefill is captured
into REPORT, with one generated token so the capture holds the prefill only.
"""

import argparse
import json
import os
import subprocess
import urllib.request

URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8080")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "读取仓库里一个文件的内容",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "列出仓库里一个目录下的文件",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "目录路径"}},
                       "required": ["path"]}}},
]


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--file-chars", type=int, default=4000, help="size of each tool result, about 1,500 tokens")
    ap.add_argument("--capture", help="nsys report of the last turn's prefill")
    args = ap.parse_args()

    corpus = open(args.corpus, encoding="utf-8").read()
    messages = [
        {"role": "system", "content": "你是一个代码助手。需要看文件时调用 read_file 或 list_dir，一次只调一个工具。"},
        {"role": "user", "content": "下面是项目的设计文档：\n\n" + corpus * args.repeat +
         "\n\n请先用 list_dir 看看 src 目录，再读取你认为最关键的文件，最后总结这个项目的结构。"},
    ]
    offset = 0
    prev_total = 0
    print(f"{'turn':>4} {'prompt':>7} {'reused':>7} {'processed':>9} {'of them old':>11} {'prefill s':>9} "
          f"{'gen':>5} {'gen tok/s':>9}  next")
    for turn in range(1, args.turns + 1):
        capture = args.capture and turn == args.turns
        if capture:
            subprocess.run(["nsys", "start", "--session=abyss", f"--output={args.capture}", "--force-overwrite=true"],
                           check=True)
        r = post("/v1/chat/completions", {"messages": messages, "tools": TOOLS,
                                          "max_tokens": 1 if capture else args.max_tokens,
                                          "temperature": 0, "cache_prompt": True})
        if capture:
            subprocess.run(["nsys", "stop", "--session=abyss"], check=True)
        t = r["timings"]
        msg = r["choices"][0]["message"]
        n_prompt = t.get("cache_n", 0) + t["prompt_n"]
        # tokens the slot held from the previous turn but processed again
        old = max(0, prev_total - t.get("cache_n", 0)) if turn > 1 else 0
        calls = msg.get("tool_calls") or []
        nxt = ", ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in calls) or "follow-up"
        print(f"{turn:>4} {n_prompt:>7} {t.get('cache_n', 0):>7} {t['prompt_n']:>9} {old:>11} "
              f"{t['prompt_ms'] / 1000:>9.1f} {t['predicted_n']:>5} {t['predicted_per_second']:>9.1f}  {nxt[:60]}")
        prev_total = n_prompt + t["predicted_n"]

        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("reasoning_content"):
            assistant["reasoning_content"] = msg["reasoning_content"]
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        if calls:
            for c in calls:
                chunk = corpus[offset:offset + args.file_chars]
                offset = (offset + args.file_chars) % max(1, len(corpus) - args.file_chars)
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": chunk})
        else:
            chunk = corpus[offset:offset + args.file_chars]
            offset = (offset + args.file_chars) % max(1, len(corpus) - args.file_chars)
            messages.append({"role": "user", "content": "再看看这段代码，说说它和上面的关系：\n\n" + chunk})


if __name__ == "__main__":
    main()
