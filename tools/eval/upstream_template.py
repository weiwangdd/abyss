#!/usr/bin/env python3
"""The prompt a chat request becomes on a server, to check that upstream llama.cpp and the fork see the same tokens.

    upstream_template.py dump OUT      the rendered prompt (/apply-template) of a few fixed requests and its tokens,
                                       saved to OUT: the chat path's own tokens where the server returns them
                                       ("with_tokens", mx-llama.cpp/0018), else /tokenize of the prompt as the chat
                                       path tokenizes it (with special tokens)
    upstream_template.py compare A B   where the prompts and the tokens of two dumps differ
"""

import json
import os
import sys
import urllib.request

URL = os.environ.get("TMPL_URL", "http://127.0.0.1:8080")
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
REQUESTS = {
    "plain": {"messages": [{"role": "user", "content": "下面这份材料讲的是什么软件？用一句话回答。"}],
              "chat_template_kwargs": {"enable_thinking": False}},
    "agent": {"messages": [
        {"role": "system", "content": "你是一个帮助维护代码仓库的助手。"},
        {"role": "user", "content": "看一下 README 里写了什么。"},
        {"role": "assistant", "content": "", "reasoning_content": "先读 README。", "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "read_file", "arguments": "{\"path\": \"README.md\"}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "# Abyss\n\n家用硬件上的分层推理系统。"},
        {"role": "assistant", "content": "README 说这是一个家用硬件上的分层推理系统。", "reasoning_content": "读完了。"},
        {"role": "user", "content": "再列一下 tools 目录。"}],
        "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": True}},
}


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def dump(out):
    res = {}
    for name, body in REQUESTS.items():
        r = post("/apply-template", dict(body, with_tokens=True))
        tokens = r.get("tokens")
        if tokens is None:
            tokens = post("/tokenize", {"content": r["prompt"], "add_special": True, "parse_special": True})["tokens"]
        res[name] = {"prompt": r["prompt"], "tokens": tokens}
        print(f"{name}: {len(r['prompt'])} chars, {len(tokens)} tokens")
    json.dump(res, open(out, "w"), ensure_ascii=False)


def compare(a, b):
    a, b = json.load(open(a)), json.load(open(b))
    for name in a:
        pa, pb = a[name]["prompt"], b[name]["prompt"]
        ta, tb = a[name]["tokens"], b[name]["tokens"]
        if pa == pb and ta == tb:
            print(f"{name}: same prompt, same {len(ta)} tokens")
            continue
        if pa != pb:
            i = next((k for k in range(min(len(pa), len(pb))) if pa[k] != pb[k]), min(len(pa), len(pb)))
            print(f"{name}: prompts differ at char {i} of {len(pa)} / {len(pb)}: {pa[max(i - 40, 0):i + 40]!r} vs "
                  f"{pb[max(i - 40, 0):i + 40]!r}")
        if ta != tb:
            i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
            print(f"{name}: tokens differ at {i} of {len(ta)} / {len(tb)}: {ta[i:i + 8]} vs {tb[i:i + 8]}")


if __name__ == "__main__":
    {"dump": lambda: dump(sys.argv[2]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
