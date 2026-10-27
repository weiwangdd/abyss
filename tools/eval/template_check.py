#!/usr/bin/env python3
"""Compare the DeepSeek-V4.1 chat template served by llama-server (/apply-template) with the official
encoding/encoding.py, on conversations that cover system and developer messages, tools, tool calls and results,
reasoning kept or dropped, and the reasoning effort.

    template_check.py <encoding_dsv41.py dir> [server url]

The official encoder has no "developer" role and takes tools on the first system message, so its input converts
developer to system and attaches the request's tools (and response_format) to the first system message, an empty one
if there is none, as the template does. Effort names outside the official low/high/max go in as the numbers the
template maps them to.
"""

import copy
import difflib
import importlib
import json
import sys
import urllib.request

enc = None  # the official encoder, see load()
URL = "http://127.0.0.1:8099"
EFFORT = {"minimal": 25, "medium": 62, "xhigh": 88}


def load(encoder_dir):
    """Import encoding_dsv41.py (the official encoding/encoding.py) from encoder_dir."""
    global enc
    sys.path.insert(0, encoder_dir)
    enc = importlib.import_module("encoding_dsv41")
    return enc

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a command with a timeout",
                                      "parameters": {"type": "object",
                                                     "properties": {"cmd": {"type": "string"},
                                                                    "timeout": {"type": "integer"},
                                                                    "env": {"type": "object"}},
                                                     "required": ["cmd"]}}},
]


def call(name, args, id_):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


AGENT = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "看一下 README，然后跑测试。"},
    {"role": "assistant", "content": "", "reasoning_content": "先读文件。",
     "tool_calls": [call("read_file", {"path": "README.md"}, "c1")]},
    {"role": "tool", "tool_call_id": "c1", "content": "# Abyss\nTiered inference."},
    {"role": "assistant", "content": "", "reasoning_content": "再跑两个命令。",
     "tool_calls": [call("run", {"cmd": "make test", "timeout": 60, "env": {"CI": "1", "N": [1, 2]}}, "c2"),
                    call("run", {"cmd": "ls"}, "c3")]},
    {"role": "tool", "tool_call_id": "c2", "content": "ok"},
    {"role": "tool", "tool_call_id": "c3", "content": "README.md\nsrc"},
    {"role": "assistant", "content": "测试通过。", "reasoning_content": "都好了。"},
    {"role": "user", "content": "谢谢，再总结一下。"},
]

CASES = [
    ("user only, thinking", [{"role": "user", "content": "你好"}], None, True, None),
    ("system + user, thinking, high", [{"role": "system", "content": "Be brief."},
                                       {"role": "user", "content": "Hi"}], None, True, "high"),
    ("user only, chat mode", [{"role": "user", "content": "你好"}], None, False, None),
    ("tools without system, low", [{"role": "user", "content": "列目录"}], TOOLS, True, "low"),
    ("agent loop with tools, medium", AGENT, TOOLS, True, "medium"),
    ("agent loop with tools, chat mode", AGENT, TOOLS, False, None),
    ("multi-turn without tools, reasoning dropped", [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        {"role": "user", "content": "Q3"}], None, True, "max"),
    ("mid-conversation system and developer", [
        {"role": "developer", "content": "D0"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        {"role": "system", "content": "Mid system."},
        {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
        {"role": "developer", "content": "Mid developer."}], TOOLS, True, "30"),
    ("tool result then user text", [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "", "reasoning_content": "R", "tool_calls": [call("read_file", {"path": "a"}, "x")]},
        {"role": "tool", "tool_call_id": "x", "content": "data"},
        {"role": "user", "content": "follow-up"}], TOOLS, True, "xhigh"),
    ("empty arguments", [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "", "reasoning_content": "R", "tool_calls": [call("run", {}, "y")]},
        {"role": "tool", "tool_call_id": "y", "content": "done"}], TOOLS, True, None),
]


def official_args(messages, tools, thinking, effort):
    """The official encoder's messages and keyword arguments for a case."""
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if m["role"] == "developer":
            m["role"] = "system"
    if tools:
        if msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = tools
    if isinstance(effort, str) and effort.isdigit():
        effort = int(effort)
    effort = EFFORT.get(effort, effort)
    return msgs, {"thinking_mode": "thinking" if thinking else "chat", "reasoning_effort": effort}


def official(messages, tools, thinking, effort):
    msgs, kwargs = official_args(messages, tools, thinking, effort)
    out = enc.encode_messages(msgs, **kwargs)
    return out[len(enc.bos_token):] if out.startswith(enc.bos_token) else out


def request_body(messages, tools, thinking, effort):
    """The chat request for a case, as a client sends it."""
    body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools:
        body["tools"] = tools
    if effort is not None:
        body["reasoning_effort"] = effort
    return body


def served(messages, tools, thinking, effort):
    body = request_body(messages, tools, thinking, effort)
    req = urllib.request.Request(URL + "/apply-template", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=30))["prompt"]
    return out[len("<s>"):] if out.startswith("<s>") else out


def main():
    global URL
    load(sys.argv[1])
    if len(sys.argv) > 2:
        URL = sys.argv[2]
    bad = 0
    for name, messages, tools, thinking, effort in CASES:
        a, b = official(messages, tools, thinking, effort), served(messages, tools, thinking, effort)
        if a == b:
            print(f"same    {name}")
            continue
        bad += 1
        print(f"DIFFERS {name}")
        for line in difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                         "official", "template", n=1):
            sys.stdout.write("    " + line if line.endswith("\n") else "    " + line + "\n")
    print(f"{len(CASES) - bad} of {len(CASES)} identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
