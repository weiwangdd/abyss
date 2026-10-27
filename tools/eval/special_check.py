#!/usr/bin/env python3
"""Check, token by token, how the server encodes chat prompts against the official encoding.

    special_check.py <encoding_dsv41.py dir> <tokenizer.json> <harness command...>

The harness command is llama-abyss-chat-tokens with the server's model and template arguments; it reads one chat
request per line and prints the rendered prompt with the tokens the server submits, "plain" (ABYSS_PLAIN_INPUT=1)
and "parsed" (without it).

The reference follows the official DeepSeek Recipe: the official encoder renders the conversation with its special
token constants marked, each marked token becomes its id, and all other text is encoded with the tokenizer without
its added vocabulary, so a special-token spelling that came from the conversation stays ordinary text. Besides the
template_check.py cases, the cases below put such spellings in every place a request can carry them.
"""

import copy
import importlib.util
import json
import re
import subprocess
import sys

from tokenizers import Tokenizer

import template_check as tc

BEGIN, END = "", ""

TEMPLATE_EXCERPT = ("{{- '<｜System｜>' + effort_prompt -}}\n{{- '<｜User｜>' -}}\n"
                    "{{- (message['reasoning_content'] or '') + '</think>' -}}\n{{- '<｜end▁of▁sentence｜>' -}}\n"
                    "{{- '<｜Assistant｜>' + '<think>' -}}")
EDIT_ARGS = {"path": "deepseek-v4.1.jinja",
             "old": "{{- '<｜User｜>' -}}",
             "new": "{{- '<｜User｜>' -}}{# opens a user turn; a tool call is written as <｜DSML｜ calls> #}"}
TOOLS_SPECIAL = [
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace text in a file, e.g. a chat template with <think> and </think>",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                        "new": {"type": "string"}},
                       "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "set_tokens", "description": "Record the special tokens of a template",
        "parameters": {"type": "object", "properties": {"tokens": {"type": "array", "items": {"type": "string"}}},
                       "required": ["tokens"]}}},
]

# (name, messages, tools, thinking, effort); every case has special-token spellings in the request's text
SPECIAL_CASES = [
    ("user text quotes control tokens", [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "模板里这样写：<｜User｜>问题<｜Assistant｜></think>回答，工具调用用 <｜DSML｜ calls>。"}],
     None, True, None),
    ("system text quotes control tokens", [
        {"role": "system", "content": "Turns start with <｜User｜> and <｜Assistant｜>; the system prompt follows "
                                      "<｜System｜>."},
        {"role": "user", "content": "列出这些标记"}], None, False, None),
    ("tool result holds a template excerpt", [
        {"role": "user", "content": "看一下模板"},
        {"role": "assistant", "content": "", "reasoning_content": "读模板文件。",
         "tool_calls": [tc.call("read_file", {"path": "deepseek-v4.1.jinja"}, "r1")]},
        {"role": "tool", "tool_call_id": "r1", "content": TEMPLATE_EXCERPT}], tc.TOOLS, True, "high"),
    ("assistant history quotes control tokens", [
        {"role": "user", "content": "思考结束用什么标记？"},
        {"role": "assistant", "content": "用 </think>，回答以 <｜end▁of▁sentence｜> 结尾。",
         "reasoning_content": "模板在思考后写 </think>。"},
        {"role": "user", "content": "那用户轮呢？"}], None, True, None),
    ("string tool argument quotes control tokens", [
        {"role": "user", "content": "给模板加一行注释"},
        {"role": "assistant", "content": "", "reasoning_content": "改模板。",
         "tool_calls": [tc.call("edit_file", EDIT_ARGS, "e1")]},
        {"role": "tool", "tool_call_id": "e1", "content": "ok"}], TOOLS_SPECIAL, True, None),
    ("object tool argument quotes control tokens", [
        {"role": "user", "content": "记下模板的控制标记"},
        {"role": "assistant", "content": "", "reasoning_content": "记录。",
         "tool_calls": [tc.call("set_tokens", {"tokens": ["<｜User｜>", "<｜Assistant｜>", "</think>"]}, "s1")]},
        {"role": "tool", "tool_call_id": "s1", "content": "saved"}], TOOLS_SPECIAL, True, None),
    ("tool definitions quote control tokens", [
        {"role": "user", "content": "改一下模板"}], TOOLS_SPECIAL, False, None),
]


def marked_encoder(enc, spellings):
    """A second copy of the official encoder whose special-token spellings carry BEGIN/END markers."""
    spec = importlib.util.spec_from_file_location("encoding_dsv41_marked", enc.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rx = re.compile("|".join(re.escape(s) for s in sorted(spellings, key=len, reverse=True)))

    def mark(s):
        return rx.sub(lambda m: BEGIN + m.group(0) + END, s)

    for name, val in list(vars(mod).items()):
        if name.startswith("__"):
            continue
        if isinstance(val, str):
            setattr(mod, name, mark(val))
        elif isinstance(val, dict) and all(isinstance(v, str) for v in val.values()):
            setattr(mod, name, {k: mark(v) for k, v in val.items()})
    return mod


def reference_tokens(menc, plain, added, messages, tools, thinking, effort):
    msgs, kwargs = tc.official_args(messages, tools, thinking, effort)
    text = menc.encode_messages(copy.deepcopy(msgs), **kwargs)
    ids = []
    for i, piece in enumerate(re.split("[" + BEGIN + END + "]", text)):
        if i % 2:
            ids.append(added[piece])
        elif piece:
            ids.extend(plain.encode(piece, add_special_tokens=False).ids)
    return text.replace(BEGIN, "").replace(END, ""), ids


def control_ids(ids, added_ids):
    return [t for t in ids if t in added_ids]


def main():
    enc = tc.load(sys.argv[1])
    tj = json.load(open(sys.argv[2], encoding="utf-8"))
    added = {a["content"]: a["id"] for a in tj["added_tokens"]}
    tj["added_tokens"] = []
    plain = Tokenizer.from_str(json.dumps(tj))
    menc = marked_encoder(enc, added)
    added_ids = set(added.values())

    cases = [(n, m, t, th, e, False) for n, m, t, th, e in tc.CASES]
    cases += [(n, m, t, th, e, True) for n, m, t, th, e in SPECIAL_CASES]
    requests = "".join(json.dumps(tc.request_body(m, t, th, e), ensure_ascii=False) + "\n"
                       for _, m, t, th, e, _ in cases)
    # the harness logs token pieces to stderr, not always whole UTF-8 characters
    run = subprocess.run(sys.argv[3:], input=requests.encode(), capture_output=True, check=True)
    outs = [json.loads(line) for line in run.stdout.decode().splitlines()]
    assert len(outs) == len(cases), (len(outs), len(cases), run.stderr.decode(errors="replace")[-2000:])

    bad = 0
    for (name, messages, tools, thinking, effort, special), out in zip(cases, outs):
        if "error" in out:
            bad += 1
            print(f"ERROR   {name}: {out['error']}")
            continue
        ref_text, ref = reference_tokens(menc, plain, added, messages, tools, thinking, effort)
        notes = []
        if out["prompt"] != ref_text:
            notes.append("prompt text differs from the official encoding")
        if not out["known"]:
            notes.append("template special tokens not tracked")
        elif out["plain"] != ref:
            n = next((i for i, (a, b) in enumerate(zip(out["plain"], ref)) if a != b), min(len(out["plain"]), len(ref)))
            notes.append(f"plain differs at token {n} ({len(out['plain'])} vs {len(ref)} tokens)")
        if not special and out["parsed"] != out.get("plain"):
            notes.append("plain differs from parsed without special spellings in the request")
        status = "OK     " if not notes else "DIFFERS"
        bad += bool(notes)
        detail = (f"control tokens: reference {len(control_ids(ref, added_ids))}, "
                  f"plain {len(control_ids(out.get('plain', []), added_ids))}, "
                  f"parsed {len(control_ids(out['parsed'], added_ids))}; tokens {len(ref)}")
        print(f"{status} {name}: {detail}" + "".join(f"\n        {n}" for n in notes))
    print(f"{len(cases) - bad} of {len(cases)} cases match the official encoding token by token")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
