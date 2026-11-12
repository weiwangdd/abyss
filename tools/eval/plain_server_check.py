#!/usr/bin/env python3
"""Check the server's chat path with ABYSS_PLAIN_INPUT on a small CPU model: the tokens /apply-template reports
("with_tokens") are the ones a chat completion submits, and a control-token spelling in the request stays text.

    plain_server_check.py <server url> <control token spelling> <its id>
"""
import json
import sys
import urllib.request

url, spelling, tok = sys.argv[1], sys.argv[2], int(sys.argv[3])


def post(path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


bad = 0
for name, messages in [
        ("plain text", [{"role": "user", "content": "Once upon a time there was a little dog."}]),
        ("spelling in the user text", [{"role": "user", "content": f"The story ends with {spelling} and then {spelling}."}]),
        ("spelling in system and history", [
            {"role": "system", "content": f"Stories end with {spelling}."},
            {"role": "user", "content": "Tell me a story."},
            {"role": "assistant", "content": f"Once upon a time. {spelling}"},
            {"role": "user", "content": "Another one."}])]:
    body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
    t = post("/apply-template", dict(body, with_tokens=True))
    n_ctrl = t["tokens"].count(tok)
    n_text = t["prompt"].count(spelling)
    c = post("/v1/chat/completions", dict(body, max_tokens=4, temperature=0, cache_prompt=False))
    n_prompt = c["usage"]["prompt_tokens"]
    ok = n_prompt == len(t["tokens"])
    bad += not ok
    print(f"{'OK     ' if ok else 'DIFFERS'} {name}: {len(t['tokens'])} tokens from /apply-template, the completion took "
          f"{n_prompt}; {spelling!r} is in the prompt {n_text} times, as its token {n_ctrl} times")
sys.exit(1 if bad else 0)
