#!/usr/bin/env python3
"""Greedy code generation for the DSpark experiment: a few programming prompts without thinking, MAX_TOKENS each,
against the server at DECODE_URL. Writes the answers to OUT as JSON, so runs with and without the draft model can be
compared token by token; the server log has the decode speed and the draft acceptance of each request.

    dspark_code.py OUT [MAX_TOKENS]
"""
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("DECODE_URL", "http://127.0.0.1:8080")
PROMPTS = [
    "用 Python 实现一个线程安全的 LRU 缓存类，支持 get、put 和过期时间，并写几个单元测试。",
    "写一个 Go 程序：读取 CSV 文件，按某一列分组，输出每组其他数值列的平均值。",
    "用 C++17 实现一个固定大小的环形缓冲区模板类，支持迭代器，并给出使用示例。",
]


def main():
    out_path = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    answers = []
    for prompt in PROMPTS:
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            res = json.load(r)
        dt = time.time() - t0
        n = res["usage"]["completion_tokens"]
        answers.append({"prompt": prompt, "content": res["choices"][0]["message"].get("content") or "", "tokens": n,
                        "seconds": dt})
        print(f"{n:4d} tokens in {dt:5.1f} s ({n / dt:5.1f} tok/s wall): {prompt[:24]}", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
