#!/usr/bin/env python3
"""Steps of restart-prefill.sh against a running server.

    restart_prefill.py <corpus> fresh      long multi-message conversation from scratch, print the prefill time
    restart_prefill.py <corpus> save F     save slot 0 to file F (under the server's --slot-save-path)
    restart_prefill.py <corpus> restore F  restore slot 0 from file F
    restart_prefill.py <corpus> next OUT   the conversation plus one more question, 32 greedy tokens with top-20
                                           logprobs saved to OUT, print what the prompt cache reused
    restart_prefill.py <corpus> compare A B
"""

import json
import math
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def conversation(corpus):
    # about 96k tokens in 16 user messages, as a long agent session has many
    text = corpus * 6
    n = 16
    step = len(text) // n
    messages = [{"role": "system", "content": "你是一个代码助手。"}]
    for i in range(n):
        messages.append({"role": "user", "content": f"第 {i + 1} 段材料：\n\n" + text[i * step:(i + 1) * step]})
        messages.append({"role": "assistant", "content": f"收到第 {i + 1} 段，已经读完。"})
    messages.append({"role": "user", "content": "以上材料分几部分？"})
    return messages


def main():
    corpus = open(sys.argv[1], encoding="utf-8").read()
    cmd = sys.argv[2]
    messages = conversation(corpus)
    if cmd == "fresh":
        t0 = time.time()
        t = post("/v1/chat/completions", {"messages": messages, "max_tokens": 1, "temperature": 0,
                                          "cache_prompt": True})["timings"]
        print(f"fresh: prompt {t['prompt_n']} tok, {t['prompt_ms'] / 1000:.1f} s "
              f"({t['prompt_per_second']:.0f} tok/s), wall {time.time() - t0:.1f} s")
    elif cmd in ("save", "restore"):
        t0 = time.time()
        r = post(f"/slots/0?action={cmd}", {"filename": sys.argv[3]})
        print(f"{cmd}: {json.dumps(r, ensure_ascii=False)}, wall {time.time() - t0:.1f} s")
    elif cmd == "next":
        messages.append({"role": "assistant", "content": "一共 16 段。"})
        messages.append({"role": "user", "content": "第 3 段讲了什么？用三句话说。"})
        r = post("/v1/chat/completions", {"messages": messages, "max_tokens": 32, "temperature": 0,
                                          "cache_prompt": True, "logprobs": True, "top_logprobs": 20})
        t = r["timings"]
        print(f"next: reused {t.get('cache_n', 0)}, processed {t['prompt_n']} tok in {t['prompt_ms'] / 1000:.1f} s")
        json.dump(r["choices"][0]["logprobs"]["content"], open(sys.argv[3], "w"))
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
