#!/usr/bin/env python3
"""Steps of ctx-size.sh against a running server.

    ctx_size.py <corpus> same OUT     about 96k tokens of the corpus and a question, 32 greedy tokens with top-20
                                      logprobs saved to OUT (the same prompt for every context size)
    ctx_size.py <corpus> fill N OUT   a passphrase, the corpus repeated up to about N - 3000 tokens and a question
                                      about the passphrase; 256 greedy tokens past the end of text (decode speed at
                                      that context), the answer checked, top-20 logprobs saved to OUT
    ctx_size.py <corpus> compare A B
"""

import json
import math
import os
import sys
import time
import urllib.request

URL = os.environ.get("CTX_URL", "http://127.0.0.1:8080")
PASS = "青石桥 7429"
NEEDLE = f"开始之前先记住一件事：通行暗号是「{PASS}」。下面是一份很长的材料。"
QUESTION = "材料最开头提到的通行暗号是什么？只回答暗号本身。"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def corpus_text(corpus, n_tokens):
    # the corpus repeated and cut to about n_tokens tokens
    per = len(post("/tokenize", {"content": corpus})["tokens"])
    reps = n_tokens // per + 1
    text = corpus * reps
    return text[:int(len(text) * n_tokens / (per * reps))]


def ask(content, **extra):
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True, "top_logprobs": 20}
    body.update(extra)
    t0 = time.time()
    r = post("/v1/chat/completions", body)
    return r, time.time() - t0


def report(name, r, wall):
    t = r["timings"]
    print(f"{name}: prompt {t['prompt_n']} tok in {t['prompt_ms'] / 1000:.1f} s ({t['prompt_per_second']:.0f} tok/s), "
          f"gen {t['predicted_n']} tok at {t['predicted_per_second']:.1f} tok/s, wall {wall:.1f} s")


def main():
    corpus = open(sys.argv[1], encoding="utf-8").read()
    cmd = sys.argv[2]
    try:
        if cmd == "same":
            r, wall = ask(corpus_text(corpus, 96000) + "\n\n上面这份材料讲的是什么软件？用一句话回答。", max_tokens=32)
            report("same", r, wall)
            json.dump(r["choices"][0]["logprobs"]["content"], open(sys.argv[3], "w"))
        elif cmd == "fill":
            n = int(sys.argv[3])
            r, wall = ask(NEEDLE + "\n\n" + corpus_text(corpus, n - 3000) + "\n\n" + QUESTION,
                          max_tokens=256, ignore_eos=True)
            report(f"fill {n}", r, wall)
            content = r["choices"][0]["message"]["content"] or ""
            ok = PASS.replace(" ", "") in content[:80].replace(" ", "")
            print(f"  answer {'correct' if ok else 'WRONG'}: {content[:40]!r}")
            json.dump(r["choices"][0]["logprobs"]["content"][:32], open(sys.argv[4], "w"))
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
    except Exception as e:  # the server running out of memory ends the request
        print(f"{cmd} failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
