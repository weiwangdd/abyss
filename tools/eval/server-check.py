"""Send a long and a short chat request to the local llama-server and print its timings."""
import json, sys, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"

def ask(content, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.load(r)
    t = resp.get("timings", {})
    msg = resp["choices"][0]["message"]
    print("prompt %6d tok %7.1f tok/s | gen %4d tok %5.1f tok/s | %s" % (
        t.get("prompt_n", 0), t.get("prompt_per_second", 0), t.get("predicted_n", 0),
        t.get("predicted_per_second", 0), (msg.get("content") or "")[:80].replace("\n", " ")))

corpus = open(sys.argv[1], encoding="utf-8").read()
ask(corpus * 3 + "\n\n用三句话总结上面的内容。", 400)
ask("写一个 Python 函数，判断一个数是不是质数，并解释思路。", 400)
ask("写一个 Python 函数，判断一个数是不是质数，并解释思路。", 400)
