#!/bin/bash
# M5: long-prompt prefill on llama-server with the CED-aware prefill off and on.
# The prompt goes into one decode call (-b 65536), since each call only trims the decoder FFNs behind its own end.
# Greedy output of both runs is printed for an end-to-end check. Expert stats stay off: their callback syncs per layer.
O=${OUT:-$HOME/abyss-eval}
S=${SERVER:-$HOME/src/mx-llama.cpp/build-ced/bin/llama-server}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi

ask() {
  python3 -I - "$E/corpus.txt" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({"messages": [{"role": "user", "content": corpus * 3 + "\n\n用三句话总结上面的内容。"}],
                   "max_tokens": 600, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
t = r["timings"]; m = r["choices"][0]["message"]
print("prompt %d tok %.1f tok/s | gen %d tok %.1f tok/s" % (t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"]))
print("content:", (m.get("content") or "").replace("\n", " ")[:400])
PY
}

for ced in 0 1; do
  log=$O/server-ced$ced-$(date +%m%d-%H%M).log
  ABYSS_CED_PREFILL=$ced GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 "$S" -m "$M" \
    -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
    -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib 14000 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" \
    --host 127.0.0.1 --port 8080 > "$log" 2>&1 &
  pid=$!
  until grep -q "listening on" "$log"; do
    kill -0 $pid 2>/dev/null || { echo "server died, see $log"; exit 1; }
    sleep 5
  done
  echo "### CED $ced"
  ask
  kill -TERM $pid
  wait $pid
done
echo "### done"
