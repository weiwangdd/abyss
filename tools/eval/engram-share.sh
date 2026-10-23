#!/bin/bash
# How much of a long-prompt prefill the Engram gather costs, with the CED-aware prefill on.
# The same prompt goes twice per server (cold, then warm page cache); LLAMA_DSV41_NO_ENGRAM=1 drops the gather.
# The prompt cache is turned off per request, so the second request prefills the whole prompt again.
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
                   "max_tokens": 16, "cache_prompt": False}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
print("prompt %d tok %.1f tok/s (%.1f s)" % (t["prompt_n"], t["prompt_per_second"], t["prompt_ms"] / 1000))
PY
}

for no_engram in 0 1; do
  log=$O/server-engram$no_engram-$(date +%m%d-%H%M).log
  LLAMA_DSV41_NO_ENGRAM=$no_engram ABYSS_CED_PREFILL=1 GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 "$S" -m "$M" \
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
  echo "### NO_ENGRAM=$no_engram, cold: $(ask)"
  echo "### NO_ENGRAM=$no_engram, warm: $(ask)"
  kill -TERM $pid
  wait $pid
done
echo "### done"
