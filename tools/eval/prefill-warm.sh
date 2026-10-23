#!/bin/bash
# Warm long-prompt prefill of a server build: one long request warms the Engram page cache and pays the
# first-request cost, the second one is the measurement. Production settings, without the expert stats.
# REPEAT (default 3) sets how many copies of the corpus make the prompt, about 16k tokens each; MOE_CACHE_MIB the
# expert cache size.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-eng/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi

ask() {  # $1 = max_tokens
  python3 -I - "$E/corpus.txt" "$1" "${REPEAT:-3}" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({"messages": [{"role": "user", "content": corpus * int(sys.argv[3]) + "\n\n用三句话总结上面的内容。"}],
                   "max_tokens": int(sys.argv[2]), "temperature": 0, "cache_prompt": False}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
t = r["timings"]; m = r["choices"][0]["message"]
print("prompt %d tok %.1f tok/s (%.1f s) | gen %d tok %.1f tok/s" % (
    t["prompt_n"], t["prompt_per_second"], t["prompt_ms"] / 1000, t["predicted_n"], t["predicted_per_second"]))
if m.get("content"):
    print("content:", m["content"].replace("\n", " ")[:300])
PY
}

log=$O/server-warm-$(date +%m%d-%H%M).log
ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 "$B/llama-server" -m "$M" \
  -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
  -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib ${MOE_CACHE_MIB:-14000} \
  --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" \
  --host 127.0.0.1 --port 8080 > "$log" 2>&1 &
pid=$!
until grep -q "listening on" "$log"; do
  kill -0 $pid 2>/dev/null || { echo "server died, see $log"; exit 1; }
  sleep 5
done
echo "### warm-up:  $(ask 1)"
echo "### measured: $(ask 600)"
kill -TERM $pid
wait $pid
echo "### done"
