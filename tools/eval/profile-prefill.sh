#!/bin/bash
# Where a warm long-prompt prefill spends its time: GPU kernels by category per device, idle time, host-device copies.
# The server runs as in production minus the expert stats, inside an nsys session. The first long request warms the
# Engram page cache and pays the first-request cost; only the second one is captured.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-eng/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)
REP=$O/prefill-48k

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi

ask() {
  python3 -I - "$E/corpus.txt" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({"messages": [{"role": "user", "content": corpus * 3 + "\n\n用三句话总结上面的内容。"}],
                   "max_tokens": 1, "cache_prompt": False}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
print("prompt %d tok %.1f tok/s (%.1f s)" % (t["prompt_n"], t["prompt_per_second"], t["prompt_ms"] / 1000))
PY
}

log=$O/server-nsys-$(date +%m%d-%H%M).log
ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 \
  nsys launch --session-new=abyss --trace=cuda,osrt --cuda-graph-trace=node "$B/llama-server" -m "$M" \
    -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
    -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib 14000 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" \
    --host 127.0.0.1 --port 8080 > "$log" 2>&1 &
for i in $(seq 120); do
  grep -q "listening on" "$log" && break
  sleep 5
done
grep -q "listening on" "$log" || { echo "server did not come up in 10 min, see $log"; tail -5 "$log"; exit 1; }

echo "### warm-up: $(ask)"
nsys start --session=abyss --output="$REP" --force-overwrite=true
echo "### captured: $(ask)"
nsys stop --session=abyss
nsys shutdown --session=abyss --kill=sigterm
sleep 10

nsys export --type sqlite --output "$REP.sqlite" --force-overwrite=true "$REP.nsys-rep" > /dev/null
python3 -I "$E/nsys_breakdown.py" "$REP.sqlite"
echo "### done"
