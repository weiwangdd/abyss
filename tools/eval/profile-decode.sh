#!/bin/bash
# Where decode at a long context spends its time: GPU kernels by category per device, idle time, host-device copies
# (expert cache uploads). The server runs as in production minus the expert stats, inside an nsys session. A first
# request prefills the ~48k-token prompt and keeps it in the prompt cache; the captured second request reuses it and
# generates GEN (default 256) tokens.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-idx/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)
REP=$O/decode-48k

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi

# a GPU whose link trained below Gen5 uploads experts at half speed (GPU1 came up at Gen4 once, 2026-10-24; a reboot fixed it)
if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi

ask() {  # $1 = max_tokens
  python3 -I - "$E/corpus.txt" "$1" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({"messages": [{"role": "user", "content": corpus * 3 + "\n\n详细总结上面的内容。"}],
                   "max_tokens": int(sys.argv[2]), "temperature": 0, "cache_prompt": True}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
print("prompt %d tok (%.1f s) | gen %d tok %.2f tok/s (%.1f ms/tok)" % (
    t["prompt_n"], t["prompt_ms"] / 1000, t["predicted_n"], t["predicted_per_second"], t["predicted_ms"] / t["predicted_n"]))
PY
}

log=$O/server-nsys-dec-$(date +%m%d-%H%M).log
ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 ABYSS_DSV41_INDEXER=1 GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 \
  nsys launch --session-new=abyss --trace=cuda,osrt --cuda-graph-trace=node "$B/llama-server" -m "$M" \
    -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
    -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib ${MOE_CACHE_MIB:-14000} \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" \
    --host 127.0.0.1 --port 8080 > "$log" 2>&1 &
for i in $(seq 120); do
  grep -q "listening on" "$log" && break
  sleep 5
done
grep -q "listening on" "$log" || { echo "server did not come up in 10 min, see $log"; tail -5 "$log"; exit 1; }

echo "### warm-up:  $(ask 64)"
nsys start --session=abyss --output="$REP" --force-overwrite=true
echo "### captured: $(ask ${GEN:-256})"
nsys stop --session=abyss
nsys shutdown --session=abyss --kill=sigterm
sleep 10
grep -E "MoE cache size|slots \(|llama_moe_cache" "$log" | head

nsys export --type sqlite --output "$REP.sqlite" --force-overwrite=true "$REP.nsys-rep" > /dev/null
python3 -I "$E/nsys_decode.py" "$REP.sqlite" "${GEN:-256}"
echo "### done"
