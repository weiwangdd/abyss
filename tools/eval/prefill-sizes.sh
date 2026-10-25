#!/bin/bash
# Prefill of fresh prompts of a few sizes, the way an outside benchmark measures it: prompt cache off, temperature 0,
# one request at a time, the median of REPS (default 3) per size. SIZES lists the prompt sizes in characters of the
# corpus (about 3 characters a token). Each config in CONFIGS (space-separated, each a comma-separated list of
# VAR=value) runs its own server with the production settings; a warm-up request pays the first-request cost.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)
SIZES=${SIZES:-"1900 7700 15500"}
CONFIGS=${CONFIGS:-"ABYSS_CKPT_TAIL_MERGE=0"}

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi
if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi

ask() {  # $1 = characters of the corpus
  python3 -I - "$E/corpus.txt" "$1" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1], encoding="utf-8").read()
text = (corpus * 4)[:int(sys.argv[2])]
body = json.dumps({"messages": [{"role": "user", "content": text + "\n\n用一句话总结上面的内容。"}],
                   "max_tokens": 1, "temperature": 0, "cache_prompt": False}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
print(t["prompt_n"], "%.2f" % (t["prompt_ms"] / 1000))
PY
}

i=0
for cfg in $CONFIGS; do
  env_args=(${cfg//,/ })
  log=$O/server-sizes-$i-$(date +%m%d-%H%M).log
  env "${env_args[@]}" ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 ABYSS_ENGRAM_PREFETCH=8 \
    ABYSS_CED_PREFILL=1 ABYSS_DSV41_INDEXER=1 GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 "$B/llama-server" -m "$M" \
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
  echo "### $cfg"
  ask 20000 > /dev/null
  for size in $SIZES; do
    runs=$(for r in $(seq ${REPS:-3}); do ask "$size"; done | sort -k2 -n)
    n=$(echo "$runs" | wc -l)
    echo "$runs" | sed -n "$(( (n + 1) / 2 ))p" | awk '{ printf "  %6d tokens: median %.2f s, %.0f tok/s\n", $1, $2, $1 / $2 }'
  done
  kill -TERM $pid
  wait $pid
  i=$((i + 1))
done
echo "### done"
