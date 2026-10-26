#!/bin/bash
# A conversation (slot_takeover.py, 36 rounds of about 1,700 tokens, one request each so the checkpoint list fills
# and gets thinned) that a request sharing only its first 22 rounds takes over, as Hermes's review of its own turn
# does, and then the conversation's next turn. Three servers with the production settings and
# --checkpoint-min-step 32768: the slot not saved and the checkpoints thinned at that step (before 0017), the slot
# saved before the takeover (ABYSS_SLOT_SAVE_LOST, default 2048), and also thinned at THIN_STEP (default 8192).
# The next turn's outputs must match across the three.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)
N=36
K=${K:-22}

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi

start() {  # environment assignments for the server
  log=$O/server-takeover-$(date +%m%d-%H%M%S).log
  env "$@" ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 \
    ABYSS_DSV41_INDEXER=1 ABYSS_MOE_BIG_SPLIT=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 GGML_CUDA_Q8_1_CACHE=0 \
    GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 \
    "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
    -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib 14000 --checkpoint-min-step 32768 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
    --host 127.0.0.1 --port 8080 > "$log" 2>&1 &
  pid=$!
  until grep -q "listening on" "$log"; do
    kill -0 $pid 2>/dev/null || { echo "server died, see $log"; exit 1; }
    sleep 3
  done
}
stop() {
  grep -a "saving it in the prompt cache\|swapping it in" "$log" | sed 's/^/    server: /'
  kill -TERM $pid
  wait $pid
}
step() { python3 -I "$E/slot_takeover.py" "$E/corpus.txt" "$@"; }
run() {  # name, then environment assignments
  name=$1
  shift
  echo "### $name"
  start "$@"
  step grow $N
  step review $K $N
  step next $N "$O/takeover-$name.json"
  stop
}

run before ABYSS_SLOT_SAVE_LOST=0
run save ABYSS_SLOT_SAVE_LOST=2048
run save-thin ABYSS_SLOT_SAVE_LOST=2048 ABYSS_CKPT_THIN_STEP=${THIN_STEP:-8192}
step compare "$O/takeover-before.json" "$O/takeover-save.json"
step compare "$O/takeover-before.json" "$O/takeover-save-thin.json"
echo "### done"
