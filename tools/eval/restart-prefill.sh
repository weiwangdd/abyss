#!/bin/bash
# The first request after a server restart: a long conversation with many user messages (restart_prefill.py, about
# 96k tokens in 16 messages) prefilled from scratch, which the mid-prompt checkpoints split into several calls; and
# the same conversation continued from a slot saved before the restart instead. Three servers with the production
# settings: the default checkpoint spacing with --slot-save-path (fresh prefill, save, one more turn), a restart that
# restores the slot and asks the same turn (the outputs must match), and CKPT_STEP (default 32768) as
# --checkpoint-min-step (fresh prefill).
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)
SLOTS=${SLOTS:-$O/slots}
mkdir -p "$SLOTS"

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi

start() {  # extra server arguments
  log=$O/server-restart-$(date +%m%d-%H%M%S).log
  ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 ABYSS_DSV41_INDEXER=1 \
    ABYSS_MOE_BIG_SPLIT=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 \
    "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
    -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib 14000 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" \
    --host 127.0.0.1 --port 8080 "$@" > "$log" 2>&1 &
  pid=$!
  until grep -q "listening on" "$log"; do
    kill -0 $pid 2>/dev/null || { echo "server died, see $log"; exit 1; }
    sleep 3
  done
}
stop() {
  kill -TERM $pid
  wait $pid
}
step() { python3 -I "$E/restart_prefill.py" "$E/corpus.txt" "$@"; }

echo "### default checkpoint spacing"
start --slot-save-path "$SLOTS"
step fresh
step save abyss-slot0.bin
ls -la "$SLOTS/abyss-slot0.bin" | awk '{ printf "slot file %.2f GiB\n", $5 / 2^30 }'
step next "$O/restart-before.json"
stop

echo "### restart, restore the slot"
start --slot-save-path "$SLOTS"
step restore abyss-slot0.bin
step next "$O/restart-after.json"
stop
step compare "$O/restart-before.json" "$O/restart-after.json"

echo "### --checkpoint-min-step ${CKPT_STEP:-32768}"
start --checkpoint-min-step "${CKPT_STEP:-32768}"
step fresh
stop
echo "### done"
