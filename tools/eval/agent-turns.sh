#!/bin/bash
# Multi-turn agent session against a server with the production settings (agent_turns.py): per turn, what the prompt
# cache reused and what was processed again. TURNS (default 6), REPEAT (corpus copies in turn 1, default 3) and
# MAX_TOKENS (default 512) are passed on; EXTRA adds server arguments. CAPTURE=1 runs the server inside nsys and
# captures the last turn's prefill into $OUT/agent-turn.nsys-rep, then prints its breakdown. ABYSS_MOE_BIG_SPLIT,
# ABYSS_CKPT_TAIL_MERGE and ABYSS_CED_ALIGN default to the production values; set them in the environment to compare.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi
if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi

log=$O/server-agent-$(date +%m%d-%H%M).log
REP=$O/agent-turn
NSYS=()
[ -n "$CAPTURE" ] && NSYS=(nsys launch --session-new=abyss --trace=cuda,osrt --cuda-graph-trace=node)
ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 ABYSS_DSV41_INDEXER=1 \
  ABYSS_MOE_BIG_SPLIT=${ABYSS_MOE_BIG_SPLIT:-1} ABYSS_CKPT_TAIL_MERGE=${ABYSS_CKPT_TAIL_MERGE:-2700} ABYSS_CED_ALIGN=${ABYSS_CED_ALIGN:-1} \
  GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 "${NSYS[@]}" "$B/llama-server" -m "$M" \
  -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
  -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib 14000 \
  --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" \
  --host 127.0.0.1 --port 8080 $EXTRA > "$log" 2>&1 &
pid=$!
until grep -q "listening on" "$log"; do
  kill -0 $pid 2>/dev/null || { echo "server died, see $log"; exit 1; }
  sleep 5
done
python3 -I "$E/agent_turns.py" "$E/corpus.txt" --turns "${TURNS:-6}" --repeat "${REPEAT:-3}" --max-tokens "${MAX_TOKENS:-512}" \
  ${CAPTURE:+--capture "$REP"}
if [ -n "$CAPTURE" ]; then
  nsys shutdown --session=abyss --kill=sigterm
  wait $pid
  nsys export --type sqlite --output "$REP.sqlite" --force-overwrite=true "$REP.nsys-rep" > /dev/null
  python3 -I "$E/nsys_breakdown.py" "$REP.sqlite"
else
  kill -TERM $pid
  wait $pid
fi
echo "### done, server log $log"
