#!/bin/bash
# The compute buffers at the production settings, with and without the indexer prefill in chunks
# (ABYSS_DSV41_IDX_CHUNK): each configuration loads a server with GGML_ALLOC_TRACE=1 and stops it once it listens (or
# once it has run a prompt of FILL tokens), then alloc_peak.py shows the size of each graph reserve's buffers and what
# each buffer holds at the largest one's peak; the growths of the device memory pools are listed too. Waits for the
# production server to go idle, saves its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh
# restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
CHUNKS=${CHUNKS:-"0 256"}

pid=$(pgrep -x llama-server)
if [ -n "$pid" ]; then
  until curl -s -m 5 http://127.0.0.1:8080/slots | grep -q '"is_processing":false'; do sleep 5; done
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=save" -H "Content-Type: application/json" \
    -d '{"filename": "slot0.bin"}'
  echo
  kill -TERM "$pid"
  for i in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  echo "### production stopped $(date +%T)"
fi

for chunk in $CHUNKS; do
  log=$O/alloc-trace-c$chunk-$(date +%m%d-%H%M).log
  GGML_ALLOC_TRACE=1 ABYSS_DSV41_IDX_CHUNK=$chunk \
    ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 ABYSS_DSV41_INDEXER=1 \
    ABYSS_MOE_BIG_SPLIT=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 \
    GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 \
    "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
    -c "${CTX:-196608}" -b 65536 -ub "${UB:-2048}" -np 1 --moe-cache-mib "${MOE_CACHE:-12000}" --checkpoint-min-step 32768 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
    --host 127.0.0.1 --port 8090 > "$log" 2>&1 &
  spid=$!
  until grep -q "listening on" "$log"; do
    kill -0 $spid 2>/dev/null || { echo "server died, see $log"; break; }
    sleep 3
  done
  echo "### chunk $chunk loaded: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ') MiB"
  if [ -n "${FILL:-}" ]; then
    # a prompt of FILL tokens: what the graphs and the device pools allocate once they run
    CTX_URL=http://127.0.0.1:8090 python3 -I "$E/ctx_size.py" "$E/corpus.txt" fill "$FILL" "$O/alloc-trace-fill.json" | head -1
    echo "### chunk $chunk after $FILL tokens: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ') MiB"
  fi
  kill -TERM $spid 2>/dev/null
  wait $spid 2>/dev/null
  grep -a "cuda pool\[" "$log" | head -${TOP:-14}
  python3 -I "$E/alloc_peak.py" "$log" "${TOP:-14}"
done

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
fi
