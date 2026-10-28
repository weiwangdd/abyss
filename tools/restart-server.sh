#!/bin/bash
# Restart llama-server (plan B) with expert stats, keeping the stats file. Slot 0 is saved to ~/abyss-stats/slots
# before the stop (with its context checkpoints) and restored after the start, so the next request continues the
# conversation instead of prefilling it again; NO_RESTORE=1 skips the restore, e.g. when the cache layout or the
# template or the weights changed. Like the official encoding, reasoning of earlier turns is dropped when there are no
# tools, and the default sampling is what the DeepSeek-V4.1-Flash-1017 model card recommends, temperature 1.0 and
# top-p 1.0; requests can override it.
set -e
cd ~/abyss-stats
mkdir -p slots
pid=$(pgrep -x llama-server || true)
if [ -n "$pid" ]; then
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=save" -H "Content-Type: application/json" \
    -d '{"filename": "slot0.bin"}' || echo "slot save failed"
  echo
  kill -TERM $pid
  for i in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -0 $pid 2>/dev/null && { echo "server $pid did not stop"; exit 1; }
fi
cp expert-stats.json "expert-stats-$(date +%H%M).json"
[ -f server.log ] && mv server.log "server-$(date +%H%M).log"
cd ~/src/mx-llama.cpp
ABYSS_EXPERT_STATS=$HOME/abyss-stats/expert-stats.json GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 ABYSS_DSV41_INDEXER=1 \
  ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_DECODE_STATS=100 GOMP_SPINCOUNT=100000000 \
  ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 ABYSS_DSV41_IDX_CHUNK=256 \
  nohup ./build-m2/bin/llama-server \
  -m ~/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf \
  -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot "blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU" \
  -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib 16000 \
  --checkpoint-min-step 32768 --slot-save-path $HOME/abyss-stats/slots \
  --jinja --chat-template-file $HOME/abyss-stats/tmpl/v41.jinja --no-reasoning-preserve \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 \
  --host 127.0.0.1 --port 8080 > ~/abyss-stats/server.log 2>&1 &
pid=$!
echo "started $pid"
if [ -z "$NO_RESTORE" ] && [ -f ~/abyss-stats/slots/slot0.bin ]; then
  until grep -q "listening on" ~/abyss-stats/server.log; do
    kill -0 $pid 2>/dev/null || { echo "server died, see ~/abyss-stats/server.log"; exit 1; }
    sleep 3
  done
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=restore" -H "Content-Type: application/json" \
    -d '{"filename": "slot0.bin"}' || echo "slot restore failed"
  echo
fi
