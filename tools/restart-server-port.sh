#!/bin/bash
# Restart llama-server on the port of the patches to upstream llama.cpp (~/src/llama.cpp-abyss, branch abyss-port; the
# patches and what each switch does are in llama.cpp/README.md and CLAUDE.md), with the settings measured there: the
# weights in upstream's layout (tools/convert/to_upstream.py), the MoE cache at 20900 MiB with 0.7 of its misses left
# to the CPU, and the template with the V4.1 tag literal upstream detects. The counterpart of restart-server.sh
# (~/abyss-stats/restart.sh, production on mx-llama.cpp); on blackwell it goes to ~/abyss-stats/restart-port.sh. The
# server runs from build-prod, a build of its own, so that the test build (build) can be rebuilt while it serves.
#
# Slot 0 is saved before the stop and restored after the start, as there, but into a file of its own: the two servers
# write different slot formats, so a slot of mx-llama.cpp goes to slot0.bin and one of the port to slot0-port.bin,
# whichever server is running when this stops it, and only slot0-port.bin is restored. NO_RESTORE=1 skips the restore.
# The port has no expert statistics (ABYSS_EXPERT_STATS is not ported); the decode statistics are on.
set -e
src=$HOME/src/llama.cpp-abyss
model=$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF-upstream
cd ~/abyss-stats
mkdir -p slots
pids=$(pgrep -x llama-server || true)
if [ -n "$pids" ]; then
  # the server on 8080 writes its own format
  slot=slot0.bin
  for pid in $pids; do
    case $(readlink "/proc/$pid/exe" 2>/dev/null) in "$src"/*) slot=slot0-port.bin ;; esac
  done
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=save" -H "Content-Type: application/json" \
    -d "{\"filename\": \"$slot\"}" || echo "slot save failed"
  echo
  # every server goes, a test server left behind too; the cleanup can hang past a minute, a second signal ends it at
  # once (the slot is saved already), and a server that still holds on is killed
  for pid in $pids; do
    kill -TERM $pid 2>/dev/null || continue
    for i in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
    if kill -0 $pid 2>/dev/null; then
      echo "server $pid still stopping, signalling again"
      kill -TERM $pid 2>/dev/null || true
      for i in $(seq 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done
    fi
    if kill -0 $pid 2>/dev/null; then
      echo "server $pid killed"
      kill -KILL $pid 2>/dev/null || true
      sleep 2
    fi
  done
  # the driver hands back the device memory a few seconds after the process is gone
  for i in $(seq 60); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }' && break
    sleep 1
  done
fi
[ -f server.log ] && mv server.log "server-$(date +%H%M).log"
# upstream takes the V4.1 tag names (leading space) and message rendering only from a template holding "｜DSML｜ calls>"
# literally; the production template builds its tags from dsml_token, so the copy gets the literal in a comment
{ printf '{#- ｜DSML｜ calls> -#}'; cat "$HOME/abyss-stats/tmpl/v41.jinja"; } > "$HOME/abyss-stats/tmpl/v41-upstream.jinja"
cd "$src"
GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 \
  ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CKPT_COUNT_START=1 ABYSS_GEN_CKPT=256 ABYSS_CKPT_THIN_STEP=8192 \
  ABYSS_SAMPLE_FAST=1 ABYSS_DIST_VEC=1 ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_KEEPALIVE_MS=4 \
  ABYSS_CED_PREFILL=1 ABYSS_CED_ALIGN=1 \
  ABYSS_FUSED_SIBLING=1 ABYSS_MOE_DEFER=1 ABYSS_MOE_ADMIT_GATE=1 ABYSS_DECODE_STATS=100 ABYSS_DSV41_CHUNK=512 \
  ABYSS_MOE_BIG_SPLIT=1 ABYSS_WEIGHT_NOWAIT=1 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x \
  ABYSS_PLAIN_INPUT=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_HEAD_SPLIT_ONLY=1 ABYSS_MMVQ_TAIL=1 \
  ABYSS_DSV41_WINDOW_ROWS=1 ABYSS_HC_GATE_FUSE=1 ABYSS_MOE_BIG_CACHE=1 \
  nohup ./build-prod/bin/llama-server \
  -m $model/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf \
  -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot "blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU,blk\.12\.ffn_(gate|up)_exps\.=CUDA0" \
  -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib 20900 --moe-cache-cpu-share 0.7 \
  --checkpoint-min-step 32768 --slot-save-path $HOME/abyss-stats/slots \
  --jinja --chat-template-file $HOME/abyss-stats/tmpl/v41-upstream.jinja --no-reasoning-preserve \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 \
  --host 127.0.0.1 --port 8080 > ~/abyss-stats/server.log 2>&1 &
pid=$!
echo "started $pid"
if [ -z "$NO_RESTORE" ] && [ -f ~/abyss-stats/slots/slot0-port.bin ]; then
  until grep -q "listening on" ~/abyss-stats/server.log; do
    kill -0 $pid 2>/dev/null || { echo "server died, see ~/abyss-stats/server.log"; exit 1; }
    sleep 3
  done
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=restore" -H "Content-Type: application/json" \
    -d '{"filename": "slot0-port.bin"}' || echo "slot restore failed"
  echo
fi
