#!/bin/bash
# Device memory of the production configuration: after loading, then the running peak of each GPU while a test server
# with the production settings (EXTRA_ENV="VAR=VALUE ..." adds settings, BUILD and MODEL another build or model,
# MOE_CACHE another --moe-cache-mib than the production 21100) runs the 96k prompt of ctx_size.py, a prompt that fills
# the context (FILL tokens, default the production 196608) and the append-only session of decode_modes.py (REPEAT
# corpus copies, ROUNDS turns per mode); nvidia-smi is sampled every 100 ms. The CUDA pool only grows, so the peak
# after a phase is that of everything up to it. Waits for the production server to go idle, saves its slot and stops
# it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/vram-peak-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
mkdir -p "$O"

pid=$(pgrep -x llama-server)
if [ -n "$pid" ]; then
  echo "### waiting for the production server to go idle $(date +%T)"
  until curl -s -m 5 http://127.0.0.1:8080/slots | grep -q '"is_processing":false'; do sleep 5; done
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=save" -H "Content-Type: application/json" \
    -d '{"filename": "slot0.bin"}'
  echo
  kill -TERM "$pid"
  for i in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  # its cleanup can hang past a minute; a second signal ends it at once, and the slot is saved already
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    for i in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  fi
  echo "### production stopped $(date +%T)"
fi
# until the stopped server is gone and the driver has handed back its device memory
for i in $(seq 90); do
  if ! pgrep -x llama-server > /dev/null &&
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }'; then
    break
  fi
  sleep 1
done

echo "### before loading: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{ printf "gpu%s %d MiB  ", $1, $2 }')"
log=$O/server.log
# shellcheck disable=SC2086
env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
  ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 \
  ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 \
  ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1 ABYSS_EXPERT_STATS="$O/stats.json" \
  ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1 \
  ABYSS_MOE_SKIP_ZERO=1 ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1 ABYSS_MMVQ_TAIL=1 \
  ABYSS_GRAPH_CHUNK=128 ABYSS_ENGRAM_KEEPALIVE_MS=4 ABYSS_HEAD_SPLIT_ONLY=1 \
  ABYSS_GEN_CKPT=256 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x ABYSS_CKPT_COUNT_START=1 ${EXTRA_ENV:-} \
  "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' -ot 'blk\.12\.ffn_(gate|up)_exps\.=CUDA0' \
  -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib "${MOE_CACHE:-21100}" --checkpoint-min-step 32768 \
  --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
spid=$!
up=1
until grep -q "listening on" "$log"; do
  kill -0 $spid 2>/dev/null || { echo "### server died while loading, see $log"; up=0; break; }
  sleep 2
done
if [ $up = 1 ]; then
  echo "### loaded: $(nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | awk -F', ' '{ printf "gpu%s %d MiB (free %d)  ", $1, $2, $3 }')"
  mem=$O/mem.csv
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits -lms 100 > "$mem" &
  smi=$!
  peak() {
    awk -F', ' -v what="$1" '{ if ($2 > m[$1]) { m[$1] = $2; f[$1] = $3 } }
      END { printf "### peak after %s:", what; for (g = 0; g in m; g++) printf "  gpu%d %d MiB (free %d)", g, m[g], f[g]; print "" }' "$mem"
  }
  CTX_URL=http://127.0.0.1:$PORT python3 -I "$E/ctx_size.py" "$E/corpus.txt" same "$O/same.json"
  peak "the 96k prompt"
  CTX_URL=http://127.0.0.1:$PORT python3 -I "$E/ctx_size.py" "$E/corpus.txt" fill "${FILL:-196608}" "$O/fill.json"
  peak "the context fill"
  DECODE_URL=http://127.0.0.1:$PORT python3 -I "$E/decode_modes.py" "$E/corpus.txt" --repeat "${REPEAT:-6}" \
    --rounds "${ROUNDS:-1}" --max-tokens 256 > "$O/session.txt" 2>&1 || echo "  session failed, see $O/session.txt"
  peak "the session"
  kill $smi
  kill -TERM $spid 2>/dev/null
  for i in $(seq 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
  kill -0 $spid 2>/dev/null && kill -TERM $spid
fi
wait $spid 2>/dev/null

if [ -n "$pid" ]; then
  for i in $(seq 90); do
    if ! pgrep -x llama-server > /dev/null &&
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }'; then
      break
    fi
    sleep 1
  done
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### vram-peak done $(date +%T), logs in $O"
