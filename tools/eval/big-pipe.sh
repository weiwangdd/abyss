#!/bin/bash
# Whether GPU0 could compute the uploaded experts of the decoder layers' large batches while it uploads them: an nsys
# profile of an agent turn's prefill with the production settings (agent_turns.py --capture: TURNS turns, default 6, of
# about 1-2k new tokens after a 48k first turn, the last one captured with one generated token), read with nsys_big.py
# (per layer: GPU0's upload and kernels, GPU1's kernels, the host's waits, the estimate) and nsys_breakdown.py.
# BUILD picks the binaries (default build-m2), EXTRA_ENV adds settings ("VAR=VALUE VAR=VALUE", after the production
# ones). Waits for the production server to go idle, saves its slot and stops it, and restarts it as soon as the capture
# is over (~/abyss-stats/restart.sh restores the slot); the export and the analysis run after that.
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/big-pipe-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
REP=$O/turn
mkdir -p "$O"

if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi

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
gpus_free() {  # until the stopped server is gone and the driver has handed back its device memory
  for i in $(seq 90); do
    if ! pgrep -x llama-server > /dev/null &&
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }'; then
      return
    fi
    sleep 1
  done
}
gpus_free

log=$O/server.log
echo "### server under nsys $(date +%T)"
env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
  ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 \
  ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 \
  ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1 ABYSS_EXPERT_STATS="$O/expert-stats.json" \
  ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1 \
  ABYSS_MOE_SKIP_ZERO=1 ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1 ABYSS_MMVQ_TAIL=1 \
  ABYSS_GRAPH_CHUNK=128 ABYSS_ENGRAM_KEEPALIVE_MS=4 ABYSS_HEAD_SPLIT_ONLY=1 ABYSS_GEN_CKPT=256 \
  ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x ABYSS_CKPT_COUNT_START=1 ABYSS_WEIGHT_NOWAIT=1 ${EXTRA_ENV:-} \
  nsys launch --session-new=abyss --trace=cuda,osrt \
  "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' -ot 'blk\.12\.ffn_(gate|up)_exps\.=CUDA0' \
  -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib 21100 --checkpoint-min-step 32768 \
  --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
up=0
for i in $(seq 120); do
  grep -q "listening on" "$log" && { up=1; break; }
  sleep 5
done
if [ $up = 1 ]; then
  echo "### agent turns $(date +%T), the last one captured"
  AGENT_URL=http://127.0.0.1:$PORT python3 -I "$E/agent_turns.py" "$E/corpus.txt" --turns "${TURNS:-6}" \
    --repeat "${AGENT_REPEAT:-3}" --max-tokens "${AGENT_TOKENS:-512}" --capture "$REP" 2>&1 | tee "$O/agent.txt" |
    sed 's/^/  /'
else
  echo "### the server did not come up in 10 min, see $log"
  tail -5 "$log"
fi
nsys shutdown --session=abyss --kill=sigterm
gpus_free
# the calls of the captured turn
grep -a "abyss: the prompt" "$log" | tail -2 | sed 's/^.*abyss: /  /'

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi

if [ -f "$REP.nsys-rep" ]; then
  nsys export --type sqlite --output "$REP.sqlite" --force-overwrite=true "$REP.nsys-rep" > /dev/null
  python3 -I "$E/nsys_breakdown.py" "$REP.sqlite" | head -12 | tee "$O/breakdown.txt"
  python3 -I "$E/nsys_big.py" "$REP.sqlite" | tee "$O/big.txt"
fi
echo "### big-pipe done $(date +%T), files in $O"
