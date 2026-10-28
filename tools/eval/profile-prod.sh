#!/bin/bash
# Where the time goes with the production settings, in one stop of the production server: an nsys profile of a decode
# turn (decode_modes.sh CAPTURE=1, the last turn of an append-only session of REPEAT corpus copies, per-token breakdown
# by nsys_decode.py) and of an agent turn's prefill (agent-turns.sh CAPTURE=1, the last of TURNS turns,
# nsys_breakdown.py); SKIP_DECODE=1 leaves the decode turn out. Waits for the production server to go idle, saves its
# slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/profile-$(date +%m%d-%H%M)
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

# in a subshell, so restart.sh below does not inherit the settings
(
  export OUT=$O PORT=8090
  export MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
  export GOMP_SPINCOUNT=100000000 ABYSS_DSV41_IDX_CHUNK=256 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1
  export ABYSS_DECODE_STATS=100
  export EXTRA='-c 196608 --moe-cache-mib 20500 -ot blk\.12\.ffn_(gate|up)_exps\.=CUDA0 --checkpoint-min-step 32768
    --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0'
  if [ "${SKIP_DECODE:-0}" != 1 ]; then
    echo "### decode turn $(date +%T)"
    CAPTURE=1 REPEAT=${REPEAT:-6} ROUNDS=1 MAX_TOKENS=${MAX_TOKENS:-256} "$E/decode_modes.sh"
    gpus_free
  fi
  echo "### agent turn prefill $(date +%T)"
  CAPTURE=1 TURNS=${TURNS:-6} REPEAT=${REPEAT:-6} MAX_TOKENS=512 "$E/agent-turns.sh"
)

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
fi
echo "### all done $(date +%T), reports in $O"
