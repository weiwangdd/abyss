#!/bin/bash
# How much of the prefill's large batches (over 32 tokens, layers 22-39) the expert cache already holds: agent-turns.sh
# with the production settings and ABYSS_MOE_BIG_STATS=1, which prints each layer batch's experts and the cached ones,
# summarized per turn by big_coverage.py. TURNS, REPEAT and MAX_TOKENS go to agent-turns.sh. Waits for the production
# server to go idle, saves its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/big-coverage-$(date +%m%d-%H%M)
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
# the driver hands back a stopped server's device memory a few seconds after the process is gone
for i in $(seq 60); do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }' && break
  sleep 1
done

# in a subshell, so restart.sh below does not inherit the settings
(
  export OUT=$O PORT=8090 TURNS=${TURNS:-10} REPEAT=${REPEAT:-6} MAX_TOKENS=${MAX_TOKENS:-512}
  export MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
  export GOMP_SPINCOUNT=100000000 ABYSS_DSV41_IDX_CHUNK=256 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1
  export ABYSS_MOE_BIG_STATS=1
  export EXTRA='-c 196608 --moe-cache-mib 20500 -ot blk\.12\.ffn_(gate|up)_exps\.=CUDA0 --checkpoint-min-step 32768
    --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0'
  echo "### agent session, $TURNS turns $(date +%T)"
  "$E/agent-turns.sh"
)
log=$(ls -t "$O"/server-agent-*.log | head -1)
python3 -I "$E/big_coverage.py" "$log"

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
fi
echo "### all done $(date +%T), server log $log"
