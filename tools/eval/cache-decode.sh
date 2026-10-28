#!/bin/bash
# Decode speed against the expert cache size: for each size (MiB) in CACHES, the append-only session of decode_modes.sh
# (REPEAT corpus copies, ROUNDS turns per mode) on a server with the production settings at CTX, the indexer prefill in
# chunks of CHUNK queries (ABYSS_DSV41_IDX_CHUNK, mx-llama.cpp/0019, which leaves room on GPU1 for a larger cache).
# Waits for the production server to go idle, saves its slot and stops it, and restarts it at the end
# (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
export MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
CACHES=${CACHES:-"12000 16000"}

pid=$(pgrep -x llama-server)
if [ -n "$pid" ]; then
  echo "### waiting for the production server to go idle $(date +%T)"
  until curl -s -m 5 http://127.0.0.1:8080/slots | grep -q '"is_processing":false'; do sleep 5; done
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=save" -H "Content-Type: application/json" \
    -d '{"filename": "slot0.bin"}'
  echo
  kill -TERM "$pid"
  for i in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  echo "### production stopped $(date +%T)"
fi

# in a subshell, so restart.sh below does not inherit the settings
(
  export GOMP_SPINCOUNT=100000000 ABYSS_DECODE_STATS=100 ABYSS_DSV41_IDX_CHUNK=${CHUNK:-256}
  export PORT=${PORT:-8090} REPEAT=${REPEAT:-6} ROUNDS=${ROUNDS:-3}
  for cache in $CACHES; do
    echo "### decode, expert cache $cache MiB $(date +%T)"
    EXTRA="-c ${CTX:-196608} --moe-cache-mib $cache --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0" \
      "$E/decode_modes.sh"
  done
)

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
fi
echo "### all done $(date +%T)"
