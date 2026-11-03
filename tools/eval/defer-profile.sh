#!/bin/bash
# Where the decode time goes with the MoE expert deferral (ABYSS_MOE_DEFER=1, mx-llama.cpp/0025) and without, with the
# production settings: for each entry "defer:stats" of KS (default "0:1 1:1"; stats 1 turns the expert stats of
# production on, ABYSS_EXPERT_STATS, written to a scratch file), an nsys profile of a decode turn (decode_modes.sh
# CAPTURE=1, the last turn of an append-only session of REPEAT corpus copies), the per-token breakdown of
# nsys_decode.py and the per-layer view of nsys_layers.py (who waits for whom in the cached layers). BUILD picks the
# binaries (default build-m2, the production build). Waits for the production server to go idle, saves its slot and
# stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/defer-profile-$(date +%m%d-%H%M)
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
  export PORT=8090 BUILD=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
  export MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
  export GOMP_SPINCOUNT=100000000 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1
  export ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_BIG_CACHE=1
  export EXTRA='-c 196608 --moe-cache-mib 20500 -ot blk\.12\.ffn_(gate|up)_exps\.=CUDA0 --checkpoint-min-step 32768
    --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0'
  for entry in ${KS:-0:1 1:1}; do
    IFS=: read -r defer stats <<< "$entry"
    d=$O/defer$defer-stats$stats
    echo "### ABYSS_MOE_DEFER=$defer, expert stats $stats $(date +%T)"
    mkdir -p "$d"
    stats_env=()
    [ "$stats" = 1 ] && stats_env=(ABYSS_EXPERT_STATS="$d/expert-stats.json")
    env "${stats_env[@]}" OUT="$d" ABYSS_MOE_DEFER="$defer" CAPTURE=1 REPEAT=${REPEAT:-6} ROUNDS=1 \
      MAX_TOKENS=${MAX_TOKENS:-256} "$E/decode_modes.sh" > "$d/decode.txt" 2>&1
    sed -n '/decode window/,$p' "$d/decode.txt" | head -24
    python3 -I "$E/nsys_layers.py" "$d/decode-modes.sqlite"
    gpus_free
  done
)

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### all done $(date +%T), reports in $O"
