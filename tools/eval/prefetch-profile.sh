#!/bin/bash
# Where the decode time goes with the expert prefetch (ABYSS_MOE_PREFETCH=K, mx-llama.cpp/0024) and without: for each
# entry "K[:max]" of KS (default "0 3"; max is ABYSS_MOE_PREFETCH_MAX, default K), an nsys profile of a decode turn
# with the production settings (decode_modes.sh CAPTURE=1, the last turn of an append-only session of REPEAT corpus
# copies), the per-token breakdown of nsys_decode.py, and the per-stream view of GPU1 of nsys_streams.py (the prefetch
# uploads on a stream of their own, and how much of them ran while GPU1 computed). BUILD picks the binaries (default
# build-pf, which has the switch). Waits for the production server to go idle, saves its slot and stops it, and
# restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/prefetch-profile-$(date +%m%d-%H%M)
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
  export PORT=8090 BUILD=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
  export MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
  export GOMP_SPINCOUNT=100000000 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1
  export ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_BIG_CACHE=1
  export EXTRA='-c 196608 --moe-cache-mib 20500 -ot blk\.12\.ffn_(gate|up)_exps\.=CUDA0 --checkpoint-min-step 32768
    --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0'
  for entry in ${KS:-0 3}; do
    IFS=: read -r k max <<< "$entry"
    max=${max:-$k}
    d=$O/k$k-m$max
    echo "### ABYSS_MOE_PREFETCH=$k ABYSS_MOE_PREFETCH_MAX=$max $(date +%T)"
    mkdir -p "$d"
    OUT=$d ABYSS_MOE_PREFETCH=$k ABYSS_MOE_PREFETCH_MAX=$max CAPTURE=1 REPEAT=${REPEAT:-6} ROUNDS=1 \
      MAX_TOKENS=${MAX_TOKENS:-256} "$E/decode_modes.sh" > "$d/decode.txt" 2>&1
    sed -n '/decode window/,$p' "$d/decode.txt" | head -30
    python3 -I "$E/nsys_streams.py" "$d/decode-modes.sqlite" "${MAX_TOKENS:-256}" 1
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
