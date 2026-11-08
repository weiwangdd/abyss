#!/bin/bash
# An nsys profile of a decode turn with the settings of ~/abyss-stats/restart.sh as of mx-llama.cpp/0041 (decode_modes.sh
# CAPTURE=1, the last turn of an append-only session of REPEAT corpus copies), read with nsys_layers.py (the cached
# layers: GPU1 busy, the host waiting, the host computing), nsys_layer.py (where both GPUs wait, and one layer's
# timeline), nsys_streams.py (GPU1's copies against its kernels), nsys_kernels.py (the kernels by their own time) and
# nsys_graphs.py (the graph launches). NSYS_GRAPH_TRACE=graph traces the graphs as a whole, which does not slow their
# launches the way node tracing does, and runs only nsys_graphs.py.
# BUILD picks the binaries (default build-m2), EXTRA_ENV adds settings ("VAR=VALUE VAR=VALUE", e.g. a switch of a test
# build), SEED fixes the sampled turns' seeds (decode_modes.py --seed), so two profiles decode the same tokens. Waits
# for the production server to go idle, saves its slot and stops it, and restarts it at the end (~/abyss-stats/
# restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/prod-profile-$(date +%m%d-%H%M)
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
  export OUT=$O PORT=8090 BUILD=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
  export MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
  export GOMP_SPINCOUNT=100000000 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1
  export ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_BIG_CACHE=1 ABYSS_MOE_DEFER=1 ABYSS_HC_KSPLIT=1
  export ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1 ABYSS_MOE_SKIP_ZERO=1
  export ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1 ABYSS_MMVQ_TAIL=1
  export ABYSS_GRAPH_CHUNK=128 ABYSS_ENGRAM_KEEPALIVE_MS=4 ABYSS_HEAD_SPLIT_ONLY=1
  export ABYSS_GEN_CKPT=256
  export ABYSS_EXPERT_STATS=$O/expert-stats.json
  for kv in ${EXTRA_ENV:-}; do
    export "$kv"
  done
  export EXTRA='-c 196608 --moe-cache-mib 21100 -ot blk\.12\.ffn_(gate|up)_exps\.=CUDA0 --checkpoint-min-step 32768
    --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0'
  echo "### decode turn $(date +%T)"
  CAPTURE=1 REPEAT=${REPEAT:-6} ROUNDS=1 MAX_TOKENS=${MAX_TOKENS:-256} "$E/decode_modes.sh" > "$O/decode.txt" 2>&1
  sed -n '/decode window/,$p' "$O/decode.txt" | head -24
)
gpus_free

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi

S=$O/decode-modes.sqlite
echo "### graph launches (nsys_graphs.py)"
python3 -I "$E/nsys_graphs.py" "$S" "${MAX_TOKENS:-256}"
# the per-kernel reads need the nodes of the graphs, which only node tracing records
if [ "${NSYS_GRAPH_TRACE:-node}" = node ]; then
  echo "### cached layers (nsys_layers.py)"
  python3 -I "$E/nsys_layers.py" "$S"
  echo "### where both GPUs wait (nsys_layer.py)"
  python3 -I "$E/nsys_layer.py" "$S" | head -60
  echo "### GPU1 copies against its kernels (nsys_streams.py)"
  python3 -I "$E/nsys_streams.py" "$S" "${MAX_TOKENS:-256}" 1
  echo "### kernels (nsys_kernels.py)"
  python3 -I "$E/nsys_kernels.py" "$S" 20
fi
echo "### all done $(date +%T), reports in $O"
