#!/bin/bash
# The tensor-parallel prototype (fork branch abyss-tp: ABYSS_TP_LAYERS=N runs layers [0, N) on a Meta device over
# both GPUs, ABYSS_TP_SPLIT=F is GPU0's share of each split weight) against the production settings on the same MoE
# cache: upstream-bench.sh with three entries, tp (N, F; -ts 0/1 puts the other layers on CUDA1, the experts of
# layers N and N+1 go to CUDA0 as layer 12's did, LLAMA_DSV41_HOST_INPUT_THREADS gathers the Engram rows on the host
# so no CPU split cuts the Meta device's layers, -fit off since the fit does not know a Meta device among a layer
# split), tpnb (the same with GGML_META_TOPK_BCAST=0: no host broadcast after the indexer's top-k, which keeps the
# Meta device from recording a whole token as one CUDA graph; both lanes compute it bit for bit alike on two equal
# GPUs) and base (production, the same cache). TESTS (default greedy same session), CACHE (default 14000), N (20),
# F (0.5), PP (the build, default build-pp of ~/src/mx-llama.cpp-pipe), ENTRIES (which of the three, default all),
# TP_EXTRA (settings added to the tensor-parallel entries, e.g. GGML_SCHED_DEBUG=1), CONFIGS to replace the entries. Runs
# upstream-bench.sh with RESTART=0 under a timeout (LIMIT seconds, default 2700), then stops a test server left on
# PORT and restarts production itself, whatever happened.
set -u
E=$(cd "$(dirname "$0")" && pwd)
PP=${PP:-$HOME/src/mx-llama.cpp-pipe/build-pp/bin}
CACHE=${CACHE:-14000}
N=${N:-20}
F=${F:-0.5}
PORT=${PORT:-8090}
OT_TP="blk\\.(2[2-9]|3[0-9])\\.ffn_(gate|up|down)_exps\\.=CPU;blk\\.($N|$((N + 1)))\\.ffn_(gate|up|down)_exps\\.=CUDA0"
TP="BIN=$PP,CACHE=$CACHE,TS=0/1,OT=$OT_TP,ARGS=-fit;off,ABYSS_TP_LAYERS=$N,ABYSS_TP_SPLIT=$F"
TP="$TP,LLAMA_DSV41_HOST_INPUT_THREADS=4,GGML_META_PROFILE=1${TP_EXTRA:+,$TP_EXTRA}"
if [ -z "${CONFIGS:-}" ]; then
  CONFIGS=""
  for e in ${ENTRIES:-tp tpnb base}; do
    case $e in
      tp) CONFIGS="$CONFIGS tp:fork:$TP" ;;
      tpnb) CONFIGS="$CONFIGS tpnb:fork:$TP,GGML_META_TOPK_BCAST=0" ;;
      base) CONFIGS="$CONFIGS base:fork:BIN=$PP,CACHE=$CACHE" ;;
    esac
  done
fi
export CONFIGS
export TESTS=${TESTS:-"greedy same session"}
export FORK_ARGS=${FORK_ARGS:-"-lv 4"}
export DROP_CACHE=${DROP_CACHE:-1} SEED=${SEED:-1} PORT RESTART=0 STOP_ON_FAIL=1

echo "### tp-proto $(date +%T): N=$N F=$F cache $CACHE"
timeout -k 30 "${LIMIT:-2700}" "$E/upstream-bench.sh"
echo "### upstream-bench.sh exit $? $(date +%T)"

# a test server still up (the timeout hit) holds the GPUs, and upstream-bench.sh's memory sampler would run on
pkill -f "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500" 2>/dev/null
for p in $(pgrep -x llama-server); do
  if tr '\0' ' ' < "/proc/$p/cmdline" | grep -q -- "--port $PORT"; then
    echo "### stopping the test server $p"
    kill -TERM "$p"
    for i in $(seq 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -0 "$p" 2>/dev/null && kill -KILL "$p"
  fi
done
# the driver hands a stopped server's memory back a few seconds later (a server still running is production, never stopped)
for i in $(seq 90); do
  pgrep -x llama-server > /dev/null && break
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }' && break
  sleep 1
done
if ! curl -s -m 5 http://127.0.0.1:8080/health | grep -q ok; then
  echo "### restarting production $(date +%T)"
  # production gets none of the test settings exported above
  env -u CONFIGS -u TESTS -u FORK_ARGS -u DROP_CACHE -u SEED -u PORT -u RESTART -u STOP_ON_FAIL ~/abyss-stats/restart.sh 2>&1 | tail -3
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
fi
curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
  echo "### WARNING: production does not answer on 8080"
echo "### tp-proto done $(date +%T)"
