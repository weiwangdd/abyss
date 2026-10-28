#!/bin/bash
# The prefill's large batches computing the experts the expert cache holds from it on GPU1 (ABYSS_MOE_BIG_CACHE=1,
# mx-llama.cpp/0023) against without, on a server with the production settings: a short prompt generating 512 tokens
# greedily fills the cache the same way in both (it only decodes), then the 96k prompt of ctx_size.py, whose outputs are
# compared; then an agent session of TURNS turns (agent_turns.py), whose prefill per turn is compared, with each large
# batch's experts and how many the cache held (ABYSS_MOE_BIG_STATS=1, big_coverage.py); and, unless MEM=0, the memory
# with the switch on (ctx-size.sh at 192K). SKIP_AB=1 runs only the memory. Waits for the production server to go idle,
# saves its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/big-cache-$(date +%m%d-%H%M)
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

variants="0 1"
[ "${SKIP_AB:-0}" = 1 ] && variants=""
for v in $variants; do
  log=$O/server-cache$v.log
  echo "### ABYSS_MOE_BIG_CACHE=$v $(date +%T)"
  env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
    ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 \
    ABYSS_MOE_ADMIT=0.3 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 \
    ABYSS_MOE_BIG_STATS=1 ABYSS_MOE_BIG_CACHE=$v \
    "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' -ot 'blk\.12\.ffn_(gate|up)_exps\.=CUDA0' \
    -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib 20500 --checkpoint-min-step 32768 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
    --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
  spid=$!
  until grep -q "listening on" "$log"; do
    kill -0 $spid 2>/dev/null || { echo "server died, see $log"; break; }
    sleep 3
  done
  # only decoding fills the cache, so both runs hold the same experts before the long prompt
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "写一首关于大海的长诗。"}], "max_tokens": 512, "temperature": 0}' > /dev/null
  CTX_URL=http://127.0.0.1:$PORT python3 -I "$E/ctx_size.py" "$E/corpus.txt" same "$O/same-cache$v.json"
  echo "  agent session"
  AGENT_URL=http://127.0.0.1:$PORT python3 -I "$E/agent_turns.py" "$E/corpus.txt" --turns "${TURNS:-10}" --repeat 6 \
    --max-tokens 512 | tee "$O/agent-cache$v.txt"
  kill -TERM $spid 2>/dev/null
  for i in $(seq 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
  kill -0 $spid 2>/dev/null && kill -TERM $spid
  wait $spid 2>/dev/null
  grep -a "abyss: moe cache" "$log" | sed 's/^/  /'
  gpus_free
done

if [ -n "$variants" ]; then
  echo "### 96k prompt, without vs with: $(python3 -I "$E/ctx_size.py" "$E/corpus.txt" compare "$O/same-cache0.json" "$O/same-cache1.json")"
  for v in 0 1; do
    echo "### large batches, ABYSS_MOE_BIG_CACHE=$v"
    python3 -I "$E/big_coverage.py" "$O/server-cache$v.log" | tail -12
  done
fi

# the memory with the cache path: ctx-size.sh at 192K (the reserve holds the path's tensors whether or not the cache is
# warm), in a subshell so that restart.sh does not inherit the switch
if [ "${MEM:-1}" = 1 ]; then
  (
    export MODEL=$M PORT GOMP_SPINCOUNT=100000000 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_MOE_ADMIT=0.3 ABYSS_CKPT_THIN_STEP=8192
    export ABYSS_PLAIN_INPUT=1 ABYSS_MOE_BIG_CACHE=1
    mkdir -p "$O/mem"
    OUT=$O/mem EXTRA='-ot blk\.12\.ffn_(gate|up)_exps\.=CUDA0' CONFIGS="c512cache:196608:20500:2048" FILLCMP="" \
      "$E/ctx-size.sh"
  )
  gpus_free
fi

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### all done $(date +%T), logs in $O"
