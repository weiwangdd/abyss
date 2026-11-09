# Starting and stopping the servers the standard benchmarks run against; sourced by benchmark/run.sh, runs on blackwell.
#
# bench_stop_production    waits for the production server (8080) to go idle, saves its slot, stops it and waits until
#                          the driver has handed back the GPUs' memory; PROD_WAS_RUNNING=1 if it was running
# bench_start KIND NAME DIR  starts a server of KIND on $PORT (default 8090), its log in DIR/server-NAME.log, both GPUs'
#                          memory sampled every 0.5 s into DIR/mem-NAME.csv; returns once it listens (1 if it died)
#                            fork  production: build-m2, the production GGUF and settings (restart.sh, 0041)
#                            up    upstream llama.cpp (~/src/llama.cpp-upstream, e95426f16) on the same weights in its
#                                  layout (tools/convert/to_upstream.py), the same command line, the MoE cache at
#                                  UP_CACHE (16000) MiB with --moe-cache-cpu-share UP_SHARE (0.7), GGML_CUDA_MMQ_PREC=q8,
#                                  the template with the V4.1 tag literal upstream detects
#                            tiny  stories260K on the CPU, no GPU: for trying a benchmark's harness end to end
# bench_stop_server        stops it, prints the load time and both GPUs' memory peak
# bench_restart_production restarts production (~/abyss-stats/restart.sh restores the slot) and checks 8080
# See tools/eval/upstream-bench.sh for how the two configurations were chosen (CLAUDE.md, 上游实测对标).

PORT=${PORT:-8090}
FORK_BIN=${FORK_BIN:-$HOME/src/mx-llama.cpp/build-m2/bin}
UP_BIN=${UP_BIN:-$HOME/src/llama.cpp-upstream/build/bin}
FORK_MODEL=${FORK_MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
UP_MODEL=${UP_MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF-upstream/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
TINY_MODEL=${TINY_MODEL:-$HOME/abyss-eval/tmpl-test/stories260K.gguf}
TMPL=${TMPL:-$HOME/abyss-stats/tmpl/v41.jinja}
PROD_WAS_RUNNING=0

bench_gpus_free() {  # until no server is left and the driver has handed back the device memory
  for i in $(seq 90); do
    if ! pgrep -x llama-server > /dev/null &&
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }'; then
      return 0
    fi
    sleep 1
  done
  return 1
}

bench_stop_production() {
  local pid
  pid=$(pgrep -x llama-server)
  [ -z "$pid" ] && return 0
  PROD_WAS_RUNNING=1
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
  bench_gpus_free
}

bench_start() {
  local kind=$1 name=$2 dir=$3 env_list args
  SERVER_KIND=$kind
  SERVER_LOG=$dir/server-$name.log
  SERVER_MEM=$dir/mem-$name.csv
  args=(-c 196608 -b 65536 -ub 2048 -np 1 --checkpoint-min-step 32768 --jinja --no-reasoning-preserve
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --host 127.0.0.1 --port "$PORT")
  local gpu=(-ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU,blk\.12\.ffn_(gate|up)_exps\.=CUDA0')
  case $kind in
    fork)
      env_list=(GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1
        ABYSS_ENGRAM_PREFETCH=8 ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1
        ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1
        ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1
        ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1
        ABYSS_MOE_SKIP_ZERO=1 ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1
        ABYSS_MMVQ_TAIL=1 ABYSS_GRAPH_CHUNK=128 ABYSS_ENGRAM_KEEPALIVE_MS=4 ABYSS_HEAD_SPLIT_ONLY=1
        ABYSS_GEN_CKPT=256 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x ABYSS_CKPT_COUNT_START=1)
      args=("$FORK_BIN/llama-server" -m "$FORK_MODEL" "${gpu[@]}" --moe-cache-mib 21100 --chat-template-file "$TMPL"
        "${args[@]}")
      ;;
    up)
      # upstream takes the V4.1 tag names and message rendering only from a template holding "｜DSML｜ calls>" literally
      { printf '{#- ｜DSML｜ calls> -#}'; cat "$TMPL"; } > "$dir/v41-upstream.jinja"
      env_list=(GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000)
      args=("$UP_BIN/llama-server" -m "$UP_MODEL" "${gpu[@]}" --moe-cache-mib "${UP_CACHE:-16000}"
        --moe-cache-cpu-share "${UP_SHARE:-0.7}" --chat-template-file "$dir/v41-upstream.jinja" "${args[@]}")
      ;;
    tiny)
      env_list=(CUDA_VISIBLE_DEVICES=)
      # its vocabulary cuts the benchmarks' text into many more tokens than DeepSeek's, so the context is larger, and
      # the server caps a slot at the model's training context unless the model says it is that long
      args=("$FORK_BIN/llama-server" -m "$TINY_MODEL" -ngl 0 -t 4 -c 1048576 -np 1
        --override-kv llama.context_length=int:1048576 --jinja --chat-template-file "$TMPL" --host 127.0.0.1 --port "$PORT")
      ;;
    *) echo "unknown server kind $kind" >&2; return 2 ;;
  esac
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500 > "$SERVER_MEM" &
  MEM_PID=$!
  SERVER_T0=$SECONDS
  env "${env_list[@]}" "${args[@]}" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  until grep -q "listening on" "$SERVER_LOG"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  $name: server died, see $SERVER_LOG"
      kill "$MEM_PID" 2>/dev/null
      return 1
    fi
    sleep 1
  done
  SERVER_LOAD_S=$((SECONDS - SERVER_T0))
  echo "  $name ($kind): listening on $PORT after ${SERVER_LOAD_S} s, memory used (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
}

bench_stop_server() {
  kill -TERM "$SERVER_PID" 2>/dev/null
  for i in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -0 "$SERVER_PID" 2>/dev/null && kill -TERM "$SERVER_PID"
  wait "$SERVER_PID" 2>/dev/null
  kill "$MEM_PID" 2>/dev/null
  wait "$MEM_PID" 2>/dev/null
  # the tiny server runs beside production, which keeps the GPUs
  [ "$SERVER_KIND" = tiny ] && return 0
  awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { printf "  memory peak (MiB): GPU0 %d, GPU1 %d of 97887\n", m[0], m[1] }' "$SERVER_MEM"
  bench_gpus_free
}

bench_restart_production() {
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
  if curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0'; then
    echo "### production answers on 8080"
  else
    echo "### WARNING: production does not answer on 8080"
  fi
}
