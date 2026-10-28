#!/bin/bash
# Standard benchmarks, so Abyss's numbers read like other systems': llama-bench's pp and tg tok/s at a few context depths
# (DEPTHS; PROMPTS and GEN tokens, REPS repetitions), and AIPerf (NVIDIA's load generator for OpenAI-style servers, in
# ~/abyss-tools/aiperf-venv) against a server, one request at a time, for each input:output token count in SHAPES
# (REQUESTS requests each; AIPerf's coding corpus, every prefill whole, ignore_eos holds the output length): time to first
# token, time per output token and tokens/s per user (std_bench.py). Both run with the production settings
# (tools/restart-server.sh); llama-bench gets the expert cache through LLAMA_ARG_MOE_CACHE_MIB (mx-llama.cpp/0021). Its
# pp is one call per test, so it shows the per-call cost of reading the host experts and not the long prompts CED speeds
# up, and its tg decodes random tokens; AIPerf's long inputs and decode_modes.sh cover those. SKIP_BENCH=1 or
# SKIP_AIPERF=1 leaves a part out. Waits for the production server to go idle, saves its slot and stops it, and restarts
# it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/std-bench-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
TOK=${TOKENIZER:-/data/models/DeepSeek-V4.1-Flash-1017}
AIPERF=${AIPERF:-$HOME/abyss-tools/aiperf-venv/bin/aiperf}
PORT=${PORT:-8090}
CACHE=${MOE_CACHE:-16000}
SHAPES=${SHAPES:-"1024:1024 8192:1024 32768:1024 98304:1024"}
mkdir -p "$O"

# the production settings, passed with env so that none of them reaches restart.sh
ENVS=(GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8
  ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=256 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1
  ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 ABYSS_DECODE_STATS=100)
COMMON=(-m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 -b 65536 -ub 2048
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU')

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
# the driver hands back a stopped process's device memory a few seconds after the process is gone
for i in $(seq 60); do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }' && break
  sleep 1
done
if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi

if [ "${SKIP_BENCH:-0}" != 1 ]; then
  echo "### llama-bench, depths ${DEPTHS:-0,32768,98304} $(date +%T)"
  env "${ENVS[@]}" LLAMA_ARG_MOE_CACHE_MIB="$CACHE" "$B/llama-bench" "${COMMON[@]}" -p "${PROMPTS:-512,2048}" \
    -n "${GEN:-128}" -d "${DEPTHS:-0,32768,98304}" -r "${REPS:-3}" -o md > "$O/llama-bench.md" 2> "$O/llama-bench.log" ||
    echo "  llama-bench failed, see $O/llama-bench.log"
  cat "$O/llama-bench.md"
  grep -a "abyss: moe cache" "$O/llama-bench.log" | sed 's/^/  /'
fi

if [ "${SKIP_AIPERF:-0}" != 1 ]; then
  log=$O/server.log
  env "${ENVS[@]}" "$B/llama-server" "${COMMON[@]}" -c "${CTX:-196608}" -np 1 --moe-cache-mib "$CACHE" \
    --checkpoint-min-step 32768 --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
  spid=$!
  until grep -q "listening on" "$log"; do
    kill -0 $spid 2>/dev/null || { echo "server died, see $log"; break; }
    sleep 3
  done
  warm=(--warmup-request-count 1)  # the first request after a start carries a one-time cost
  for shape in $SHAPES; do
    IFS=: read -r isl osl <<< "$shape"
    echo "### AIPerf, $isl input and $osl output tokens, ${REQUESTS:-3} requests $(date +%T)"
    # AIPerf's prompts share leading text, which the server would take from its prompt cache: cache_prompt false keeps
    # every prefill whole, as in the other systems' published runs
    "$AIPERF" profile --model deepseek-v4.1-flash --url "http://127.0.0.1:$PORT" --endpoint-type completions --streaming \
      --tokenizer "$TOK" --prompt-corpus coding --isl "$isl" --osl "$osl" --extra-inputs ignore_eos:true cache_prompt:false \
      --use-legacy-max-tokens --use-server-token-count --concurrency 1 --request-count "${REQUESTS:-3}" "${warm[@]}" \
      --random-seed 1 --no-gpu-telemetry --no-server-metrics --output-artifact-dir "$O/aiperf-$isl-$osl" \
      > "$O/aiperf-$isl-$osl.log" 2>&1 || echo "  AIPerf failed, see $O/aiperf-$isl-$osl.log"
    warm=()
  done
  kill -TERM $spid 2>/dev/null
  wait $spid 2>/dev/null
  python3 -I "$E/std_bench.py" "$O"
  grep -a "abyss: moe cache" "$log" | sed 's/^/  /'
fi

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
fi
echo "### all done $(date +%T), results in $O"
