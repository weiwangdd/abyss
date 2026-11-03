#!/bin/bash
# The hyper-connection mixes as one split-K product with their norm folded in (ABYSS_HC_KSPLIT=1, mx-llama.cpp/0027)
# against without: test-backend-ops on the fused pair (CUDA0 against the CPU, then the time per graph), then servers
# with the production settings and the expert stats (through cb_moe_ids, 0026), one per entry of CONFIGS
# ("name:ksplit"): greedy_kl.py on PROMPTS corpus slices of 128 greedy tokens each and the 96k prompt of ctx_size.py
# with its 32 greedy tokens, compared with the first entry's (top-20 KL over the identical prefix), and the
# append-only session of decode_modes.py (REPEAT corpus copies, ROUNDS turns per mode), whose decode speed per turn is
# summed up per entry. PROFILE=1 adds an nsys profile of a decode turn with ABYSS_HC_KSPLIT=1 (decode_modes.sh
# CAPTURE=1) and its kernels (nsys_kernels.py) and layers (nsys_layers.py). BUILD picks the binaries (default
# build-pf). Waits for the production server to go idle, saves its slot and stops it, and restarts it at the end
# (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/hc-ksplit-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
CONFIGS=${CONFIGS:-"base:0 ksplit:1"}
mkdir -p "$O"

if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi

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

echo "### test-backend-ops RMS_NORM_MUL_MAT $(date +%T)"
for k in 0 1; do
  echo "  ABYSS_HC_KSPLIT=$k:"
  ABYSS_HC_KSPLIT=$k "$B/test-backend-ops" test -o RMS_NORM_MUL_MAT -b CUDA0 2>&1 | grep -E "RMS_NORM_MUL_MAT|tests passed" | sed 's/^/    /'
  ABYSS_HC_KSPLIT=$k "$B/test-backend-ops" perf -o RMS_NORM_MUL_MAT -b CUDA0 2>&1 | grep -E "RMS_NORM_MUL_MAT" | sed 's/^/    /'
done

first=""
for c in $CONFIGS; do
  IFS=: read -r name ksplit <<< "$c"
  log=$O/server-$name.log
  echo "### $name: ABYSS_HC_KSPLIT=$ksplit $(date +%T)"
  env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
    ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 \
    ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 \
    ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1 ABYSS_EXPERT_STATS="$O/stats-$name.json" \
    ABYSS_HC_KSPLIT="$ksplit" \
    "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' -ot 'blk\.12\.ffn_(gate|up)_exps\.=CUDA0' \
    -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib 20500 --checkpoint-min-step 32768 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
  spid=$!
  up=1
  until grep -q "listening on" "$log"; do
    kill -0 $spid 2>/dev/null || { echo "  server died, see $log"; up=0; break; }
    sleep 3
  done
  if [ $up = 1 ]; then
    GREEDY_URL=http://127.0.0.1:$PORT python3 -I "$E/greedy_kl.py" run "$E/corpus.txt" "$O/greedy-$name.json" \
      --prompts "${PROMPTS:-8}" | tail -2
    CTX_URL=http://127.0.0.1:$PORT python3 -I "$E/ctx_size.py" "$E/corpus.txt" same "$O/same-$name.json"
    DECODE_URL=http://127.0.0.1:$PORT python3 -I "$E/decode_modes.py" "$E/corpus.txt" --repeat "${REPEAT:-6}" \
      --rounds "${ROUNDS:-3}" --max-tokens 256 > "$O/session-$name.txt" 2>&1 || echo "  session failed, see $O/session-$name.txt"
    kill -TERM $spid 2>/dev/null
    for i in $(seq 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
    kill -0 $spid 2>/dev/null && kill -TERM $spid
  fi
  wait $spid 2>/dev/null
  echo "  exit $? $(date +%T)"
  # the session's turns are the requests of 256 tokens; the first of the ctx_size.py request's 32 is left out
  awk '/       eval time/ && / 256 tokens/ { match($0, /[0-9.]+ tokens per second/); s += substr($0, RSTART, RLENGTH) + 0; n++ }
    END { if (n) printf "  session: %d turns, mean %.2f tok/s\n", n, s / n }' "$log"
  awk '/abyss: decode stats/ { n++; match($0, /[0-9.]+ ms\/token/); t += substr($0, RSTART, RLENGTH) + 0
      match($0, /hit rate [0-9.]+%/); h += substr($0, RSTART + 9, RLENGTH - 10) + 0
      match($0, /[0-9.]+ uploaded/); u += substr($0, RSTART, RLENGTH) + 0
      match($0, /[0-9.]+ by the CPU/); c += substr($0, RSTART, RLENGTH) + 0 }
    END { if (n) printf "  decode stats: %d windows, %.1f ms/token, hit rate %.1f%%, per token %.1f uploaded, %.1f by the CPU\n",
      n, t / n, h / n, u / n, c / n }' "$log"
  if [ -n "$first" ] && [ -s "$O/greedy-$first.json" ] && [ -s "$O/greedy-$name.json" ]; then
    echo "  corpus slices, $first vs $name:"
    python3 -I "$E/greedy_kl.py" compare "$O/greedy-$first.json" "$O/greedy-$name.json" | sed 's/^/    /'
  fi
  if [ -n "$first" ] && [ -s "$O/same-$first.json" ] && [ -s "$O/same-$name.json" ]; then
    echo "  96k prompt, $first vs $name: $(python3 -I "$E/ctx_size.py" "$E/corpus.txt" compare "$O/same-$first.json" "$O/same-$name.json")"
  fi
  [ -z "$first" ] && first=$name
  gpus_free
done

if [ "${PROFILE:-0}" = 1 ]; then
  # in a subshell, so restart.sh below does not inherit the settings
  (
    d=$O/profile
    mkdir -p "$d"
    echo "### nsys profile of a decode turn, ABYSS_HC_KSPLIT=1 $(date +%T)"
    export PORT BUILD=$B MODEL=$M
    export GOMP_SPINCOUNT=100000000 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1
    export ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_BIG_CACHE=1 ABYSS_MOE_DEFER=1 ABYSS_HC_KSPLIT=1
    export ABYSS_EXPERT_STATS=$d/expert-stats.json
    export EXTRA='-c 196608 --moe-cache-mib 20500 -ot blk\.12\.ffn_(gate|up)_exps\.=CUDA0 --checkpoint-min-step 32768
      --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0'
    OUT="$d" CAPTURE=1 REPEAT=${REPEAT:-6} ROUNDS=1 MAX_TOKENS=256 "$E/decode_modes.sh" > "$d/decode.txt" 2>&1
    sed -n '/decode window/,$p' "$d/decode.txt" | head -24
    python3 -I "$E/nsys_layers.py" "$d/decode-modes.sqlite"
    python3 -I "$E/nsys_kernels.py" "$d/decode-modes.sqlite" 30
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
