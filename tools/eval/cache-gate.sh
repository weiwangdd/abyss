#!/bin/bash
# The MoE cache's admission and eviction (mx-llama.cpp/0030: ABYSS_MOE_EVICT=cold evicts the coldest expert by decayed
# heat, ABYSS_MOE_ADMIT_GATE=1 admits a miss only if hotter than it, ABYSS_MOE_SKIP_ZERO=1 lets the CPU's experts of a
# token read no zero slot) against the production plan, and later switches the same way (0031: ABYSS_MOE_DEFER_UPLOAD=1
# defers a token's uploaded experts with the CPU's; 0032: ABYSS_HC_FUSE=1, ABYSS_GATHER_FUSE=1 fuse kernels), on servers
# with the production settings (now those of 0044, so VAR=0 turns a switch of them off), one per entry of CONFIGS
# ("name:VAR=VALUE,VAR=VALUE", the settings on top of those; an entry without them is production; BIN=<dir> among them
# runs that entry's server from another build, MODEL=<gguf> on another model, T=<n> with n CPU threads instead of 16,
# CACHE=<MiB> with another --moe-cache-mib than the production 21100):
# greedy_kl.py on PROMPTS corpus slices of 128 greedy tokens each and the 96k prompt of ctx_size.py with its 32 greedy
# tokens, compared with the first entry's (a cache that admits other experts gives the CPU other experts, whose sums
# round differently; the zero slots alone change nothing), and the append-only session of decode_modes.py (REPEAT
# corpus copies, ROUNDS turns per mode; SEED fixes the sampled turns' seeds, so entries whose logits are the same bit
# for bit hold the same conversation; DROP_CACHE=1 drops the model files from the page cache before each entry, so the
# Engram rows of that conversation are as cold in every entry), whose decode speed per turn and decode stats are summed
# up per entry. BUILD picks the binaries (default build-pf). Waits for the production server to go idle, saves its slot
# and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/cache-gate-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
CONFIGS=${CONFIGS:-"prod"}
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

first=""
for c in $CONFIGS; do
  IFS=: read -r name settings <<< "$c"
  extra_env=()
  [ -n "$settings" ] && IFS=, read -r -a extra_env <<< "$settings"
  bin=$B
  model=$M
  threads=16
  cache=21100
  for k in "${!extra_env[@]}"; do
    if [[ ${extra_env[$k]} == BIN=* ]]; then
      bin=${extra_env[$k]#BIN=}
      unset 'extra_env[k]'
    elif [[ ${extra_env[$k]} == MODEL=* ]]; then
      model=${extra_env[$k]#MODEL=}
      unset 'extra_env[k]'
    elif [[ ${extra_env[$k]} == T=* ]]; then
      threads=${extra_env[$k]#T=}
      unset 'extra_env[k]'
    elif [[ ${extra_env[$k]} == CACHE=* ]]; then
      cache=${extra_env[$k]#CACHE=}
      unset 'extra_env[k]'
    fi
  done
  log=$O/server-$name.log
  echo "### $name: ${settings:-production} $(date +%T)"
  if [ -n "${DROP_CACHE:-}" ]; then
    # every entry starts with none of the model's mapped pages (the Engram tables) in the page cache, so a seeded
    # session finds the rows of its text cold in each entry, not only in the first
    python3 -I -c 'import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)' "$(dirname "$model")"/*.gguf
  fi
  env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
    ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 \
    ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 \
    ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1 ABYSS_EXPERT_STATS="$O/stats-$name.json" \
    ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1 \
    ABYSS_MOE_SKIP_ZERO=1 ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1 ABYSS_MMVQ_TAIL=1 \
    ABYSS_GRAPH_CHUNK=128 ABYSS_ENGRAM_KEEPALIVE_MS=4 ABYSS_HEAD_SPLIT_ONLY=1 \
    ABYSS_GEN_CKPT=256 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x ABYSS_CKPT_COUNT_START=1 ABYSS_WEIGHT_NOWAIT=1 "${extra_env[@]}" \
    "$bin/llama-server" -m "$model" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t "$threads" \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' -ot 'blk\.12\.ffn_(gate|up)_exps\.=CUDA0' \
    -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib "$cache" --checkpoint-min-step 32768 \
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
      --rounds "${ROUNDS:-3}" --max-tokens 256 ${SEED:+--seed "$SEED"} > "$O/session-$name.txt" 2>&1 ||
      echo "  session failed, see $O/session-$name.txt"
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
      match($0, /[0-9.]+ by the CPU/); c += substr($0, RSTART, RLENGTH) + 0
      match($0, /[0-9.]+ major page faults/); f += substr($0, RSTART, RLENGTH) + 0 }
    END { if (n) printf "  decode stats: %d windows, %.1f ms/token, hit rate %.1f%%, per token %.1f uploaded, %.1f by the CPU, %.1f disk faults\n",
      n, t / n, h / n, u / n, c / n, f / n }' "$log"
  grep -a -E "abyss: moe cache ubatch <= 8|evicting the|deferred uploads" "$log" | sed 's/^.*abyss: /  /; s/^.*ABYSS_MOE_CPU_MISS/  ABYSS_MOE_CPU_MISS/'
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

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### all done $(date +%T), logs in $O"
