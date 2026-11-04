#!/bin/bash
# The MoE cache's admission and eviction (mx-llama.cpp/0030: ABYSS_MOE_EVICT=cold evicts the coldest expert by decayed
# heat, ABYSS_MOE_ADMIT_GATE=1 admits a miss only if hotter than it, ABYSS_MOE_SKIP_ZERO=1 lets the CPU's experts of a
# token read no zero slot) against the production plan, and later switches of the cache the same way (0031:
# ABYSS_MOE_DEFER_UPLOAD=1 defers a token's uploaded experts with the CPU's), on servers with the production settings
# of 0029, one per entry of CONFIGS ("name:VAR=VALUE,VAR=VALUE", the settings on top of those; an entry without them is
# that production):
# greedy_kl.py on PROMPTS corpus slices of 128 greedy tokens each and the 96k prompt of ctx_size.py with its 32 greedy
# tokens, compared with the first entry's (a cache that admits other experts gives the CPU other experts, whose sums
# round differently; the zero slots alone change nothing), and the append-only session of decode_modes.py (REPEAT
# corpus copies, ROUNDS turns per mode), whose decode speed per turn and decode stats are summed up per entry. BUILD
# picks the binaries (default build-pf). Waits for the production server to go idle, saves its slot and stops it, and
# restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/cache-gate-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
CONFIGS=${CONFIGS:-"base z:ABYSS_MOE_SKIP_ZERO=1 g995:ABYSS_MOE_SKIP_ZERO=1,ABYSS_MOE_EVICT=cold,ABYSS_MOE_ADMIT_GATE=1,ABYSS_MOE_HEAT_DECAY=0.995 g99:ABYSS_MOE_SKIP_ZERO=1,ABYSS_MOE_EVICT=cold,ABYSS_MOE_ADMIT_GATE=1,ABYSS_MOE_HEAT_DECAY=0.99"}
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
  log=$O/server-$name.log
  echo "### $name: ${settings:-production} $(date +%T)"
  env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
    ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 \
    ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 \
    ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1 ABYSS_EXPERT_STATS="$O/stats-$name.json" \
    ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 "${extra_env[@]}" \
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
