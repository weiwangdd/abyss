#!/bin/bash
# Upstream llama.cpp (ggml-org, ~/src/llama.cpp-upstream/build, e95426f16) against production, each entry of CONFIGS
# ("name:fork" or "name:up", then optionally ":VAR=VALUE,VAR=VALUE") on a fresh server with the same command line:
# fork is production (build-m2, the production GGUF and settings), up is upstream on the same weights laid out for it
# (tools/convert/to_upstream.py), with the experts of layers 22-39 in host memory and those of layer 12 split as in
# production, GGML_CUDA_MMQ_PREC=q8 (upstream's default for MXFP4 on Blackwell is 4-bit activations, the official
# reference is W4A8) and the template with the V4.1 tag literal upstream detects. Settings: CACHE=<MiB>
# (--moe-cache-mib, default 21100 for fork, UP_CACHE for up), SHARE=<f> (--moe-cache-cpu-share, up only, default
# UP_SHARE), T=<n> CPU threads, BIN=<dir>, MODEL=<gguf>, TS=<split> (-ts, default 12/29), OT=<overrides> (-ot, default
# production's two, ';' standing for ','), ARGS=<arguments> (added to that entry's command line, ';' standing for ' ');
# other VAR=VALUE go to the environment. TESTS, in order:
# template (the prompt and tokens of fixed requests, upstream_template.py), greedy (greedy_kl.py on PROMPTS corpus
# slices), same (the 96k prompt of ctx_size.py, its prefill and 32 greedy tokens), session (decode_modes.py: REPEAT
# corpus copies, ROUNDS turns per mode, SEED), agent (agent_turns.py: AGENT_TURNS turns of AGENT_TOKENS tokens on
# AGENT_REPEAT copies), fill (ctx_size.py fill FILL, default 196608), split (ckpt_split.py: two conversations with many
# messages from scratch, edited at about 60% and one more turn, with the llama_decode() calls and checkpoints the
# server log shows for each, from mx-llama.cpp/0045 on), short (one greedy completion of 8 tokens on a short prompt,
# to log a few decode steps). Each entry prints its load time, both GPUs' memory
# after loading and at the peak (sampled every 0.5 s), what its log says about buffers and the MoE cache, and the
# tests' results; greedy, same, template and split are compared with the first entry. FORK_ARGS and UP_ARGS add arguments to
# the servers of that kind (UP_ARGS="-lv 4" makes upstream log its MoE cache statistics at exit, and a few debug lines
# per token). DROP_CACHE=1 drops the model files from
# the page cache before each entry, STOP_ON_FAIL=1 skips the entries after one whose server does not come up. Waits for the production server to go idle, saves its slot and stops it, and
# restarts it at the end (RESTART=0 never, RESTART=1 also when it was not running; ~/abyss-stats/restart.sh restores
# the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/upstream-$(date +%m%d-%H%M)
FORK_BIN=${FORK_BIN:-$HOME/src/mx-llama.cpp/build-m2/bin}
UP_BIN=${UP_BIN:-$HOME/src/llama.cpp-upstream/build/bin}
FORK_MODEL=${FORK_MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
UP_MODEL=${UP_MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF-upstream/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
CONFIGS=${CONFIGS:-"prod:fork up:up"}
TESTS=${TESTS:-"template greedy same session agent"}
mkdir -p "$O"
# upstream takes the V4.1 tag names (leading space) and message rendering only from a template holding "｜DSML｜ calls>"
# literally; the production template builds its tags from dsml_token, so the copy gets the literal in a comment
{ printf '{#- ｜DSML｜ calls> -#}'; cat "$HOME/abyss-stats/tmpl/v41.jinja"; } > "$O/v41-upstream.jinja"

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
  IFS=: read -r name kind settings <<< "$c"
  extra_env=()
  [ -n "$settings" ] && IFS=, read -r -a extra_env <<< "$settings"
  threads=16
  share=${UP_SHARE:-0}
  ts=12/29
  entry_args=()
  ot='blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU,blk\.12\.ffn_(gate|up)_exps\.=CUDA0'
  if [ "$kind" = fork ]; then
    bin=$FORK_BIN model=$FORK_MODEL cache=21100 tmpl=$HOME/abyss-stats/tmpl/v41.jinja
    base_env=(GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1
      ABYSS_ENGRAM_PREFETCH=8 ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1
      ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1
      ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1
      ABYSS_EXPERT_STATS="$O/stats-$name.json" ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1
      ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1 ABYSS_MOE_SKIP_ZERO=1 ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1
      ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1 ABYSS_MMVQ_TAIL=1 ABYSS_GRAPH_CHUNK=128 ABYSS_ENGRAM_KEEPALIVE_MS=4
      ABYSS_HEAD_SPLIT_ONLY=1 ABYSS_GEN_CKPT=256 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x ABYSS_CKPT_COUNT_START=1 ABYSS_WEIGHT_NOWAIT=1)
    kind_args=(${FORK_ARGS:-})
  else
    bin=$UP_BIN model=$UP_MODEL cache=${UP_CACHE:-16000} tmpl=$O/v41-upstream.jinja
    base_env=(GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000)
    kind_args=(${UP_ARGS:-} --moe-cache-cpu-share)
  fi
  for k in "${!extra_env[@]}"; do
    case ${extra_env[$k]} in
      BIN=*) bin=${extra_env[$k]#BIN=}; unset 'extra_env[k]' ;;
      MODEL=*) model=${extra_env[$k]#MODEL=}; unset 'extra_env[k]' ;;
      T=*) threads=${extra_env[$k]#T=}; unset 'extra_env[k]' ;;
      CACHE=*) cache=${extra_env[$k]#CACHE=}; unset 'extra_env[k]' ;;
      SHARE=*) share=${extra_env[$k]#SHARE=}; unset 'extra_env[k]' ;;
      TS=*) ts=${extra_env[$k]#TS=}; unset 'extra_env[k]' ;;
      OT=*) ot=${extra_env[$k]#OT=}; ot=${ot//;/,}; unset 'extra_env[k]' ;;
      ARGS=*) a=${extra_env[$k]#ARGS=}; read -r -a entry_args <<< "${a//;/ }"; unset 'extra_env[k]' ;;
    esac
  done
  [ "$kind" = up ] && kind_args+=("$share")
  kind_args+=("${entry_args[@]}")
  log=$O/server-$name.log
  echo "### $name ($kind): cache $cache MiB${kind_args[*]:+, ${kind_args[*]}}${settings:+, $settings} $(date +%T)"
  if [ -n "${DROP_CACHE:-}" ]; then
    python3 -I -c 'import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)' "$(dirname "$model")"/*.gguf
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500 > "$O/mem-$name.csv" &
  mpid=$!
  t0=$SECONDS
  env "${base_env[@]}" "${extra_env[@]}" \
    "$bin/llama-server" -m "$model" -ngl 99 -sm layer -ts "$ts" -fa 1 -lm dio -t "$threads" -ot "$ot" \
    -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib "$cache" "${kind_args[@]}" --checkpoint-min-step 32768 \
    --jinja --chat-template-file "$tmpl" --no-reasoning-preserve \
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
  spid=$!
  up=1
  until grep -q "listening on" "$log"; do
    kill -0 $spid 2>/dev/null || { echo "  server died, see $log"; up=0; break; }
    sleep 1
  done
  if [ $up = 1 ]; then
    echo "  loaded in $((SECONDS - t0)) s; memory used after loading (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    for t in $TESTS; do
      case $t in
        template) TMPL_URL=http://127.0.0.1:$PORT python3 -I "$E/upstream_template.py" dump "$O/template-$name.json" | sed 's/^/  template /' ;;
        greedy) GREEDY_URL=http://127.0.0.1:$PORT python3 -I "$E/greedy_kl.py" run "$E/corpus.txt" "$O/greedy-$name.json" \
            --prompts "${PROMPTS:-8}" | tail -2 | sed 's/^/  greedy /' ;;
        same) CTX_URL=http://127.0.0.1:$PORT python3 -I "$E/ctx_size.py" "$E/corpus.txt" same "$O/same-$name.json" | sed 's/^/  /' ;;
        session) DECODE_URL=http://127.0.0.1:$PORT python3 -I "$E/decode_modes.py" "$E/corpus.txt" --repeat "${REPEAT:-6}" \
            --rounds "${ROUNDS:-3}" --max-tokens 256 ${SEED:+--seed "$SEED"} > "$O/session-$name.txt" 2>&1 ||
            echo "  session failed, see $O/session-$name.txt"
          head -1 "$O/session-$name.txt" | sed 's/^/  session /' ;;
        agent) AGENT_URL=http://127.0.0.1:$PORT python3 -I "$E/agent_turns.py" "$E/corpus.txt" --turns "${AGENT_TURNS:-10}" \
            --repeat "${AGENT_REPEAT:-3}" --max-tokens "${AGENT_TOKENS:-1024}" > "$O/agent-$name.txt" 2>&1 ||
            echo "  agent_turns.py failed, see $O/agent-$name.txt"
          sed 's/^/  agent /' "$O/agent-$name.txt" ;;
        fill) CTX_URL=http://127.0.0.1:$PORT python3 -I "$E/ctx_size.py" "$E/corpus.txt" fill "${FILL:-196608}" "$O/fill-$name.json" | sed 's/^/  /' ;;
        split) CKPT_URL=http://127.0.0.1:$PORT python3 -I "$E/ckpt_split.py" "$E/corpus.txt" run "$O/split-$name.json" --log "$log" |
            sed 's/^/  split /' ;;
        short) curl -s -m 300 "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
            -d '{"prompt": "The capital of France is", "n_predict": 8, "temperature": 0, "cache_prompt": false}' |
            python3 -I -c 'import json, sys; r = json.load(sys.stdin); print(repr(r.get("content")), r.get("timings", {}).get("predicted_per_second"))' |
            sed 's/^/  short /' ;;
      esac
    done
    kill -TERM $spid 2>/dev/null
    for i in $(seq 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
    kill -0 $spid 2>/dev/null && kill -TERM $spid
  fi
  wait $spid 2>/dev/null
  echo "  exit $? $(date +%T)"
  kill $mpid 2>/dev/null
  wait $mpid 2>/dev/null
  awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { printf "  memory peak (MiB): GPU0 %d, GPU1 %d of 97887\n", m[0], m[1] }' "$O/mem-$name.csv"
  grep -a -E "(model|KV|RS|compute|output) +buffer size|MoE cache size|layers, .*slots|llama_moe_cache: |abyss: decode stats" "$log" |
    grep -v "decode stats" | sed 's/^/  log: /' | head -40
  # the session's turns are the requests of 256 tokens
  awk '/       eval time/ && / 256 tokens/ { match($0, /[0-9.]+ tokens per second/); s += substr($0, RSTART, RLENGTH) + 0; n++ }
    END { if (n) printf "  session decode: %d turns, mean %.2f tok/s\n", n, s / n }' "$log"
  awk '/abyss: decode stats/ { n++; match($0, /[0-9.]+ ms\/token/); t += substr($0, RSTART, RLENGTH) + 0
      match($0, /hit rate [0-9.]+%/); h += substr($0, RSTART + 9, RLENGTH - 10) + 0 }
    END { if (n) printf "  decode stats: %d windows, %.1f ms/token, hit rate %.1f%%\n", n, t / n, h / n }' "$log"
  if [ -n "$first" ]; then
    for t in $TESTS; do
      case $t in
        template) [ -s "$O/template-$first.json" ] && [ -s "$O/template-$name.json" ] &&
          python3 -I "$E/upstream_template.py" compare "$O/template-$first.json" "$O/template-$name.json" | sed "s/^/  $first vs $name, template /" ;;
        greedy) [ -s "$O/greedy-$first.json" ] && [ -s "$O/greedy-$name.json" ] &&
          python3 -I "$E/greedy_kl.py" compare "$O/greedy-$first.json" "$O/greedy-$name.json" | sed "s/^/  $first vs $name, corpus slices: /" ;;
        same) [ -s "$O/same-$first.json" ] && [ -s "$O/same-$name.json" ] &&
          echo "  $first vs $name, 96k prompt: $(python3 -I "$E/ctx_size.py" "$E/corpus.txt" compare "$O/same-$first.json" "$O/same-$name.json")" ;;
        split) [ -s "$O/split-$first.json" ] && [ -s "$O/split-$name.json" ] &&
          python3 -I "$E/ckpt_split.py" "$E/corpus.txt" compare "$O/split-$first.json" "$O/split-$name.json" | sed "s/^/  $first vs $name, split /" ;;
      esac
    done
  fi
  [ -z "$first" ] && first=$name
  gpus_free
  if [ $up = 0 ] && [ -n "${STOP_ON_FAIL:-}" ]; then
    echo "### $name did not come up, skipping the rest"
    break
  fi
done

if [ "${RESTART:-}" != 0 ] && { [ -n "$pid" ] || [ "${RESTART:-}" = 1 ]; }; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### upstream-bench done $(date +%T), logs in $O"
