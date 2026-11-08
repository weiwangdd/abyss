#!/bin/bash
# How much of the last answer a conversation's next turn computes again (mx-llama.cpp/0041), each entry of CONFIGS
# ("name:VAR=VALUE,VAR=VALUE" on top of the production settings, BIN=<dir> among them for another build; default
# without and with the checkpoints in the answer, ABYSS_GEN_CKPT=256; since 0042 a slot always continues from its end
# when the prompt extends the whole slot, which a build with 0041 but not 0042 does only with ABYSS_CKPT_CONTINUE=1
# among the entry's settings, so the behavior before 0041 needs BIN=<a build without 0041>) on a fresh server:
# gen_ckpt.py (PROMPTS corpus slices, each a greedy turn 1 with the tools, its turn 2 and a turn 2b that
# changes the answer at 60%, with top-20 logprobs; turn 1 must be the same in all entries, turns 2 and 2b are compared
# with the first entry; a later slice's turn 1 can differ already, by the expert cache the earlier turns left), then
# the agent session of agent_turns.py (TURNS turns of MAX_TOKENS tokens, greedy, on REPEAT corpus copies). Prints per
# entry the server's "abyss: the prompt ... continuing from" lines next to the requests, the checkpoints made in answers
# and the agent table. BUILD picks the binaries (default build-pf). Waits for the production server to go idle, saves
# its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/gen-ckpt-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
CONFIGS=${CONFIGS:-"cont:ABYSS_GEN_CKPT=0 gen"}
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

names=()
for c in $CONFIGS; do
  IFS=: read -r name settings <<< "$c"
  names+=("$name")
  extra_env=()
  [ -n "$settings" ] && IFS=, read -r -a extra_env <<< "$settings"
  bin=$B
  for k in "${!extra_env[@]}"; do
    if [[ ${extra_env[$k]} == BIN=* ]]; then
      bin=${extra_env[$k]#BIN=}
      unset 'extra_env[k]'
    fi
  done
  log=$O/server-$name.log
  echo "### $name: ${settings:-production} $(date +%T)"
  env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
    ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 \
    ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 \
    ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1 ABYSS_EXPERT_STATS="$O/stats-$name.json" \
    ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1 \
    ABYSS_MOE_SKIP_ZERO=1 ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1 ABYSS_MMVQ_TAIL=1 \
    ABYSS_GRAPH_CHUNK=128 ABYSS_ENGRAM_KEEPALIVE_MS=4 ABYSS_HEAD_SPLIT_ONLY=1 \
    ABYSS_GEN_CKPT=256 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x \
    "${extra_env[@]}" \
    "$bin/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' -ot 'blk\.12\.ffn_(gate|up)_exps\.=CUDA0' \
    -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib 21100 --checkpoint-min-step 32768 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
  spid=$!
  up=1
  until grep -q "listening on" "$log"; do
    kill -0 $spid 2>/dev/null || { echo "  server died, see $log"; up=0; break; }
    sleep 3
  done
  if [ $up = 1 ]; then
    GEN_CKPT_URL=http://127.0.0.1:$PORT python3 -I "$E/gen_ckpt.py" "$E/corpus.txt" "$O/kl-$name.json" \
      --prompts "${PROMPTS:-4}" > "$O/kl-$name.txt" 2>&1 || echo "  gen_ckpt.py failed, see $O/kl-$name.txt"
    AGENT_URL=http://127.0.0.1:$PORT python3 -I "$E/agent_turns.py" "$E/corpus.txt" --turns "${TURNS:-10}" \
      --repeat "${REPEAT:-3}" --max-tokens "${MAX_TOKENS:-1024}" > "$O/agent-$name.txt" 2>&1 ||
      echo "  agent_turns.py failed, see $O/agent-$name.txt"
    kill -TERM $spid 2>/dev/null
    for i in $(seq 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
    kill -0 $spid 2>/dev/null && kill -TERM $spid
  fi
  wait $spid 2>/dev/null
  echo "  exit $? $(date +%T)"
  cat "$O/kl-$name.txt" | sed 's/^/  /'
  cat "$O/agent-$name.txt" | sed 's/^/  /'
  # where each request continued the slot, with its prompt time, and the checkpoints made in answers
  grep -a "abyss: the prompt of\|prompt eval time\|checkpoints in the answer" "$log" |
    sed 's/^.*abyss: /  /; s/^.*prompt eval time = */    prompt eval /'
  gpus_free
done

if [ ${#names[@]} -ge 2 ]; then
  for name in "${names[@]:1}"; do
    echo "### ${names[0]} vs $name, per slice turn 1, turn 2, turn 2b"
    python3 -I "$E/greedy_kl.py" compare "$O/kl-${names[0]}.json" "$O/kl-$name.json"
    # the turns apart: turn 1 must match, turns 2 and 2b carry the change
    for k in 0 1 2; do
      python3 -I - "$O/kl-${names[0]}.json" "$O/kl-$name.json" "$O/_a.json" "$O/_b.json" "$k" <<'PY'
import json, sys
for src, dst in ((sys.argv[1], sys.argv[3]), (sys.argv[2], sys.argv[4])):
    json.dump(json.load(open(src))[int(sys.argv[5])::3], open(dst, "w"))
PY
      echo "  $(echo "turn 1|turn 2|turn 2b" | cut -d'|' -f$((k + 1))): $(python3 -I "$E/greedy_kl.py" compare "$O/_a.json" "$O/_b.json" | tr '\n' ' ')"
    done
    rm -f "$O/_a.json" "$O/_b.json"
  done
fi

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### gen-ckpt done $(date +%T), logs in $O"
