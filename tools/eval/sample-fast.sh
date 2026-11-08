#!/bin/bash
# The sampling fast path (ABYSS_SAMPLE_FAST=1, mx-llama.cpp/0035: when the chain reduces to dist or greedy over the
# logits, as with the production sampling, the token comes from the logits themselves instead of the chain building
# the 129,280 candidates and writing their probabilities) off and on, each on a fresh server with the production
# settings (which have it on) and the server stats of 0033: seeded_tokens.py's requests (corpus slices answered with
# the server's sampling and fixed seeds and greedily, two chats that call a tool), whose tokens must be the same, each
# printed next to the stats line it ended with, then the mean sampling time and time per token of each kind of
# request. Only requests without probabilities and without ignore_eos take the fast path, so decode_modes.py's session
# does not. CONFIGS ("name:VAR=VALUE,VAR=VALUE" on top of the production settings, BIN=<dir> among them for another
# build) defaults to the two; BUILD picks the binaries (default build-pf). Waits for the production server to go idle,
# saves its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/sample-fast-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
CONFIGS=${CONFIGS:-"off:ABYSS_SAMPLE_FAST=0 fast"}
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
    ABYSS_GEN_CKPT=256 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x ABYSS_SERVER_STATS=1000 \
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
    SEEDED_URL=http://127.0.0.1:$PORT python3 -I "$E/seeded_tokens.py" run "$E/corpus.txt" "$O/seeded-$name.json" \
      > "$O/seeded-$name.txt" 2>&1 || echo "  seeded requests failed, see $O/seeded-$name.txt"
    kill -TERM $spid 2>/dev/null
    for i in $(seq 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
    kill -0 $spid 2>/dev/null && kill -TERM $spid
  fi
  wait $spid 2>/dev/null
  echo "  exit $? $(date +%T)"
  # the requests in order, each next to the stats line it printed when it ended (one that generates a single token
  # decodes none and prints none), then the mean sampling time and time per token of each kind
  python3 -I - "$O/seeded-$name.txt" "$log" <<'PY'
import re, sys
reqs = [l.strip() for l in open(sys.argv[1]) if re.match(r"(sampled|greedy|tools) \d+:", l)]
reqs = [r for r in reqs if int(re.search(r"gen (\d+) tok", r).group(1)) > 1]
ends = [l.split("abyss: server stats, ", 1)[1].strip() for l in open(sys.argv[2], errors="replace")
        if "abyss: server stats" in l and "(request end)" in l]
kinds = {}
for i, s in enumerate(ends):
    print(f"  {reqs[i] if i < len(reqs) else '?':72s} {s}")
    m, t = re.search(r"\+ sampling ([0-9.]+)", s), re.search(r"([0-9.]+) ms/token", s)
    if m and t and i < len(reqs):
        kinds.setdefault(reqs[i].split()[0], []).append((float(m.group(1)), float(t.group(1))))
if len(ends) != len(reqs):
    print(f"  ({len(reqs)} requests that decode, {len(ends)} request-end lines)")
mean = lambda xs: sum(xs) / len(xs)
print("  per token: " + ", ".join(f"{k} sampling {mean([x for x, _ in v]):.3f} of {mean([y for _, y in v]):.2f} ms"
                                  for k, v in kinds.items()))
PY
  gpus_free
done

if [ ${#names[@]} -ge 2 ]; then
  for name in "${names[@]:1}"; do
    echo "### ${names[0]} vs $name"
    python3 -I "$E/seeded_tokens.py" compare "$O/seeded-${names[0]}.json" "$O/seeded-$name.json"
  done
fi

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### sample-fast done $(date +%T), logs in $O"
