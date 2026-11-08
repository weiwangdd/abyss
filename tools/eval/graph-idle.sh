#!/bin/bash
# Whether a build's CUDA graphs outlive a pause (mx-llama.cpp/0043, upstream 00f113ece): for each build of BUILDS
# (default the production build-m2 and build-pf), the stories260K model on GPU1 with -lv LV (default 5, which the CUDA
# backend's debug lines need), two greedy requests of N_PREDICT tokens IDLE seconds apart (default 15, past the cache's
# 10 s), and the CUDA graphs that finished their warmup (each one captured anew) before and after the pause; a build
# that evicts graphs after a pause warms them all up again, one that keeps them none, and the tokens of both requests
# must be the same in every build. Runs beside the production server: the small model needs a few hundred MiB of GPU1,
# its own CUDA context included.
set -u
O=${OUT:-$HOME/abyss-eval}/graph-idle-$(date +%m%d-%H%M)
BUILDS=${BUILDS:-"build-m2 build-pf"}
MODEL=${MODEL:-$HOME/abyss-eval/tmpl-test/stories260K.gguf}
PORT=${PORT:-8093}
IDLE=${IDLE:-15}
mkdir -p "$O"

ask() {  # the tokens of one greedy completion
  curl -s -m 60 "http://127.0.0.1:$PORT/completion" -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Once upon a time\", \"n_predict\": ${N_PREDICT:-64}, \"temperature\": 0, \"cache_prompt\": true, \"return_tokens\": true}" |
    python3 -c 'import json, sys; print(" ".join(map(str, json.load(sys.stdin).get("tokens", []))))'
}

for b in $BUILDS; do
  log=$O/server-$b.log
  CUDA_VISIBLE_DEVICES=1 "$HOME/src/mx-llama.cpp/$b/bin/llama-server" -m "$MODEL" -ngl 99 -c 4096 -np 1 -lv "${LV:-5}" \
    --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
  pid=$!
  until grep -q "listening on" "$log"; do kill -0 $pid 2>/dev/null || { echo "$b: server died, see $log"; continue 2; }; sleep 1; done
  first=$(ask)
  before=$(grep -c "CUDA graph warmup complete" "$log")
  sleep "$IDLE"
  second=$(ask)
  after=$(( $(grep -c "CUDA graph warmup complete" "$log") - before ))
  kill -TERM $pid; wait $pid 2>/dev/null
  echo "$b: graphs warmed up before the pause $before, after it $after; tokens $([ "$first" = "$second" ] && echo same || echo DIFFER) across the pause"
  echo "$first" > "$O/tokens-$b.txt"
done
set -- $BUILDS
if [ $# -ge 2 ]; then
  cmp -s "$O/tokens-$1.txt" "$O/tokens-$2.txt" && echo "tokens of $1 and $2: the same" || echo "tokens of $1 and $2: DIFFER"
fi
echo "logs in $O"
