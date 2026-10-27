#!/bin/bash
# Context sizes beyond 128K (ctx_size.py). For each configuration NAME:CTX:MOE_CACHE_MIB:UBATCH in CONFIGS, a server
# with the production settings: the memory after loading and the peak while it runs a 96k-token prompt (its outputs
# must match across the configurations) and a prompt filling the context (a passphrase at the start, asked about at
# the end; prefill and decode speed at that context). Each fill length in FILLCMP that fits is run again on the larger
# configurations, so outputs past 128K are compared too.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)
CONFIGS=${CONFIGS:-"c128:131072:14000:2048 c192:196608:14000:2048 c256:262144:14000:2048"}
FILLCMP=${FILLCMP:-196608}

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi
rm -f "$O"/ctx-*-same.json "$O"/ctx-*-fill*.json

start() {  # context, expert cache MiB, ubatch
  log=$O/server-ctx-$(date +%m%d-%H%M%S).log
  ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 ABYSS_DSV41_INDEXER=1 \
    ABYSS_MOE_BIG_SPLIT=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 \
    GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 \
    "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
    -c "$1" -b 65536 -ub "$3" -np 1 --moe-cache-mib "$2" --checkpoint-min-step 32768 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
    --host 127.0.0.1 --port 8080 > "$log" 2>&1 &
  pid=$!
  until grep -q "listening on" "$log"; do
    if ! kill -0 $pid 2>/dev/null; then
      echo "    server died while loading, see $log"
      grep -a -i -E "out of memory|failed to allocate|error" "$log" | head -3 | sed 's/^/    /'
      return 1
    fi
    sleep 3
  done
}
stop() {
  kill -TERM $pid 2>/dev/null
  wait $pid 2>/dev/null
  grep -a "compute buffer size of" "$log" | sed 's/^.*~llama_context: */    server: /'
}
step() { python3 -I "$E/ctx_size.py" "$E/corpus.txt" "$@"; }

for cfg in $CONFIGS; do
  IFS=: read -r name ctx cache ub <<< "$cfg"
  echo "### $name: -c $ctx --moe-cache-mib $cache -ub $ub"
  start "$ctx" "$cache" "$ub" || continue
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{ printf "    loaded: gpu%s %d MiB\n", $1, $2 }'
  mem=$O/ctx-$name-mem.csv
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500 > "$mem" &
  smi=$!
  step same "$O/ctx-$name-same.json"
  step fill "$ctx" "$O/ctx-$name-fill.json"
  for c in $FILLCMP; do
    if [ "$c" -lt "$ctx" ]; then step fill "$c" "$O/ctx-$name-fill$c.json"; fi
  done
  kill $smi
  awk -F', ' '{ if ($2 > m[$1]) m[$1] = $2 } END { for (g in m) printf "    peak: gpu%s %d MiB\n", g, m[g] }' "$mem"
  kill -0 $pid 2>/dev/null || echo "    server died, see $log"
  stop
done

# the 96k prompt against the first configuration; each FILLCMP fill against the configuration of that size
IFS=: read -r first _ <<< "${CONFIGS%% *}"
for cfg in $CONFIGS; do
  IFS=: read -r name ctx _ <<< "$cfg"
  if [ "$name" != "$first" ] && [ -f "$O/ctx-$name-same.json" ]; then
    echo "same, $first vs $name: $(step compare "$O/ctx-$first-same.json" "$O/ctx-$name-same.json")"
  fi
  for c in $FILLCMP; do
    [ "$ctx" = "$c" ] || continue
    for other in $CONFIGS; do
      oname=${other%%:*}
      if [ -f "$O/ctx-$name-fill.json" ] && [ -f "$O/ctx-$oname-fill$c.json" ]; then
        echo "fill $c, $name vs $oname: $(step compare "$O/ctx-$name-fill.json" "$O/ctx-$oname-fill$c.json")"
      fi
    done
  done
done
echo "### done"
