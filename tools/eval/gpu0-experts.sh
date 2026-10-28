#!/bin/bash
# GPU0's spare memory for the expert cache: some of layer 12's expert tensors move to GPU0 (-ot) and the expert cache on
# GPU1 grows by what they free. For each NAME:CACHE_MIB:PATTERN in VARIANTS (an empty PATTERN moves nothing),
# ctx-size.sh at 192K (memory after loading and peak, the 96k prompt, a prompt filling the context) and decode_modes.sh
# (an append-only session of REPEAT corpus copies, ROUNDS turns per mode), both with the production settings; then each
# variant's outputs against the first one's. Waits for the production server to go idle, saves its slot and stops it,
# and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/gpu0-experts-$(date +%m%d-%H%M)
VARIANTS=${VARIANTS:-'base:16000: down:18000:blk\.12\.ffn_down_exps\.=CUDA0 gateup:20500:blk\.12\.ffn_(gate|up)_exps\.=CUDA0'}
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
  echo "### production stopped $(date +%T)"
fi
gpus_free() {  # the driver hands back a stopped server's device memory a few seconds after the process is gone
  for i in $(seq 60); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }' && return
    sleep 1
  done
}
gpus_free

# in a subshell, so restart.sh below does not inherit the settings
(
  export MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
  export PORT=8090 GOMP_SPINCOUNT=100000000 ABYSS_DECODE_STATS=100 ABYSS_DSV41_IDX_CHUNK=256
  for v in $VARIANTS; do
    IFS=: read -r name cache pat <<< "$v"
    ot=${pat:+-ot $pat}
    mkdir -p "$O/$name"
    echo "### $name: expert cache $cache MiB, ${pat:-no expert tensors moved} $(date +%T)"
    OUT=$O/$name EXTRA=$ot CONFIGS="$name:196608:$cache:2048" FILLCMP="" "$E/ctx-size.sh"
    gpus_free
    OUT=$O/$name REPEAT=${REPEAT:-6} ROUNDS=${ROUNDS:-3} \
      EXTRA="-c 196608 --moe-cache-mib $cache --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 $ot" \
      "$E/decode_modes.sh" > "$O/$name/decode.log" 2>&1
    awk '/eval .* 256 tokens/ { match($0, /\( *[0-9.]+ tokens per second\)$/); split(substr($0, RSTART + 1), a, " ");
      s += a[1]; n++ } END { if (n) printf "    decode: %d turns, mean %.2f tok/s\n", n, s / n }' "$O/$name/decode.log"
    grep "ubatch <= 8" "$O/$name/decode.log" | sed 's/^ */    /'
    gpus_free
  done

  echo "### outputs against the first variant"
  IFS=: read -r first _ <<< "${VARIANTS%% *}"
  for v in $VARIANTS; do
    IFS=: read -r name _ <<< "$v"
    [ "$name" = "$first" ] && continue
    for p in same fill; do
      a=$O/$first/ctx-$first-$p.json b=$O/$name/ctx-$name-$p.json
      [ -f "$a" ] && [ -f "$b" ] && echo "  $p, $first vs $name: $(python3 -I "$E/ctx_size.py" "$E/corpus.txt" compare "$a" "$b")"
    done
  done
)

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
fi
echo "### all done $(date +%T), results in $O"
