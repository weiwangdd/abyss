#!/bin/bash
# Cheap knobs with the production settings, in one stop of the production server: decode speed with CUDA graphs as they
# are, off (GGML_CUDA_DISABLE_GRAPHS=1) and with ggml's graph optimization (GGML_CUDA_GRAPH_OPT=1), from the append-only
# session of decode_modes.sh (REPEAT corpus copies, ROUNDS turns per mode); and the indexer prefill in chunks of each
# size in CHUNKS (ctx-size.sh at 192K: memory after loading and peak, the 96k prompt, a prompt filling the context), the
# outputs compared with the first size's; SKIP_DECODE=1 leaves the decode runs out. Waits for the production server to go
# idle, saves its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/quick-knobs-$(date +%m%d-%H%M)
CHUNKS=${CHUNKS:-"256 512"}
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

# in a subshell, so restart.sh below does not inherit the settings
(
  export PORT=8090 REPEAT=${REPEAT:-6} ROUNDS=${ROUNDS:-3}
  export MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
  export GOMP_SPINCOUNT=100000000 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3
  ot='-ot blk\.12\.ffn_(gate|up)_exps\.=CUDA0'
  rest="-c 196608 --moe-cache-mib 20500 $ot --checkpoint-min-step 32768 --no-reasoning-preserve --temp 1.0 --top-p 1.0"
  rest="$rest --top-k 0 --min-p 0"

  variants="base nographs graphopt"
  [ "${SKIP_DECODE:-0}" = 1 ] && variants=""
  for v in $variants; do
    case $v in
      nographs) knob=GGML_CUDA_DISABLE_GRAPHS=1 ;;
      graphopt) knob=GGML_CUDA_GRAPH_OPT=1 ;;
      *) knob=ABYSS_NOTHING=1 ;;
    esac
    echo "### decode, $v $(date +%T)"
    env "$knob" ABYSS_DSV41_IDX_CHUNK=256 OUT=$O EXTRA="$rest" "$E/decode_modes.sh" > "$O/decode-$v.log" 2>&1
    awk -v v="$v" '/eval .* 256 tokens/ { match($0, /\( *[0-9.]+ tokens per second\)$/); split(substr($0, RSTART + 1), x, " ");
      s += x[1]; n++ } /ubatch <= 8/ { match($0, /hit rate [0-9.]+%/); h = substr($0, RSTART + 9, RLENGTH - 9) }
      END { if (n) printf "    %s: %d turns, mean %.2f tok/s, hit rate %s\n", v, n, s / n, h; else print "    " v ": no turns" }' \
      "$O/decode-$v.log"
    gpus_free
  done

  for c in $CHUNKS; do
    echo "### indexer chunks of $c $(date +%T)"
    mkdir -p "$O/c$c"
    ABYSS_DSV41_IDX_CHUNK=$c OUT=$O/c$c EXTRA=$ot CONFIGS="c$c:196608:20500:2048" FILLCMP="" "$E/ctx-size.sh"
    gpus_free
  done
  first=${CHUNKS%% *}
  for c in $CHUNKS; do
    [ "$c" = "$first" ] && continue
    for p in same fill; do
      a=$O/c$first/ctx-c$first-$p.json b=$O/c$c/ctx-c$c-$p.json
      [ -f "$a" ] && [ -f "$b" ] && echo "  $p, chunks of $first vs $c: $(python3 -I "$E/ctx_size.py" "$E/corpus.txt" compare "$a" "$b")"
    done
  done
)

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### all done $(date +%T), logs in $O"
