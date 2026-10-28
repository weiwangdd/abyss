#!/bin/bash
# The V4.1 indexer prefill in chunks of queries (ABYSS_DSV41_IDX_CHUNK, mx-llama.cpp/0019) against all at once: KLD on
# the corpus (CHUNKS x CTX tokens, ubatch 512, chunks of KL_CHUNK queries), then ctx-size.sh without chunks on BASE and
# with chunks of CHUNK queries on CONFIGS (memory after loading and peak, the 96k prompt, a prompt filling the context),
# the outputs compared; KL=0 skips the KLD. Waits for the production server to go idle, saves its slot and stops it, and
# restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}
export MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
BASE=${BASE:-"c192:196608:12000:2048"}
CONFIGS=${CONFIGS:-"c192:196608:12000:2048 c192g16:196608:16000:2048 c256:262144:12000:2048"}
CHUNK=${CHUNK:-256}

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

if [ "${KL:-1}" != 0 ]; then
  echo "### KLD, chunks of ${KL_CHUNK:-128} queries against all at once $(date +%T)"
  (
    export BIN=$HOME/src/mx-llama.cpp/build-m2/bin CHUNKS=${KL_CHUNKS:-14} CTX=${KL_CTX:-1024}
    export ABYSS_DSV41_INDEXER=1 ABYSS_CED_PREFILL=1
    "$E/kl.sh" base idx-c0 512
    ABYSS_DSV41_IDX_CHUNK=${KL_CHUNK:-128} "$E/kl.sh" cmp idx-c0 512
  )
fi

echo "### long context, all at once $(date +%T)"
mkdir -p "$O/idx-c0" "$O/idx-c$CHUNK"
OUT=$O/idx-c0 PORT=8090 CONFIGS=$BASE FILLCMP="" ABYSS_DSV41_IDX_CHUNK=0 "$E/ctx-size.sh"
echo "### long context, chunks of $CHUNK queries $(date +%T)"
OUT=$O/idx-c$CHUNK PORT=8090 CONFIGS=$CONFIGS FILLCMP=196608 ABYSS_DSV41_IDX_CHUNK=$CHUNK "$E/ctx-size.sh"

echo "### outputs, all at once against chunks"
cmp() { python3 -I "$E/ctx_size.py" "$E/corpus.txt" compare "$1" "$2"; }
IFS=: read -r bname bctx _ <<< "$BASE"
for cfg in $CONFIGS; do
  IFS=: read -r name ctx _ <<< "$cfg"
  a=$O/idx-c0/ctx-$bname-same.json
  b=$O/idx-c$CHUNK/ctx-$name-same.json
  [ -f "$a" ] && [ -f "$b" ] && echo "  96k prompt, $bname all at once vs $name in chunks: $(cmp "$a" "$b")"
  # the fill of the base's length: the configuration's own fill, or the extra fill of that length on a larger one
  a=$O/idx-c0/ctx-$bname-fill.json
  b=$O/idx-c$CHUNK/ctx-$name-fill.json
  [ "$ctx" = "$bctx" ] || b=$O/idx-c$CHUNK/ctx-$name-fill$bctx.json
  [ -f "$a" ] && [ -f "$b" ] && echo "  fill $bctx, $bname all at once vs $name in chunks: $(cmp "$a" "$b")"
done

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
fi
echo "### all done $(date +%T)"
