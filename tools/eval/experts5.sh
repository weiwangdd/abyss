#!/bin/bash
# Five routed experts per token instead of six (deepseek41.expert_used_count=5, as Strata's "Faster" mode): KLD against
# six on the corpus (prefill path, CHUNKS x CTX tokens) and the decode speed of an append-only session (decode_modes.sh,
# REPEAT corpus copies), both with the production weights and settings. Waits for the production server to go idle,
# saves its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
OV=(--override-kv deepseek41.expert_used_count=int:5)

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

export BIN=$B MODEL=$M CHUNKS=${CHUNKS:-14} CTX=${CTX:-1024}
echo "### KLD base: six experts $(date +%T)"
"$E/kl.sh" base v1017-e6 512
echo "### KLD: five experts against six $(date +%T)"
"$E/kl.sh" cmp v1017-e6 512 "${OV[@]}"

export GOMP_SPINCOUNT=100000000 ABYSS_DECODE_STATS=100 PORT=${PORT:-8090} REPEAT=${REPEAT:-3} ROUNDS=${ROUNDS:-2}
base="-c 196608 --moe-cache-mib 12000 --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0"
for n in 6 5; do
  echo "### decode, $n experts $(date +%T)"
  if [ "$n" = 5 ]; then extra="$base ${OV[*]}"; else extra=$base; fi
  EXTRA=$extra "$E/decode_modes.sh"
done

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
fi
echo "### all done $(date +%T)"
