#!/bin/bash
# Perplexity of the decode path with the MoE expert deferral (ABYSS_MOE_DEFER=1, mx-llama.cpp/0025) and without:
# llama-abyss-dump with the production settings forces, one token at a time after a corpus prompt, the corpus text
# that follows it and a C++ source of the fork (GEN tokens in all, default 4000), saving each token's NLL; then
# nll_compare.py pairs the two runs token by token. BUILD picks the binaries (default build-pf, which has the switch).
# Waits for the production server to go idle, saves its slot and stops it, and restarts it at the end
# (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/defer-ppl-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
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

# the prompt is the start of the corpus, the forced text what follows it and then C++ code
python3 -I - "$E/corpus.txt" "$HOME/src/mx-llama.cpp/src/llama-moe-cache.cpp" "$O" <<'PY'
import sys
corpus = open(sys.argv[1], encoding="utf-8").read()
code = open(sys.argv[2], encoding="utf-8").read()
open(sys.argv[3] + "/prompt.txt", "w", encoding="utf-8").write(corpus[:3000])
open(sys.argv[3] + "/force.txt", "w", encoding="utf-8").write(corpus[3000:6000] + "\n\n" + code[:8000])
PY

for v in 0 1; do
  mkdir -p "$O/defer$v"
  echo "### ABYSS_MOE_DEFER=$v $(date +%T)"
  env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
    ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_ADMIT=0.3 \
    ABYSS_MOE_BIG_SPLIT=1 ABYSS_MOE_BIG_CACHE=1 ABYSS_PLAIN_INPUT=1 ABYSS_DECODE_STATS=100 ABYSS_MOE_DEFER=$v \
    ABYSS_DUMP_RE= ABYSS_DUMP_DIR="$O/defer$v" ABYSS_DUMP_FORCE="$O/force.txt" ABYSS_DUMP_GEN="${GEN:-4000}" \
    "$B/llama-abyss-dump" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' -ot 'blk\.12\.ffn_(gate|up)_exps\.=CUDA0' \
    -c 16384 -b 8192 -ub 2048 --moe-cache-mib 20500 -f "$O/prompt.txt" > "$O/defer$v/dump.log" 2>&1
  echo "  exit $? $(date +%T)"
  grep -a -E "forcing|forced tokens" "$O/defer$v/dump.log" | sed 's/^.*abyss-dump: /  /'
  grep -a "abyss: decode stats" "$O/defer$v/dump.log" | awk '{ match($0, /[0-9.]+ ms\/token/); t += substr($0, RSTART, RLENGTH) + 0; n++ }
    END { if (n) printf "  %d decode stats windows, %.1f ms/token\n", n, t / n }'
  gpus_free
done
if [ -s "$O/defer0/nll.txt" ] && [ -s "$O/defer1/nll.txt" ]; then
  echo "### without (A) vs with (B) the deferral"
  python3 -I "$E/nll_compare.py" "$O/defer0/nll.txt" "$O/defer1/nll.txt" | sed 's/^/  /'
fi

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### all done $(date +%T), results in $O"
