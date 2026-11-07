#!/bin/bash
# Whether a server's device memory keeps growing over prefills of many sizes (the q8_1 cache's pool of its own of
# mx-llama.cpp/0036 kept a buffer for each size it was asked for, 0037 caches only entries of up to 128 KiB in buffers
# of that one size): one server with the production settings per entry of CONFIGS ("name:VAR=VALUE,VAR=VALUE", the
# settings on top of those, BIN=<dir> among them for another build; default the cache off, 0036's pool from build-m2
# and 0037's from build-pf), each running N chats (default 40) of corpus slices of pseudo-random lengths (the same
# lengths for every entry, about 300 to 18,000 tokens, no prompt cache, 4 tokens out), with nvidia-smi's memory of
# both GPUs after each; prints them and, per GPU, after loading, the most and the last. Waits for the production
# server to go idle, saves its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the
# slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/q81-creep-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
CONFIGS=${CONFIGS:-"off leg36:BIN=$HOME/src/mx-llama.cpp/build-m2/bin,GGML_CUDA_Q8_1_CACHE=1,ABYSS_Q8_1_CACHE_LEG=1 leg37:GGML_CUDA_Q8_1_CACHE=1,ABYSS_Q8_1_CACHE_LEG=1"}
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

for c in $CONFIGS; do
  IFS=: read -r name settings <<< "$c"
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
    python3 -I - "http://127.0.0.1:$PORT" "$E/corpus.txt" "${N:-40}" > "$O/creep-$name.txt" 2>&1 <<'PY' || echo "  requests failed, see $O/creep-$name.txt"
import json, random, subprocess, sys, time, urllib.request
url, corpus, n = sys.argv[1], open(sys.argv[2], encoding="utf-8").read(), int(sys.argv[3])
corpus = corpus * (60000 // len(corpus) + 2)  # the corpus is shorter than the longest slices
def used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]
rng = random.Random(1)
print("loaded", *used(), flush=True)
for i in range(n):
    chars = rng.randint(1000, 60000)
    start = rng.randint(0, len(corpus) - chars)
    body = {"messages": [{"role": "user", "content": corpus[start:start + chars] + "\n\n上面这段讲了什么？"}],
            "max_tokens": 4, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        t = json.load(r)["timings"]
    print(f"request {i}: {t['prompt_n']} tok in {time.time() - t0:.1f} s, used", *used(), flush=True)
PY
    kill -TERM $spid 2>/dev/null
    for i in $(seq 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
    kill -0 $spid 2>/dev/null && kill -TERM $spid
  fi
  wait $spid 2>/dev/null
  echo "  exit $? $(date +%T)"
  awk '/^loaded/ { l0 = $2; l1 = $3 } /^request/ { n++; u0 = $(NF-1); u1 = $NF; if (u0 > m0) m0 = u0; if (u1 > m1) m1 = u1
         if (n == 5) { f0 = u0; f1 = u1 } }
       END { printf "  %d requests; GPU0 loaded %d, after 5 %d, most %d, last %d MiB; GPU1 loaded %d, after 5 %d, most %d, last %d MiB\n",
             n, l0, f0, m0, u0, l1, f1, m1, u1 }' "$O/creep-$name.txt"
  grep -a -c "did not fit" "$log" | sed 's/^/  "did not fit" warnings: /'
  gpus_free
done

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### q81-creep done $(date +%T), logs in $O"
