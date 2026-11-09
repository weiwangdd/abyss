#!/bin/bash
# Where the wall time of a decoded token goes on the server side (ABYSS_SERVER_STATS, mx-llama.cpp/0033), on servers
# with the production settings, one per entry of CONFIGS ("name:VAR=VALUE,VAR=VALUE", the settings on top of those,
# BIN=<dir> among them for another build; default one entry with the production settings): the append-only session of
# decode_modes.py (REPEAT corpus copies, ROUNDS turns per mode of 256 tokens, with the agent tools), whose turns each
# print one server stats line when they end, then one streamed request with the tools and the server's sampling that
# may generate up to LONG tokens (default 3000; it can end sooner). Prints each turn's mode and time next to its stats
# line, and the decode stats. BUILD picks the binaries (default build-pf). Waits for the production server to go idle,
# saves its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/server-stats-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
CONFIGS=${CONFIGS:-"prod"}
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
    ABYSS_GEN_CKPT=256 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x ABYSS_CKPT_COUNT_START=1 ABYSS_SERVER_STATS=1000 \
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
    DECODE_URL=http://127.0.0.1:$PORT python3 -I "$E/decode_modes.py" "$E/corpus.txt" --repeat "${REPEAT:-6}" \
      --rounds "${ROUNDS:-3}" --max-tokens 256 > "$O/session-$name.txt" 2>&1 || echo "  session failed, see $O/session-$name.txt"
    # one long streamed answer with the tools, after a short prompt
    python3 -I - "http://127.0.0.1:$PORT" "${LONG:-3000}" > "$O/long-$name.txt" 2>&1 <<'PY' || echo "  long request failed"
import json, sys, time, urllib.request
url, n = sys.argv[1], int(sys.argv[2])
tools = [{"type": "function", "function": {"name": "read_file", "description": "读取一个文件的内容",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
body = {"messages": [{"role": "system", "content": "你是一个代码助手。需要看文件时调用 read_file。"},
                     {"role": "user", "content": "不用看文件。用 C++ 写一个线程安全的 LRU 缓存，带完整的单元测试，"
                                                 "再逐段详细解释设计和每个函数，越详细越好。"}],
        "tools": tools, "max_tokens": n, "stream": True}
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0, chunks = time.time(), 0
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
            chunks += 1
print(f"long: {chunks} chunks in {time.time() - t0:.1f} s")
PY
    kill -TERM $spid 2>/dev/null
    for i in $(seq 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
    kill -0 $spid 2>/dev/null && kill -TERM $spid
  fi
  wait $spid 2>/dev/null
  echo "  exit $? $(date +%T)"
  # the session's turns in order, then the long request, each next to the stats line it printed when it ended
  python3 -I - "$O/session-$name.txt" "$O/long-$name.txt" "$log" <<'PY'
import re, sys
turns = [l.split(":", 1)[1].strip() for l in open(sys.argv[1]) if re.match(r"turn \d+:", l) and "prefill" not in l]
turns += [l.strip() for l in open(sys.argv[2]) if l.startswith("long:")]
stats = [l.split("abyss: server stats, ", 1)[1].strip() for l in open(sys.argv[3], errors="replace")
         if "abyss: server stats" in l]
ends = [s for s in stats if "(request end)" in s]
for i, s in enumerate(ends):
    print(f"  {turns[i] if i < len(turns) else '?':32s} {s}")
if len(ends) != len(turns):
    print(f"  ({len(turns)} requests, {len(ends)} request-end lines, {len(stats)} lines in all)")
PY
  awk '/       eval time/ && / 256 tokens/ { match($0, /[0-9.]+ tokens per second/); s += substr($0, RSTART, RLENGTH) + 0; n++ }
    END { if (n) printf "  session: %d turns, mean %.2f tok/s\n", n, s / n }' "$log"
  awk '/abyss: decode stats/ { n++; match($0, /[0-9.]+ ms\/token/); t += substr($0, RSTART, RLENGTH) + 0
      match($0, /hit rate [0-9.]+%/); h += substr($0, RSTART + 9, RLENGTH - 10) + 0 }
    END { if (n) printf "  decode stats: %d windows, %.2f ms/token, hit rate %.1f%%\n", n, t / n, h / n }' "$log"
  gpus_free
done

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### server-stats done $(date +%T), logs in $O"
