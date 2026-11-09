#!/bin/bash
# Where the host thread of a decode step spends its time outside the GPU's work: a test server with the production
# settings (EXTRA_ENV="VAR=VALUE ..." adds settings, BUILD and MODEL another build or model) prefills the first turn
# of the session of decode_modes.py (REPEAT corpus copies, default 6, about 96k tokens), then decodes TOKENS greedy
# tokens (default 2048) after a new question, whose section number changes from run to run so the answer is new text.
# From 1 s into that decode, offcputime-bpfcc records for OFFCPU_SECONDS (default 8, after a few seconds of
# compiling) the user and kernel stacks the server's main thread, which runs the decode loop, leaves the CPU from, and
# for how long; then perf samples the thread at FREQ Hz (default 20000) for PERF_SECONDS (default 8, 0 skips it),
# which sees a thread only while it runs. Both run as root (sudo), since kernel.perf_event_paranoid keeps users out.
# Prints the off-CPU stacks by their time and perf's functions by their share of the samples, by library and by
# symbol. Waits for the production server to go idle, saves its slot and stops it, and restarts it at the end
# (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/host-profile-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
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
# until the stopped server is gone and the driver has handed back its device memory
for i in $(seq 90); do
  if ! pgrep -x llama-server > /dev/null &&
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }'; then
    break
  fi
  sleep 1
done

log=$O/server.log
# shellcheck disable=SC2086
env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
  ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 \
  ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 \
  ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1 ABYSS_EXPERT_STATS="$O/stats.json" \
  ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1 \
  ABYSS_MOE_SKIP_ZERO=1 ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1 ABYSS_MMVQ_TAIL=1 \
  ABYSS_GRAPH_CHUNK=128 ABYSS_ENGRAM_KEEPALIVE_MS=4 ABYSS_HEAD_SPLIT_ONLY=1 \
  ABYSS_GEN_CKPT=256 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x ABYSS_CKPT_COUNT_START=1 ABYSS_WEIGHT_NOWAIT=1 ${EXTRA_ENV:-} \
  "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' -ot 'blk\.12\.ffn_(gate|up)_exps\.=CUDA0' \
  -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib 21100 --checkpoint-min-step 32768 \
  --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
spid=$!
up=1
until grep -q "listening on" "$log"; do
  kill -0 $spid 2>/dev/null || { echo "### server died while loading, see $log"; up=0; break; }
  sleep 2
done
if [ $up = 1 ]; then
  echo "### prefill $(date +%T)"
  # the first turn of decode_modes.py's session, then one long greedy answer to a new question
  URL=http://127.0.0.1:$PORT python3 -I - "$E" "$E/corpus.txt" "${REPEAT:-6}" "${TOKENS:-2048}" > "$O/driver.txt" 2>&1 <<'PY' &
import json, sys, time, urllib.request
sys.path.insert(0, sys.argv[1])
import decode_modes as dm
dm.URL = __import__("os").environ["URL"]
corpus, repeat, tokens = open(sys.argv[2], encoding="utf-8").read(), int(sys.argv[3]), int(sys.argv[4])
messages = [
    {"role": "system", "content": "你是一个代码助手。需要看文件时调用 read_file 或 list_dir。"},
    {"role": "user", "content": "下面是项目的设计文档：\n\n" + corpus * repeat + "\n\n先说说它分几节。"},
]
t0 = time.time()
content, _ = dm.request({"messages": messages, "tools": dm.TOOLS, "max_tokens": 1, "temperature": 0, "cache_prompt": True})
print(f"prefill {time.time() - t0:.1f} s", flush=True)
# a section number that changes from run to run, so the answer is new text whose Engram rows are not all cached yet
messages += [{"role": "assistant", "content": content},
             {"role": "user", "content": dm.QUESTIONS[0].format(int(time.time()) % 40 + 2)}]
print("decode starts", flush=True)
t0 = time.time()
dm.request({"messages": messages, "tools": dm.TOOLS, "max_tokens": tokens, "temperature": 0, "cache_prompt": True,
            "ignore_eos": True})
print(f"decode {time.time() - t0:.1f} s", flush=True)
PY
  drv=$!
  until grep -q "decode starts" "$O/driver.txt"; do
    kill -0 $drv 2>/dev/null || break
    sleep 1
  done
  sleep 1
  echo "### off-CPU stacks of the main thread $(date +%T)"
  sudo offcputime-bpfcc -d -f -t $spid "${OFFCPU_SECONDS:-8}" > "$O/offcpu.txt" 2> "$O/offcpu-err.txt"
  if [ "${PERF_SECONDS:-8}" != 0 ]; then
    echo "### perf on the main thread $(date +%T)"
    sudo perf record -F "${FREQ:-20000}" -t $spid -o "$O/perf.data" -- sleep "${PERF_SECONDS:-8}" > "$O/perf-record.txt" 2>&1
    sudo chown "$(id -u):$(id -g)" "$O/perf.data"
  fi
  wait $drv
  cat "$O/driver.txt"
  grep -a "abyss: decode stats" "$log" | tail -3 | sed 's/^.*abyss: /  /'
  kill -TERM $spid 2>/dev/null
  for i in $(seq 60); do kill -0 $spid 2>/dev/null || break; sleep 1; done
  kill -0 $spid 2>/dev/null && kill -TERM $spid
fi
wait $spid 2>/dev/null
for i in $(seq 90); do
  if ! pgrep -x llama-server > /dev/null &&
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }'; then
    break
  fi
  sleep 1
done

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi

if [ -s "$O/perf.data" ]; then
  echo "### by library"
  perf report -i "$O/perf.data" --no-children --sort dso --stdio 2>/dev/null | grep -v '^#' | grep -v '^$' | head -15
  echo "### by symbol"
  perf report -i "$O/perf.data" --no-children --sort dso,sym --stdio 2>/dev/null | grep -v '^#' | grep -v '^$' | head -70
fi
if [ -s "$O/offcpu.txt" ]; then
  echo "### off-CPU time by stack (the innermost user frames, then the innermost kernel frames)"
  python3 -I - "$O/offcpu.txt" "${OFFCPU_SECONDS:-8}" <<'PY'
import collections, sys
tot, by = 0, collections.Counter()
for line in open(sys.argv[1]):
    stack, _, us = line.rstrip().rpartition(" ")
    if not us.isdigit():
        continue
    # folded with -d: the thread name, the user frames from the outermost, "-", the kernel frames from the outermost
    frames = stack.split(";")[1:]
    user, kernel = (frames[:frames.index("-")], frames[frames.index("-") + 1:]) if "-" in frames else (frames, [])
    by[" < ".join(reversed(user[-4:])) + "  |  " + " < ".join(reversed(kernel[-2:]))] += int(us)
    tot += int(us)
secs = float(sys.argv[2])
print(f"  {tot / 1e3:.1f} ms off the CPU in {secs:.0f} s ({tot / secs / 1e3:.1f} ms per second)")
for k, v in by.most_common(15):
    print(f"  {v / secs / 1e3:6.2f} ms/s  {k}")
PY
fi
echo "### done $(date +%T), reports in $O"
