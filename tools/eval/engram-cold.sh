#!/bin/bash
# Engram prefetch (ABYSS_ENGRAM_PREFETCH): correctness, then long-prompt prefill on a cold page cache with it off and on.
# The prefetch only reads ahead, so its logits must match the build without it exactly.
# Before each server start the two Engram shards leave the page cache; a short request takes the first-request
# overhead, then the long prompt measures cold Engram reads and a second send of it the warm ones.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-eng/bin}
D=$HOME/models/DeepSeek-V4.1-Flash-GGUF
M=$D/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
SHARDS=("$D/DeepSeek-V4.1-Flash-MXFP4-00009-of-00012.gguf" "$D/DeepSeek-V4.1-Flash-MXFP4-00011-of-00012.gguf")
E=$(cd "$(dirname "$0")" && pwd)
PF=${PF:-8}

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi

echo "### correctness: prefetch on vs the 8k CED-off base of build-ced"
ABYSS_ENGRAM_PREFETCH=$PF BIN=$B CTX=8192 CHUNKS=1 CORPUS="$O/corpus2.txt" BASE="$O/ced-c8192-ub2048.kld" \
  "$E/kl.sh" cmp eng-c8192 2048

drop() {
  python3 -I - "${SHARDS[@]}" <<'PY'
import os, sys
for f in sys.argv[1:]:
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
PY
  awk '/^Cached:/ { printf "page cache after drop: %.1f GiB\n", $2/1048576 }' /proc/meminfo
}

ask() {  # $1 = long | short
  python3 -I - "$E/corpus.txt" "$1" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1], encoding="utf-8").read()
text = corpus * 3 + "\n\n用三句话总结上面的内容。" if sys.argv[2] == "long" else "你好"
body = json.dumps({"messages": [{"role": "user", "content": text}], "max_tokens": 8, "cache_prompt": False}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
print("prompt %d tok %.1f tok/s (%.1f s)" % (t["prompt_n"], t["prompt_per_second"], t["prompt_ms"] / 1000))
PY
}

for pf in 0 $PF; do
  drop
  log=$O/server-engpf$pf-$(date +%m%d-%H%M).log
  ABYSS_ENGRAM_PREFETCH=$pf ABYSS_CED_PREFILL=1 GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 "$B/llama-server" -m "$M" \
    -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
    -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib 14000 \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" \
    --host 127.0.0.1 --port 8080 > "$log" 2>&1 &
  pid=$!
  until grep -q "listening on" "$log"; do
    kill -0 $pid 2>/dev/null || { echo "server died, see $log"; exit 1; }
    sleep 5
  done
  echo "### prefetch $pf, warm-up: $(ask short)"
  echo "### prefetch $pf, cold:    $(ask long)"
  echo "### prefetch $pf, warm:    $(ask long)"
  kill -TERM $pid
  wait $pid
done
echo "### done"
