#!/bin/bash
# M2: decode with the expert cache's misses split between uploads and the CPU (ABYSS_MOE_CPU_MISS=1), for several shares
# of admitted misses (ABYSS_MOE_ADMIT). Each config runs the same prompt (REPEAT copies of the corpus, default 3, about
# 16k tokens each) at temperature 0 with top-20 logprobs, compared with the first config, then reuses the cached prompt
# for a second generation, which is the decode speed. The cache's counts are printed when the server exits.
# CONFIGS overrides the list: space-separated, each a comma-separated list of VAR=value.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)
GEN=${GEN:-256}
CONFIGS=${CONFIGS:-"ABYSS_MOE_CPU_MISS=0 ABYSS_MOE_CPU_MISS=1,ABYSS_MOE_ADMIT=1 ABYSS_MOE_CPU_MISS=1,ABYSS_MOE_ADMIT=0.5 ABYSS_MOE_CPU_MISS=1,ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_CPU_MISS=1,ABYSS_MOE_ADMIT=0"}

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi
if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi

ask() {  # $1 = output json or -, $2 = cache_prompt
  python3 -I - "$E/corpus.txt" "$GEN" "$1" "$2" "${REPEAT:-3}" <<'PY'
import json, sys, urllib.request
corpus = open(sys.argv[1], encoding="utf-8").read()
body = json.dumps({"messages": [{"role": "user", "content": corpus * int(sys.argv[5]) + "\n\n详细总结上面的内容。"}],
                   "max_tokens": int(sys.argv[2]), "temperature": 0, "cache_prompt": sys.argv[4] == "1",
                   "logprobs": True, "top_logprobs": 20}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
t = r["timings"]
print("prompt %d tok (%.1f s) | gen %d tok %.2f tok/s" % (t["prompt_n"], t["prompt_ms"] / 1000, t["predicted_n"],
                                                          t["predicted_per_second"]))
if sys.argv[3] != "-":
    json.dump(r["choices"][0]["logprobs"]["content"], open(sys.argv[3], "w"))
PY
}

compare() {  # $1 = reference json, $2 = json
  python3 -I - "$1" "$2" <<'PY'
import json, math, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
n = min(len(a), len(b))
same = next((i for i in range(n) if a[i]["token"] != b[i]["token"]), n)
kl = []
for i in range(same):
    pa = {t["token"]: t["logprob"] for t in a[i]["top_logprobs"]}
    pb = {t["token"]: t["logprob"] for t in b[i]["top_logprobs"]}
    floor = min(pb.values())
    kl.append(sum(math.exp(lp) * (lp - pb.get(t, floor)) for t, lp in pa.items()))
print(f"identical tokens: first {same} of {n}" + (f", top-20 KL over those mean {sum(kl) / same:.2e} max {max(kl):.2e}" if same else ""))
PY
}

i=0
for cfg in $CONFIGS; do
  env_args=(${cfg//,/ })
  log=$O/server-m2-$i-$(date +%m%d-%H%M).log
  env "${env_args[@]}" ABYSS_DSV41_DECODE_GATHER=1 ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 ABYSS_DSV41_INDEXER=1 \
    GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 "$B/llama-server" -m "$M" \
    -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
    -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
    -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib ${MOE_CACHE_MIB:-14000} \
    --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" \
    --host 127.0.0.1 --port 8080 > "$log" 2>&1 &
  pid=$!
  until grep -q "listening on" "$log"; do
    kill -0 $pid 2>/dev/null || { echo "server died, see $log"; exit 1; }
    sleep 5
  done
  echo "### $cfg"
  echo "  fresh prompt:  $(ask "$O/m2-$i.json" 0)"
  echo "  cached prompt: $(ask - 1)"
  kill -TERM $pid
  wait $pid
  grep "abyss: moe cache" "$log" | sed 's/^/  /'
  [ $i -gt 0 ] && echo "  vs first config: $(compare "$O/m2-0.json" "$O/m2-$i.json")"
  i=$((i + 1))
done
echo "### done"
