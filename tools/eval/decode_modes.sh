#!/bin/bash
# Decode speed of a long agent-like conversation (decode_modes.py) on a server with the production settings: per turn
# the mode (greedy or default sampling, streamed or not), the prefill of what the turn appended and the decode speed
# from the server log. REPEAT (corpus copies in turn 1, default 6, about 96k tokens), ROUNDS (turns per mode, default
# 2), MAX_TOKENS (default 256) and MODES are passed on; EXTRA adds server arguments, MODEL and PORT override the model
# and the port (another port keeps clients of the production server out). CAPTURE=1 runs the server inside
# nsys and captures the last turn into $OUT/decode-modes.nsys-rep, then prints the per-token breakdown of its decode.
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8080}
export DECODE_URL=http://127.0.0.1:$PORT
E=$(cd "$(dirname "$0")" && pwd)

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi
if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi

log=$O/server-modes-$(date +%m%d-%H%M).log
REP=$O/decode-modes
NSYS=()
[ -n "$CAPTURE" ] && NSYS=(nsys launch --session-new=abyss --trace=cuda,osrt --cuda-graph-trace=node)
ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_ENGRAM_PREFETCH=8 ABYSS_CED_PREFILL=1 ABYSS_DSV41_INDEXER=1 \
  ABYSS_MOE_BIG_SPLIT=${ABYSS_MOE_BIG_SPLIT:-1} ABYSS_CKPT_TAIL_MERGE=${ABYSS_CKPT_TAIL_MERGE:-2700} ABYSS_CED_ALIGN=${ABYSS_CED_ALIGN:-1} \
  GGML_CUDA_Q8_1_CACHE=${GGML_CUDA_Q8_1_CACHE:-0} GGML_CUDA_MMQ_PREC=q8 "${NSYS[@]}" "$B/llama-server" -m "$M" \
  -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
  -c 131072 -b 65536 -ub 2048 -np 1 --moe-cache-mib 14000 \
  --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" \
  --host 127.0.0.1 --port "$PORT" $EXTRA > "$log" 2>&1 &
pid=$!
until grep -q "listening on" "$log"; do
  kill -0 $pid 2>/dev/null || { echo "server died, see $log"; exit 1; }
  sleep 5
done
python3 -I "$E/decode_modes.py" "$E/corpus.txt" --repeat "${REPEAT:-6}" --rounds "${ROUNDS:-2}" \
  --max-tokens "${MAX_TOKENS:-256}" ${MODES:+--modes "$MODES"} ${CAPTURE:+--capture "$REP"}
if [ -n "$CAPTURE" ]; then
  nsys shutdown --session=abyss --kill=sigterm
else
  kill -TERM $pid
fi
wait $pid
echo "### server timings (prompt = what the turn appended)"
grep -E "prompt eval time|       eval time|stop processing" "$log" | awk '
  /prompt eval time/ { split($0, a, "= *"); p = a[2] }
  /       eval time/ { split($0, a, "= *"); e = a[2] }
  /stop processing/ { match($0, /n_tokens = [0-9]+/); n = substr($0, RSTART + 11, RLENGTH - 11)
    printf "  ctx %6d | prompt %s | eval %s\n", n, p, e }' | sed -E 's/ \( *[0-9.]+ ms per token, / (/g'
grep "abyss: moe cache" "$log" | sed 's/^/  /'
if [ -n "$CAPTURE" ]; then
  nsys export --type sqlite --output "$REP.sqlite" --force-overwrite=true "$REP.nsys-rep" > /dev/null
  python3 -I "$E/nsys_decode.py" "$REP.sqlite" "${MAX_TOKENS:-256}"
fi
echo "### done, server log $log"
