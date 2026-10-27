#!/bin/bash
# One configuration of the DSpark experiment: a server on PORT (default 8090) with the production settings at CTX
# (default 65536) and MOE_CACHE MiB of expert cache (default 12000), plus EXTRA (the draft model arguments, empty for
# the baseline). Runs the append-only conversation (decode_modes.py, REPEAT corpus copies, ROUNDS turns per mode) and
# greedy code generation (dspark_code.py, answers saved as $OUT/dspark-NAME.json), then prints the decode speed, the
# draft acceptance of each request and the GPU memory after loading. The production server must be stopped first.
#
#   NAME=base dspark.sh
#   NAME=dspark EXTRA="-md DRAFT.gguf --spec-type draft-dspark --spec-draft-n-max 5 -ngld 99 -devd CUDA1" dspark.sh
O=${OUT:-$HOME/abyss-eval}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
PORT=${PORT:-8090}
NAME=${NAME:-run}
E=$(cd "$(dirname "$0")" && pwd)
export DECODE_URL=http://127.0.0.1:$PORT

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi

log=$O/server-dspark-$NAME-$(date +%m%d-%H%M).log
GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_DECODE_STATS=100 \
  ABYSS_CED_PREFILL=${ABYSS_CED_PREFILL:-1} ABYSS_ENGRAM_PREFETCH=8 ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_DECODE_GATHER=1 \
  ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_CKPT_THIN_STEP=8192 \
  ABYSS_PLAIN_INPUT=1 \
  "$B/llama-server" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
  -c "${CTX:-65536}" -b 65536 -ub 2048 -np 1 --moe-cache-mib "${MOE_CACHE:-12000}" --checkpoint-min-step 32768 \
  --jinja --chat-template-file "$HOME/abyss-stats/tmpl/v41.jinja" --no-reasoning-preserve \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 --host 127.0.0.1 --port "$PORT" $EXTRA > "$log" 2>&1 &
pid=$!
until grep -q "listening on" "$log"; do
  kill -0 $pid 2>/dev/null || { echo "server died, see $log"; tail -5 "$log"; exit 1; }
  sleep 5
done
echo "### loaded, GPU memory (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ')"
python3 -I "$E/decode_modes.py" "$E/corpus.txt" --repeat "${REPEAT:-3}" --rounds "${ROUNDS:-2}" --max-tokens 256
python3 -I "$E/dspark_code.py" "$O/dspark-$NAME.json" 512
echo "### peak-ish GPU memory (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ')"
kill -TERM $pid
wait $pid
echo "### requests (prompt = what the request appended)"
grep -E "prompt eval time|       eval time|draft acceptance|stop processing" "$log" | awk '
  /prompt eval time/ { split($0, a, "= *"); p = a[2] }
  /       eval time/ { split($0, a, "= *"); e = a[2] }
  /draft acceptance/ { sub(/.*draft acceptance/, "draft acceptance"); d = $0 }
  /stop processing/ { match($0, /n_tokens = [0-9]+/); n = substr($0, RSTART + 11, RLENGTH - 11)
    printf "  ctx %6d | prompt %s | eval %s%s\n", n, p, e, (d ? " | " d : ""); d = "" }' | sed -E 's/ \( *[0-9.]+ ms per token, / (/g'
echo "### done, server log $log"
