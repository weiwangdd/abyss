#!/bin/bash
# Dump the V4.1 indexer tensors of layers 2, 20 and 24 for a ~4k-token prompt, for indexer_oracle.py:
#   PYTHONPATH=~/src/mx-llama.cpp/gguf-py python indexer_oracle.py ~/abyss-eval/dump-idx ~/models/DeepSeek-V4.1-Flash-GGUF
# Needs a build with mx-llama.cpp/abyss-dump (BUILD) and about 650 MB in OUT.
O=${OUT:-$HOME/abyss-eval/dump-idx}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-dump/bin}
M=$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf
E=$(cd "$(dirname "$0")" && pwd)

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running; stop it first" >&2
  exit 1
fi

rm -rf "$O" && mkdir -p "$O"
python3 -I -c 'import sys; t = open(sys.argv[1], encoding="utf-8").read(); open(sys.argv[2], "w", encoding="utf-8").write(t[:14500])' \
  "$E/corpus.txt" "$O/prompt.txt"
ABYSS_DUMP_RE='^(qr_norm|attn_norm|idx_latent|idx_q|idx_w|idx_k|idx_score|idx_top_k)-(2|20|24)$' ABYSS_DUMP_DIR="$O" \
ABYSS_DSV41_INDEXER=1 GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 "$B/llama-abyss-dump" -m "$M" \
  -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' \
  -c 8192 -b 8192 -ub 2048 -f "$O/prompt.txt" > "$O/dump.log" 2>&1
echo "dump exit $?"
grep -E "abyss-dump" "$O/dump.log" | head -30
du -sh "$O"
