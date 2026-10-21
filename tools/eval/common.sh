# Shared settings for the plan B evaluation scripts; sourced, runs on blackwell.

BIN=${BIN:-$HOME/src/mx-llama.cpp/build-v41/bin}
MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf}
OUT=${OUT:-$HOME/abyss-eval}
EVAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# the q8_1 activation cache asserts in the CUDA VMM pool (see CLAUDE.md)
export GGML_CUDA_Q8_1_CACHE=0
# MXFP4 MMQ defaults to 4-bit activations on Blackwell; the reference runs FP4 experts with 8-bit ones (see CLAUDE.md)
export GGML_CUDA_MMQ_PREC=${GGML_CUDA_MMQ_PREC:-q8}

PLANB_ARGS=(
  -m "$MODEL" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU'
)

mkdir -p "$OUT"

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running and holds the GPUs; stop it first (~/abyss-stats/restart.sh restarts it)" >&2
  exit 1
fi
