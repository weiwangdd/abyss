#!/bin/bash
# The CED-aware prefill on a small random DeepSeek-V4.1 model, without stopping production: ced_check decodes a random
# prompt in one or more llama_decode() calls and generates a few tokens, once with ABYSS_CED_PREFILL=0 and once with 1,
# and ced_compare.py compares the logits after every call and every generated token. The prefill only skips work that
# nothing reads, so the two must agree to rounding (about 1e-6 on the CPU; on CUDA about 5e-4, where a different ubatch
# size alone moves the logits by 2e-3), and the generated tokens must be the same.
#
# The model is the fixture of upstream's tests/test-llama-archs with ABYSS_TEST_DSV41_CED=1: 12 layers, a decoder of
# six behind a window of 16, a top-k of 8 with candidate blocks, 8 experts. It is written on the first run, with the
# weights drawn at SEED and STDEV (the generator's defaults: a random seed and 0.1). At 0.1 nearly every token selects
# the same two experts; STDEV=1.0 spreads the selection, which the tests of the large-batch split need
# (ON_ENV="ABYSS_MOE_BIG_SPLIT=1 ..." with OUT pointing at a directory of its own).
#
# usage: ced-tiny.sh [ngl]    0 (default) runs on the CPU, 99 offloads every layer to the GPU CUDA_VISIBLE_DEVICES
#                             selects. A case takes about 1 GiB of device memory with the experts in host memory
#                             (EXTRA="--host-experts 2 --moe-cache 48") and more with them on the device: next to
#                             production, GPU1 has about 0.9 GiB left once the CUDA context is up, and a large
#                             ubatch then aborts in an out-of-memory CUDA error
# SRC is the llama.cpp tree (default ~/src/llama.cpp-abyss, built in $SRC/build), OUT the work directory. CASES_FILE
# lists the cases, a line each of "calls ubatch [ced_check options]" (default: the list below; CASES=gpu a shorter
# one). EXTRA goes to every ced_check command line, e.g. "--host-experts 2 --moe-cache 48 --cpu-share 0.7" keeps the
# experts of layers 2 and up in host memory behind a MoE cache. OFF_ENV and ON_ENV add environment to the two sides,
# e.g. ABYSS_CED_ALIGN=1 on both. With misses computed on the CPU the generated rows differ more (1e-2 in places): the
# two sides fill the cache differently, and the CPU and the GPU round differently.
set -u
E=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/src/llama.cpp-abyss}
O=${OUT:-$HOME/abyss-eval/ced-tiny}
PY=${PY:-$HOME/src/comfyui/.venv/bin/python}
NGL=${1:-0}
M=$O/models/deepseek41-moe.gguf
mkdir -p "$O/models"

if [ ! -x "$O/ced_check" ] || [ "$E/ced_check.cpp" -nt "$O/ced_check" ] || [ "$SRC/build/bin/libllama.so" -nt "$O/ced_check" ]; then
  g++ -O2 -std=c++17 -I "$SRC/include" -I "$SRC/ggml/include" "$E/ced_check.cpp" -o "$O/ced_check" \
    -L "$SRC/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$SRC/build/bin" || exit 1
fi
if [ ! -s "$M" ]; then
  # shellcheck disable=SC2086
  ABYSS_TEST_DSV41_CED=1 CUDA_VISIBLE_DEVICES= "$SRC/build/bin/test-llama-archs" -a '^deepseek41$' \
    ${SEED:+-s $SEED} ${STDEV:+-d $STDEV} -o "$O/models/" 2>&1 | tail -1
  [ -s "$M" ] || { echo "no model at $M: build test-llama-archs in $SRC/build"; exit 1; }
fi

cases=$O/cases.txt
if [ -n "${CASES_FILE:-}" ]; then
  cases=$CASES_FILE
elif [ "${CASES:-}" = gpu ]; then
  printf '%s\n' "300 64" "300 512" "257 64" "65 64" "17 64" "1000 64" "1000 256" "999 100" "300,200 64" \
    "120,5,300,64 64" "500,1,1,300 64" > "$cases"
else
  # one call of several lengths and ubatch sizes, prompts split over calls, and outputs in the middle of a call
  printf '%s\n' "300 64" "300 32" "300 128" "300 512" "257 64" "64 64" "65 64" "2 64" "17 64" "40 64" "129 64" \
    "1000 64" "1000 256" "999 100" "300,200 64" "120,5,300,64 64" "500,1,1,300 64" "64,64,64 64" \
    "300 64 --outputs 50" "700 64 --outputs 1" > "$cases"
fi

fail=0
while read -r calls ub extra; do
  [ -z "$calls" ] && continue
  # shellcheck disable=SC2086
  env ABYSS_CED_PREFILL=0 ${OFF_ENV:-} nice -n 19 "$O/ced_check" "$M" "$O/off.bin" --calls "$calls" --ubatch "$ub" \
    --gen "${GEN:-20}" --ngl "$NGL" ${EXTRA:-} $extra > "$O/off.log" 2>&1 ||
    { echo "calls $calls ub $ub $extra: failed without CED"; tail -3 "$O/off.log"; fail=1; continue; }
  # shellcheck disable=SC2086
  env ABYSS_CED_PREFILL=1 ${ON_ENV:-} nice -n 19 "$O/ced_check" "$M" "$O/on.bin" --calls "$calls" --ubatch "$ub" \
    --gen "${GEN:-20}" --ngl "$NGL" ${EXTRA:-} $extra > "$O/on.log" 2>&1 ||
    { echo "calls $calls ub $ub $extra: failed with CED"; grep -E "ASSERT|error|failed" "$O/on.log" | tail -3; fail=1; continue; }
  toff=$(awk '/^call:/ { s += $5 } END { printf "%.0f", s }' "$O/off.log")
  ton=$(awk '/^call:/ { s += $5 } END { printf "%.0f", s }' "$O/on.log")
  printf "calls %-16s ub %-4s %-13s prefill %4s -> %4s ms  " "$calls" "$ub" "$extra" "$toff" "$ton"
  "$PY" "$E/ced_compare.py" "$O/off.bin" "$O/on.bin" 128 || fail=1
done < "$cases"
exit $fail
