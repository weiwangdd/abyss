#!/bin/bash
# Plan B speed baseline: decode at empty and 16k context, prefill at 2k and 8k.
#
#   bench.sh <tag> [extra llama args...]   results in $OUT/bench-<tag>-<date>.md
set -euo pipefail
source "$(dirname "$0")/common.sh"

tag=$1; shift
md="$OUT/bench-$tag-$(date +%m%d-%H%M).md"
reps=${REPS:-2}

{
  echo "# bench $tag $(date '+%F %T')"
  echo
  echo "extra args: $*"
  echo
  "$BIN/llama-bench" "${PLANB_ARGS[@]}" -p 0 -n 64 -d 0,16384 -ub 2048 -r "$reps" -o md "$@"
  echo
  "$BIN/llama-bench" "${PLANB_ARGS[@]}" -p 2048,8192 -n 0 -b 8192 -ub 2048 -r "$reps" -o md "$@"
} > "$md" 2> "${md%.md}.log"

grep -E "^\| (model|deep)" "$md"
echo "results: $md"
