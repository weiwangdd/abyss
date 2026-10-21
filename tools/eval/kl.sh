#!/bin/bash
# Compare a configuration's logits against a saved base, on a fixed corpus.
#
#   kl.sh base <tag> <ubatch> [extra llama args...]   save base logits to $OUT/<tag>-ub<ubatch>.kld
#   kl.sh cmp  <tag> <ubatch> [extra llama args...]   KL divergence and top-1 agreement against that base
#   CHUNKS (default 4) sets how many 512-token chunks of the corpus are scored.
#
# ubatch 1 runs every token through the decode path (n_tokens == 1); 512 runs the prefill path.
set -euo pipefail
source "$(dirname "$0")/common.sh"

mode=$1; tag=$2; ub=$3; shift 3
chunks=${CHUNKS:-4}
base=${BASE:-$OUT/$tag-ub$ub.kld}  # BASE overrides it, e.g. to compare the decode path against a prefill-path base
log="$OUT/kl-$mode-$tag-ub$ub-$(date +%m%d-%H%M).log"

case $mode in
  base) kl=(--kl-divergence-base "$base") ;;
  cmp)  kl=(--kl-divergence-base "$base" --kl-divergence) ;;
  *)    echo "mode must be base or cmp" >&2; exit 2 ;;
esac

"$BIN/llama-perplexity" "${PLANB_ARGS[@]}" -f "$EVAL_DIR/corpus.txt" \
  -c 512 --chunks "$chunks" -b 512 -ub "$ub" "${kl[@]}" "$@" > "$log" 2>&1

if [ "$mode" = base ]; then
  grep -E "Final estimate" "$log" || tail -3 "$log"
else
  sed -n '/====== Perplexity statistics/,$p' "$log" | grep -E "Mean PPL|Mean    KLD|Maximum KLD|Same top p|RMS Δp"
fi
echo "log: $log"
