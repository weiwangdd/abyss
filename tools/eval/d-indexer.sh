#!/bin/bash
# D: the V4.1 sparse attention indexer (ABYSS_DSV41_INDEXER=1) against the dense attention over all compressed rows.
#  - up to 512 visible compressed positions the top-k keeps them all, so the indexer must match dense exactly
#  - beyond that it selects; the model trained with the sparse selection, so its perplexity should not be worse
#  - past 2048 candidate blocks (16k tokens on the ratio 1 tier) the candidate step joins in
#  - with the CED-aware prefill on top, the subset maps must not change the result
# Stops at the first exactness check that fails (the saved logits round KLD to about 1e-6, so 1e-5 is the bar).
# SKIP_EXACT=1 skips the 512-token checks.
E=$(cd "$(dirname "$0")" && pwd); O=${OUT:-$HOME/abyss-eval}
export BIN=${BIN:-$HOME/src/mx-llama.cpp/build-idx/bin}
run() { echo "### $1"; shift; "$@"; }
kld() { sed -n '/====== Perplexity statistics/,$p' "$(ls -t "$O"/kl-cmp-*.log | head -1)" | awk '/Mean    KLD/ { print $3 }'; }

for n in 2 3 5; do
  for i in $(seq $n); do cat "$E/corpus.txt"; done > "$O/corpus$n.txt"
done

if [ -z "${SKIP_EXACT:-}" ]; then
run "512, prefill path: base, indexer off" env CTX=512 CHUNKS=4 "$E/kl.sh" base idx-c512 512
run "512, prefill path: indexer on"        env CTX=512 CHUNKS=4 ABYSS_DSV41_INDEXER=1 "$E/kl.sh" cmp idx-c512 512
echo "KLD $(kld)"
awk -v k="$(kld)" 'BEGIN { exit !(k < 1e-5) }' || { echo "### not exact at 512, stopping"; exit 1; }

run "512, decode path: base, indexer off" env CTX=512 CHUNKS=2 "$E/kl.sh" base idx-c512 1
run "512, decode path: indexer on"        env CTX=512 CHUNKS=2 ABYSS_DSV41_INDEXER=1 "$E/kl.sh" cmp idx-c512 1
echo "KLD $(kld)"
awk -v k="$(kld)" 'BEGIN { exit !(k < 1e-5) }' || { echo "### not exact at 512 (decode), stopping"; exit 1; }
fi

run "8k: base, indexer off"  env CTX=8192 CHUNKS=1 CORPUS="$O/corpus2.txt" "$E/kl.sh" base idx-c8192 2048
run "8k: indexer on"         env CTX=8192 CHUNKS=1 CORPUS="$O/corpus2.txt" ABYSS_DSV41_INDEXER=1 "$E/kl.sh" cmp idx-c8192 2048
run "16k: base, indexer off" env CTX=16384 CHUNKS=1 CORPUS="$O/corpus3.txt" "$E/kl.sh" base idx-c16384 2048
run "16k: indexer on"        env CTX=16384 CHUNKS=1 CORPUS="$O/corpus3.txt" ABYSS_DSV41_INDEXER=1 "$E/kl.sh" cmp idx-c16384 2048
run "32k: base, indexer off" env CTX=32768 CHUNKS=1 CORPUS="$O/corpus5.txt" "$E/kl.sh" base idx-c32768 2048
run "32k: indexer on"        env CTX=32768 CHUNKS=1 CORPUS="$O/corpus5.txt" ABYSS_DSV41_INDEXER=1 "$E/kl.sh" cmp idx-c32768 2048

run "16k: base, indexer on, CED off" env CTX=16384 CHUNKS=1 CORPUS="$O/corpus3.txt" ABYSS_DSV41_INDEXER=1 "$E/kl.sh" base idx-on-c16384 2048
run "16k: indexer on, CED on"        env CTX=16384 CHUNKS=1 CORPUS="$O/corpus3.txt" ABYSS_DSV41_INDEXER=1 ABYSS_CED_PREFILL=1 "$E/kl.sh" cmp idx-on-c16384 2048

for f in $(ls -tr "$O"/kl-*-idx*.log | tail -12); do
  printf "%-46s %s\n" "$(basename "$f")" "$(grep -o "[0-9.]* seconds per pass" "$f")"
done
echo "### done"
