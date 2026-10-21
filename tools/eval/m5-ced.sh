#!/bin/bash
# M5: CED-aware prefill (ABYSS_CED_PREFILL=1), correctness and speed on llama-perplexity.
# llama-perplexity scores only the second half of each chunk, so with long chunks the start of a chunk is out of
# reach of every output and runs no decoder FFN. The change only touches decoder layers, so its KL against the
# full run must stay at the decoder-only floor (about 0.0005, flag above 0.001).
E=$(cd "$(dirname "$0")" && pwd); O=${OUT:-$HOME/abyss-eval}
export BIN=${BIN:-$HOME/src/mx-llama.cpp/build-ced/bin}
run() { echo "### $1"; shift; "$@"; }

# llama-perplexity wants twice the chunk length in tokens; the corpus is about 16k tokens
cat "$E/corpus.txt" "$E/corpus.txt" > "$O/corpus2.txt"
cat "$E/corpus.txt" "$E/corpus.txt" "$E/corpus.txt" > "$O/corpus3.txt"

run "8k chunk: base, CED off"   env CTX=8192 CHUNKS=1 CORPUS="$O/corpus2.txt" "$E/kl.sh" base ced-c8192 2048
run "8k chunk: CED on"          env CTX=8192 CHUNKS=1 CORPUS="$O/corpus2.txt" ABYSS_CED_PREFILL=1 "$E/kl.sh" cmp ced-c8192 2048
run "16k chunk: base, CED off"  env CTX=16384 CHUNKS=1 CORPUS="$O/corpus3.txt" "$E/kl.sh" base ced-c16384 2048
run "16k chunk: CED on"         env CTX=16384 CHUNKS=1 CORPUS="$O/corpus3.txt" ABYSS_CED_PREFILL=1 "$E/kl.sh" cmp ced-c16384 2048

for f in $(ls -tr "$O"/kl-*-ced-c*.log | tail -4); do
  printf "%-44s %s\n" "$(basename "$f")" "$(grep -o "[0-9.]* seconds per pass" "$f")"
done
echo "### done"
