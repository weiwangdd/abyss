#!/bin/bash
# M1: the fork's upload-on-miss expert cache (--moe-cache-mib) against plan B, on the corpus.
# llama-bench feeds random tokens, which route unlike real text, so speed comes from llama-perplexity:
# ubatch 1 pushes every token through the decode path.
E=$(cd "$(dirname "$0")" && pwd); O=${OUT:-$HOME/abyss-eval}
B=$O/planb-ub1.kld
run() { echo "### $1"; shift; "$@"; }

run "cache 14000 MiB, decode path"  env BASE=$B "$E/kl.sh" cmp planb 1 --moe-cache-mib 14000
run "cache 7000 MiB, decode path"   env BASE=$B "$E/kl.sh" cmp planb 1 --moe-cache-mib 7000
run "cache 14000 MiB, prefill path" env BASE=$B "$E/kl.sh" cmp planb 512 --moe-cache-mib 14000

# llama-perplexity logs no totals or cache hit rates here; the first chunk's time per pass
# (cache still cold) is the speed figure, 512 tokens each
for f in "$O"/kl-base-planb-ub1-*.log $(ls -t "$O"/kl-cmp-planb-ub*.log | head -3); do
  printf "%-42s %s\n" "$(basename "$f")" "$(grep -o "[0-9.]* seconds per pass" "$f")"
done
echo "### done"
