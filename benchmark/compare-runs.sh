#!/bin/bash
# Two runs of run.sh against each other, for one server: compare-runs.sh OLD_DIR NEW_DIR [SERVER] (default fork) prints
# compare.py's tables with the two runs as its two columns (new / old where both have the benchmark), then, for AgentX
# and TraceLab, pairs.py over the requests both runs sent (a faster server gets further in the fixed window, so the
# percentiles alone take in different requests).
set -u
B=$(cd "$(dirname "$0")" && pwd)
old=$1 new=$2 server=${3:-fork}
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
for bench in mlperf agentx tracelab; do
  [ -d "$old/$bench-$server" ] && ln -s "$old/$bench-$server" "$t/$bench-old"
  [ -d "$new/$bench-$server" ] && ln -s "$new/$bench-$server" "$t/$bench-new"
done
echo "new: $new"
echo "old: $old"
python3 -I "$B/compare.py" "$t" new old
for bench in agentx tracelab; do
  if [ -d "$t/$bench-old" ] && [ -d "$t/$bench-new" ]; then
    echo
    echo "### $bench, the same requests"
    echo
    python3 -I "$B/agentx/pairs.py" "$t/$bench-new" "$t/$bench-old"
  fi
done
