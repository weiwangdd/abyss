#!/bin/bash
# The output head of a single output split over both GPUs (ABYSS_HEAD_SPLIT=F, mx-llama.cpp/0034: a copy of the
# head's first F rows on GPU0 computes them while GPU1 computes the rest) against production, in one stop of the
# production server: cache-gate.sh compares the two on greedy corpus slices, the 96k prompt and the append-only session
# (the logits must be the same bit for bit, so the outputs too), both with the server stats of 0033 for the wait on the
# GPU after each decode, and vram-peak.sh measures the device memory with the split, after loading and at its peak
# over a 96k prompt, a fill of the context and a session. F defaults to 0.5, BUILD to build-pf. Waits for the
# production server to go idle, saves its slot and stops it, and restarts it at the end.
set -u
E=$(cd "$(dirname "$0")" && pwd)
F=${F:-0.5}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-pf/bin}
O=${OUT:-$HOME/abyss-eval}/head-split-$(date +%m%d-%H%M)
mkdir -p "$O"

pid=$(pgrep -x llama-server)
if [ -n "$pid" ]; then
  echo "### waiting for the production server to go idle $(date +%T)"
  until curl -s -m 5 http://127.0.0.1:8080/slots | grep -q '"is_processing":false'; do sleep 5; done
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=save" -H "Content-Type: application/json" \
    -d '{"filename": "slot0.bin"}'
  echo
  kill -TERM "$pid"
  for i in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  # its cleanup can hang past a minute; a second signal ends it at once, and the slot is saved already
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    for i in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  fi
  echo "### production stopped $(date +%T)"
fi

# with production stopped, neither script stops or restarts it
OUT=$O BUILD=$B CONFIGS="off:ABYSS_SERVER_STATS=1000,ABYSS_HEAD_SPLIT=0 split:ABYSS_SERVER_STATS=1000,ABYSS_HEAD_SPLIT=$F" "$E/cache-gate.sh"
for log in "$O"/cache-gate-*/server-*.log; do
  echo "### $(basename "$log" .log): $(grep -a -m1 "ABYSS_HEAD_SPLIT" "$log" | sed 's/^.*abyss: //')"
  grep -a "abyss: server stats" "$log" | grep "request end" | sed 's/^.*= //' | awk -F' [+] ' '
    { for (i = 1; i <= NF; i++) { k = split($i, a, " "); s[i] += a[k]; name[i] = $i; sub(/ [0-9.]+$/, "", name[i]) } m = NF; n++ }
    END { if (n) { printf "  per token over %d requests:", n; for (i = 1; i <= m; i++) printf " %s %.3f", name[i], s[i] / n; print "" } }'
done
OUT=$O BUILD=$B EXTRA_ENV="ABYSS_HEAD_SPLIT=$F" "$E/vram-peak.sh"

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### head-split done $(date +%T), results in $O"
