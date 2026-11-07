#!/bin/bash
# The two measurements before deciding on tensor parallelism of layers 0-21, in one stop of the production server: the
# latency of a two-GPU all-reduce of a decode step's sizes with the GPUs otherwise idle (p2p_allreduce.cu), and an nsys
# profile of a decode turn with the production settings (prod-decode-profile.sh) read with nsys_tp.py, once with the
# extra time of a 20 KB all-reduce over the local floor in a CUDA graph of both GPUs (the lower bound) and once issued
# step by step from the host (closer to ggml, which splits a graph where it crosses GPUs and launches each part), the
# cheapest of the ways that sum correctly. Waits for the production server to go idle, saves its slot and stops it, and
# restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/tp-measure-$(date +%m%d-%H%M)
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
gpus_free() {  # until the stopped server is gone and the driver has handed back its device memory
  for i in $(seq 90); do
    if ! pgrep -x llama-server > /dev/null &&
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }'; then
      return
    fi
    sleep 1
  done
}
gpus_free

echo "### all-reduce latency (p2p_allreduce.cu) $(date +%T)"
/usr/local/cuda/bin/nvcc -O2 -arch=sm_120 -o "$O/p2p_allreduce" "$E/p2p_allreduce.cu" &&
  "$O/p2p_allreduce" 2>&1 | tee "$O/p2p.txt"
# the extra microseconds of a 20 KB all-reduce over the local floor, graph and eager, of the modes that sum correctly
read -r us_graph us_eager < <(python3 -I - "$O/p2p.txt" <<'PY'
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.split()[:1] == ["20480"]]
floor = {r[1]: (float(r[2]), float(r[3])) for r in rows if r[1] == "local"}["local"]
ok = [(float(r[3]) - floor[1], float(r[2]) - floor[0]) for r in rows if r[1] != "local" and r[4] == "0" and float(r[3]) > 0]
g, e = min(x[0] for x in ok), min(x[1] for x in ok)
print(f"{max(g, 0):.1f} {max(e, 0):.1f}")
PY
)
echo "### a 20 KB all-reduce costs ${us_graph:-?} us in a graph, ${us_eager:-?} us issued step by step"
gpus_free

echo "### decode turn under nsys (prod-decode-profile.sh) $(date +%T)"
OUT=$O "$E/prod-decode-profile.sh" > "$O/profile.txt" 2>&1
S=$(ls -d "$O"/prod-profile-*/decode-modes.sqlite 2>/dev/null | tail -1)
if [ -n "$S" ]; then
  echo "### layers 0-21 with tensor parallelism (nsys_tp.py), all-reduces as in a graph"
  python3 -I "$E/nsys_tp.py" "$S" "${us_graph:-20}" | tee "$O/tp-graph.txt"
  echo "### the same, all-reduces issued step by step"
  python3 -I "$E/nsys_tp.py" "$S" "${us_eager:-20}" | tee "$O/tp-eager.txt"
else
  echo "### no profile, see $O/profile.txt"
fi
gpus_free

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### tp-measure done $(date +%T), logs in $O"
