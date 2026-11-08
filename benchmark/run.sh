#!/bin/bash
# The standard benchmarks (README.md) against production and upstream llama.cpp, in one stop of production: for each
# server of SERVERS (default "fork up", see servers.sh) and each benchmark of BENCHES (default "mlperf agentx"), a fresh
# server on $PORT, the benchmark's run.sh against it, the server stopped; then the comparison (compare.py) and production
# restarted (RESTART=0 never, RESTART=1 also when it was not running). Output in $OUT/bench-<date> on blackwell.
set -u
B=$(cd "$(dirname "$0")" && pwd)
source "$B/servers.sh"
O=${OUT:-$HOME/abyss-eval}/bench-$(date +%m%d-%H%M)
SERVERS=${SERVERS:-"fork up"}
BENCHES=${BENCHES:-"mlperf agentx"}
mkdir -p "$O"

if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi
case " $SERVERS " in *" fork "*|*" up "*) bench_stop_production ;; esac

for bench in $BENCHES; do
  for server in $SERVERS; do
    echo "### $bench on $server $(date +%T)"
    if bench_start "$server" "$bench-$server" "$O"; then
      "$B/$bench/run.sh" "http://127.0.0.1:$PORT" "$O" "$server"
      bench_stop_server
    fi
  done
done

python3 -I "$B/compare.py" "$O" $SERVERS | tee "$O/compare.md"

if [ "${RESTART:-}" != 0 ] && { [ "$PROD_WAS_RUNNING" = 1 ] || [ "${RESTART:-}" = 1 ]; }; then
  bench_restart_production
fi
echo "### benchmarks done $(date +%T), logs in $O"
