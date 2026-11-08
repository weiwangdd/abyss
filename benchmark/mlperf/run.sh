#!/bin/bash
# MLPerf Agentic Inference against one server: run.sh URL OUT_DIR NAME. The conversations are MLPERF_IDS (default one
# coding trajectory and two Workato conversations, see README.md), cut from the official dataset by subset.py; the
# report goes to OUT_DIR/mlperf-NAME (report.txt, performance/result_summary.json, events.jsonl, scores.json).
# MLPERF_MAX_NEW_TOKENS caps each turn's output (default the official 8192), for trying the harness only.
set -u
H=$(cd "$(dirname "$0")" && pwd)
url=$1 out=$2 name=$3
EP=${MLPERF_HOME:-$HOME/abyss-tools/bench/endpoints}
DATA=${MLPERF_DATA:-$EP/examples/10_Agentic_Inference/datasets/mlperf_agentic_inference_dataset.jsonl}
IDS=${MLPERF_IDS:-"superjson-error-stack-serialization sim_000203 sim_000424"}
report=$out/mlperf-$name
mkdir -p "$report"

python3 -I "$H/subset.py" write "$DATA" "$report/subset.jsonl" $IDS
t0=$SECONDS
(
  cd "$EP" &&
  MLPERF_TOKENIZER=${MLPERF_TOKENIZER:-$HOME/abyss-tools/bench/dsv41-tokenizer} MLPERF_DATASET=$report/subset.jsonl \
  MLPERF_TRAJECTORIES=$(echo $IDS | wc -w) MLPERF_ENDPOINT=$url MLPERF_REPORT_DIR=$report \
  .venv/bin/inference-endpoint benchmark from-config --config "$H/dsv41_abyss.yaml"
) > "$report/client.log" 2>&1
echo "  mlperf $name: exit $?, $((SECONDS - t0)) s, report in $report"
python3 -I "$H/summary.py" "$report" | sed 's/^/  /'
