#!/bin/bash
# TraceLab against one server: run.sh URL OUT_DIR NAME. The Claude Code and Codex rounds of the TraceLab corpus
# (uw-syfi, v0.0.1, syfi_coding_trace.jsonl.gz, CC BY 4.0) replayed by the AgentX harness (agentx-harness 1.0.13, an AIPerf
# fork with a TraceLab reader) under the rules of its agentx scenario, which only takes the Weka traces: --unsafe-override
# lets the dataset through and marks the run submission_valid=false, everything else is as for AgentX (agentx/run.sh):
# one session tree at a time, TRACELAB_DURATION seconds of profiling (default 900), seed TRACELAB_SEED (default 42). The
# client takes --max-context-length for Weka traces only, so the trees whose recorded peak exceeds 190,000 tokens (the
# server holds 196,608) are dropped beforehand by filter.py, with the reader's own steps (TRACELAB_DATA, default
# syfi_coding_trace-190k.jsonl.gz: 2,893 of 3,460 traces, 91,017 of 357,161 rounds). Artifacts go to
# OUT_DIR/tracelab-NAME (profile_export_aiperf.json, profile_export.jsonl, client.log).
set -u
H=$(cd "$(dirname "$0")" && pwd)
url=$1 out=$2 name=$3
AGX=${AGX_HOME:-$HOME/abyss-tools/bench/agx}
DATA=${TRACELAB_DATA:-$HOME/abyss-tools/bench/tracelab/syfi_coding_trace-190k.jsonl.gz}
art=$out/tracelab-$name
mkdir -p "$art"
t0=$SECONDS
# the overflow message of llama-server, as for AgentX
AIPERF_AGENTX_CONTEXT_OVERFLOW_SUBSTRINGS=${AIPERF_AGENTX_CONTEXT_OVERFLOW_SUBSTRINGS:-'["context length", "maximum context", "context_length_exceeded", "prompt is too long", "context size"]'} \
  "$AGX/bin/agentx" profile --url "$url" --endpoint /v1/chat/completions --model deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer "${TRACELAB_TOKENIZER:-$HOME/abyss-tools/bench/dsv41-tokenizer}" --concurrency 1 \
  --custom-dataset-type tracelab --input-file "$DATA" --unsafe-override \
  --benchmark-duration "${TRACELAB_DURATION:-900}" --random-seed "${TRACELAB_SEED:-42}" \
  --output-artifact-dir "$art" ${TRACELAB_ARGS:-} > "$art/client.log" 2>&1
echo "  tracelab $name: exit $?, $((SECONDS - t0)) s, artifacts in $art"
python3 -I "$H/../agentx/summary.py" "$art" | sed 's/^/  /'
