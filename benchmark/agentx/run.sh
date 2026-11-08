#!/bin/bash
# InferenceX v3 AgentX against one server: run.sh URL OUT_DIR NAME. The agentx scenario of SemiAnalysis's harness
# (agentx-harness 1.0.13, an AIPerf fork) on the Claude Code traces InferenceX uses (semianalysisai/cc-traces-weka-062126),
# one session tree at a time, whole traces whose recorded peak exceeds 190,000 tokens dropped (the server holds 196,608),
# AGENTX_DURATION seconds of profiling (default 900, the scenario's minimum), seed AGENTX_SEED (default 42). Artifacts go
# to OUT_DIR/agentx-NAME (profile_export_aiperf.json, profile_export.jsonl, the console tables in client.log).
set -u
H=$(cd "$(dirname "$0")" && pwd)
url=$1 out=$2 name=$3
AGX=${AGX_HOME:-$HOME/abyss-tools/bench/agx}
art=$out/agentx-$name
mkdir -p "$art"
t0=$SECONDS
# llama-server says "exceeds the available context size" on an overflow, which AIPerf would count as a plain error:
# its default list (a JSON array) plus that
AIPERF_AGENTX_CONTEXT_OVERFLOW_SUBSTRINGS=${AIPERF_AGENTX_CONTEXT_OVERFLOW_SUBSTRINGS:-'["context length", "maximum context", "context_length_exceeded", "prompt is too long", "context size"]'} \
  "$AGX/bin/agentx" profile --url "$url" --endpoint /v1/chat/completions --model deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer "${AGENTX_TOKENIZER:-$HOME/abyss-tools/bench/dsv41-tokenizer}" --concurrency 1 \
  --public-dataset "${AGENTX_DATASET:-semianalysis-cc-traces-weka-062126}" --max-context-length 190000 \
  --benchmark-duration "${AGENTX_DURATION:-900}" --random-seed "${AGENTX_SEED:-42}" \
  --output-artifact-dir "$art" ${AGENTX_ARGS:-} > "$art/client.log" 2>&1
echo "  agentx $name: exit $?, $((SECONDS - t0)) s, artifacts in $art"
python3 -I "$H/summary.py" "$art" | sed 's/^/  /'
