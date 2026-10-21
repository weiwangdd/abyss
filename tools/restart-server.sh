#!/bin/bash
# Restart llama-server (plan B) with expert stats, keeping the stats file.
set -e
cd ~/abyss-stats
pid=$(pgrep -x llama-server || true)
if [ -n "$pid" ]; then
  kill -TERM $pid
  for i in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -0 $pid 2>/dev/null && { echo "server $pid did not stop"; exit 1; }
fi
cp expert-stats.json "expert-stats-$(date +%H%M).json"
[ -f server.log ] && mv server.log "server-$(date +%H%M).log"
cd ~/src/mx-llama.cpp
ABYSS_EXPERT_STATS=$HOME/abyss-stats/expert-stats.json GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 nohup ./build-v41/bin/llama-server \
  -m ~/models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf \
  -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot "blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU" \
  -c 131072 -b 2048 -ub 2048 -np 1 \
  --jinja --chat-template-file $HOME/abyss-stats/tmpl/v41.jinja \
  --host 127.0.0.1 --port 8080 > ~/abyss-stats/server.log 2>&1 &
echo "started $!"
