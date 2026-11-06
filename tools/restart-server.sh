#!/bin/bash
# Restart llama-server (plan B) with expert stats, keeping the stats file. Slot 0 is saved to ~/abyss-stats/slots
# before the stop (with its context checkpoints) and restored after the start, so the next request continues the
# conversation instead of prefilling it again; NO_RESTORE=1 skips the restore, e.g. when the cache layout or the
# template or the weights changed. Like the official encoding, reasoning of earlier turns is dropped when there are no
# tools, and the default sampling is what the DeepSeek-V4.1-Flash-1017 model card recommends, temperature 1.0 and
# top-p 1.0; requests can override it. The experts the CPU computes for a token are deferred to the next layer
# (ABYSS_MOE_DEFER=1, mx-llama.cpp/0025); restart-nodefer.sh is the fallback without. The hyper-connection mixes of a
# token run as one split-K product with their norm folded in (ABYSS_HC_KSPLIT=1, 0027); restart-noksplit.sh is the
# fallback without. The expert stats read each layer's selection after the graph (0026); NEW_STATS=1 starts them
# from zero, the old file kept aside. A decode step's Engram rows are read ahead (ABYSS_ENGRAM_DECODE_PREFETCH=1, 0028)
# and the dist sampler is vectorized (ABYSS_DIST_VEC=1, 0029); neither changes the outputs, and leaving them out is the
# fallback. The MoE cache evicts its coldest expert by decayed heat and admits a miss only if hotter than that one, and
# the experts the CPU computes for a token read no zero slot (ABYSS_MOE_EVICT=cold, ABYSS_MOE_ADMIT_GATE=1,
# ABYSS_MOE_SKIP_ZERO=1, 0030); leaving them out is the fallback. A token's hyper-connection gates and combs, and the
# keys and mask of its attention with the top-k compressed rows, run in fewer kernels (ABYSS_HC_FUSE=1,
# ABYSS_GATHER_FUSE=1, 0032), with the same results; leaving them out is the fallback. A single output's head runs on
# both GPUs at once, the first half of its rows from a copy on GPU0 (ABYSS_HEAD_SPLIT=0.5, 0034), with the same
# results; GPU0 then has 631 MiB less free, so the desktop must not use GPU0, and leaving it out is the fallback. A
# request whose sampling reduces to dist or greedy over the logits, as the defaults do, samples from the logits without
# building the candidates (ABYSS_SAMPLE_FAST=1, 0035), the same tokens; leaving it out is the fallback. The
# matrix-vector products whose K loop would leave many threads idle take several rows per block (ABYSS_MMVQ_TAIL=1,
# 0036), and the q8_1 cache, now with a pool of its own, quantizes an activation once for the products that share it
# (GGML_CUDA_Q8_1_CACHE=1, ABYSS_Q8_1_CACHE_LEG=1, 0036; GPU0 about 110 MiB more at its peak), the same results;
# ABYSS_MMVQ_TAIL=0 and GGML_CUDA_Q8_1_CACHE=0 are the fallback (without the pool of its own the cache must stay off:
# it frees out of order, and the VMM pool asserts).
# HEAD_Q8=1 runs the copy of the model whose output head is Q8_0 instead of BF16 (tools/convert/retype_tensor.py: about
# 2% faster per request, top-20 KL about 5e-4, the decode-path PPL the same); off by default.
set -e
model=$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF
if [ -n "${HEAD_Q8:-}" ]; then
  model=$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF-headq8
fi
cd ~/abyss-stats
mkdir -p slots
pids=$(pgrep -x llama-server || true)
if [ -n "$pids" ]; then
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=save" -H "Content-Type: application/json" \
    -d '{"filename": "slot0.bin"}' || echo "slot save failed"
  echo
  # every server goes, a test server left behind too; the cleanup can hang past a minute, a second signal ends it at
  # once (the slot is saved already), and a server that still holds on is killed
  for pid in $pids; do
    kill -TERM $pid 2>/dev/null || continue
    for i in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
    if kill -0 $pid 2>/dev/null; then
      echo "server $pid still stopping, signalling again"
      kill -TERM $pid 2>/dev/null || true
      for i in $(seq 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done
    fi
    if kill -0 $pid 2>/dev/null; then
      echo "server $pid killed"
      kill -KILL $pid 2>/dev/null || true
      sleep 2
    fi
  done
  # the driver hands back the device memory a few seconds after the process is gone
  for i in $(seq 60); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }' && break
    sleep 1
  done
fi
if [ -f expert-stats.json ]; then
  if [ -n "$NEW_STATS" ]; then
    mv expert-stats.json "expert-stats-$(date +%m%d-%H%M)-before-new.json"
  else
    cp expert-stats.json "expert-stats-$(date +%H%M).json"
  fi
fi
[ -f server.log ] && mv server.log "server-$(date +%H%M).log"
cd ~/src/mx-llama.cpp
ABYSS_EXPERT_STATS=$HOME/abyss-stats/expert-stats.json GGML_CUDA_Q8_1_CACHE=1 ABYSS_Q8_1_CACHE_LEG=1 GGML_CUDA_MMQ_PREC=q8 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 ABYSS_DSV41_INDEXER=1 \
  ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1 ABYSS_DECODE_STATS=100 GOMP_SPINCOUNT=100000000 \
  ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_BIG_CACHE=1 ABYSS_MOE_DEFER=1 ABYSS_HC_KSPLIT=1 \
  ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1 ABYSS_MOE_SKIP_ZERO=1 \
  ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1 ABYSS_MMVQ_TAIL=1 \
  nohup ./build-m2/bin/llama-server \
  -m $model/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf \
  -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot "blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU" -ot "blk\.12\.ffn_(gate|up)_exps\.=CUDA0" \
  -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib 20500 \
  --checkpoint-min-step 32768 --slot-save-path $HOME/abyss-stats/slots \
  --jinja --chat-template-file $HOME/abyss-stats/tmpl/v41.jinja --no-reasoning-preserve \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0 \
  --host 127.0.0.1 --port 8080 > ~/abyss-stats/server.log 2>&1 &
pid=$!
echo "started $pid"
if [ -z "$NO_RESTORE" ] && [ -f ~/abyss-stats/slots/slot0.bin ]; then
  until grep -q "listening on" ~/abyss-stats/server.log; do
    kill -0 $pid 2>/dev/null || { echo "server died, see ~/abyss-stats/server.log"; exit 1; }
    sleep 3
  done
  curl -s -m 120 -X POST "http://127.0.0.1:8080/slots/0?action=restore" -H "Content-Type: application/json" \
    -d '{"filename": "slot0.bin"}' || echo "slot restore failed"
  echo
fi
