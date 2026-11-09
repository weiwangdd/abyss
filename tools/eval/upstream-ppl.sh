#!/bin/bash
# Perplexity of production and of upstream llama.cpp on the same weights (the setup of upstream-bench.sh), with
# llama-perplexity, whose source is the same on both sides: CHUNKS chunks of CTX tokens of CORPUS, each scored on its
# second half, once in batches of the whole chunk (the prefill path) and once a token at a time (-ub 1, the decode path,
# with production's expert defer and both MoE caches). For each path in PATHS (default "batch decode") production saves
# its logits as the base and upstream is compared with them: the PPL of both, their ratio, the KL divergence and how
# often the top token is the same. UP_CACHE (default 16000), UP_SHARE (default 0.7), FORK_ARGS and UP_ARGS as in
# upstream-bench.sh. Waits for the production server to go idle, saves its slot and stops it, and restarts it at the end
# (RESTART=0 never, RESTART=1 also when it was not running; ~/abyss-stats/restart.sh restores the slot).
set -u
E=$(cd "$(dirname "$0")" && pwd)
O=${OUT:-$HOME/abyss-eval}/upstream-ppl-$(date +%m%d-%H%M)
FORK_BIN=${FORK_BIN:-$HOME/src/mx-llama.cpp/build-m2/bin}
UP_BIN=${UP_BIN:-$HOME/src/llama.cpp-upstream/build/bin}
FORK_MODEL=${FORK_MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
UP_MODEL=${UP_MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF-upstream/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
CHUNKS=${CHUNKS:-14}
CTX=${CTX:-1024}
CORPUS=${CORPUS:-$E/corpus.txt}
PATHS=${PATHS:-"batch decode"}
mkdir -p "$O"

if nvidia-smi --query-gpu=index,pcie.link.gen.max --format=csv,noheader | awk -F', ' '$2 < 5 { bad = 1 } END { exit bad }'; then :; else
  echo "warning: a GPU link is below PCIe Gen5 (nvidia-smi -q | grep -A6 'GPU Link Info'), speeds will be off" >&2
fi

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
gpus_free() {  # until the stopped process is gone and the driver has handed back its device memory
  for i in $(seq 90); do
    if ! pgrep -x llama-server > /dev/null && ! pgrep -x llama-perplexit > /dev/null &&
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { busy = 1 } END { exit busy }'; then
      return
    fi
    sleep 1
  done
}
gpus_free

FORK_ENV=(GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1
  ABYSS_ENGRAM_PREFETCH=8 ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1
  ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_BIG_SPLIT=1 ABYSS_MOE_BIG_CACHE=1 ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CED_ALIGN=1
  ABYSS_CKPT_THIN_STEP=8192 ABYSS_PLAIN_INPUT=1 ABYSS_DECODE_STATS=100 ABYSS_MOE_ADMIT=0.3 ABYSS_MOE_DEFER=1
  ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_DECODE_PREFETCH=1 ABYSS_DIST_VEC=1 ABYSS_MOE_EVICT=cold ABYSS_MOE_ADMIT_GATE=1
  ABYSS_MOE_SKIP_ZERO=1 ABYSS_HC_FUSE=1 ABYSS_GATHER_FUSE=1 ABYSS_HEAD_SPLIT=0.5 ABYSS_SAMPLE_FAST=1 ABYSS_MMVQ_TAIL=1
  ABYSS_GRAPH_CHUNK=128 ABYSS_ENGRAM_KEEPALIVE_MS=4 ABYSS_GEN_CKPT=256 ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x ABYSS_CKPT_COUNT_START=1)
# production but for 0040: ABYSS_HEAD_SPLIT_ONLY allows one output per batch, llama-perplexity asks for half a chunk,
# so the head stays whole (a single output still splits it, the same logits bit for bit) and the cache at 20500
COMMON=(-ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU,blk\.12\.ffn_(gate|up)_exps\.=CUDA0'
  -f "$CORPUS" -c "$CTX" -b "$CTX" --chunks "$CHUNKS")

for path in $PATHS; do
  ub=$CTX
  [ "$path" = decode ] && ub=1
  base=$O/fork-$path.kld
  echo "### $path path (-ub $ub), production saves the base $(date +%T)"
  t0=$SECONDS
  env "${FORK_ENV[@]}" "$FORK_BIN/llama-perplexity" -m "$FORK_MODEL" "${COMMON[@]}" -ub "$ub" --moe-cache-mib 20500 \
    ${FORK_ARGS:-} --kl-divergence-base "$base" > "$O/fork-$path.log" 2>&1
  echo "  exit $?, $((SECONDS - t0)) s"
  grep -a "Final estimate" "$O/fork-$path.log" | sed 's/^/  production: /'
  gpus_free
  echo "### $path path, upstream against it $(date +%T)"
  t0=$SECONDS
  env GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 "$UP_BIN/llama-perplexity" -m "$UP_MODEL" "${COMMON[@]}" -ub "$ub" \
    --moe-cache-mib "${UP_CACHE:-16000}" --moe-cache-cpu-share "${UP_SHARE:-0.7}" ${UP_ARGS:-} \
    --kl-divergence-base "$base" --kl-divergence > "$O/up-$path.log" 2>&1
  echo "  exit $?, $((SECONDS - t0)) s"
  sed -n '/====== Perplexity statistics/,$p' "$O/up-$path.log" |
    grep -a -E "Mean PPL|Cor\(ln|Mean ln|Mean    KLD|Median  KLD|99.0%   KLD|Maximum KLD|Same top p|RMS Δp" | sed 's/^/  /'
  gpus_free
done

if [ "${RESTART:-}" != 0 ] && { [ -n "$pid" ] || [ "${RESTART:-}" = 1 ]; }; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### upstream-ppl done $(date +%T), logs in $O"
