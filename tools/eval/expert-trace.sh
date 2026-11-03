#!/bin/bash
# Routing trace for expert prediction and expert-cache policy studies: decodes up to GEN tokens (default 4000) of a
# coding task in thinking mode at temperature TEMP (default 1.0, as production) through llama-abyss-dump with the
# production settings, saving at every decode step the router input (ffn_norm) and the six chosen experts
# (ffn_moe_topk) of layers 18-39. expert_predict.py analyses the dump. Waits for the production server to go idle, saves
# its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot).
set -u
O=${OUT:-$HOME/abyss-eval}/expert-trace-$(date +%m%d-%H%M)
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
M=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
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

python3 -I -c 'import sys; open(sys.argv[1], "w", encoding="utf-8").write(
    "<｜User｜>你是一个资深后端工程师。请用 Python 3 实现一个单机键值存储服务，要求：\n"
    "1. 支持 GET、SET、DELETE，以及带过期时间的 SET；\n"
    "2. 用 asyncio 实现 TCP 服务，协议自己设计，先说明协议格式；\n"
    "3. 数据定期写入磁盘快照，重启后能恢复；\n"
    "4. 给出完整代码和单元测试，最后说明并发和故障恢复上的设计取舍。<｜Assistant｜><think>")' "$O/prompt.txt"

echo "### decoding up to ${GEN:-4000} tokens $(date +%T)"
env GGML_CUDA_Q8_1_CACHE=0 GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 ABYSS_CED_PREFILL=1 ABYSS_ENGRAM_PREFETCH=8 \
  ABYSS_DSV41_INDEXER=1 ABYSS_DSV41_IDX_CHUNK=512 ABYSS_DSV41_DECODE_GATHER=1 ABYSS_MOE_CPU_MISS=1 ABYSS_MOE_ADMIT=0.3 \
  ABYSS_MOE_BIG_SPLIT=1 ABYSS_MOE_BIG_CACHE=1 ABYSS_PLAIN_INPUT=1 ABYSS_DECODE_STATS=100 \
  ABYSS_DUMP_RE='^(ffn_norm|ffn_moe_topk)-(1[89]|2[0-9]|3[0-9])$' ABYSS_DUMP_DIR="$O" ABYSS_DUMP_GEN="${GEN:-4000}" \
  ABYSS_DUMP_TEMP="${TEMP:-1.0}" ABYSS_DUMP_SEED="${SEED:-1}" \
  "$B/llama-abyss-dump" -m "$M" -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \
  -ot 'blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU' -ot 'blk\.12\.ffn_(gate|up)_exps\.=CUDA0' \
  -c 16384 -b 8192 -ub 2048 --moe-cache-mib 20500 -f "$O/prompt.txt" > "$O/dump.log" 2>&1
echo "### dump exit $? $(date +%T)"
grep -a -E "end of generation|abyss: decode stats" "$O/dump.log" | tail -3
grep -a -c "generated token" "$O/dump.log"
du -sh "$O"
gpus_free

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### all done $(date +%T), dump in $O"
