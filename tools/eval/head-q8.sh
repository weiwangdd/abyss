#!/bin/bash
# The output head (BF16, 1.32 GB, read once per decoded token) as Q8_0 against production. retype_tensor.py makes the
# model copy first if it is not there (Q, default ~/models/DeepSeek-V4.1-Flash-1017-GGUF-headq8: only the shard holding
# output.weight is a real file, the others are symlinks); then, in one stop of the production server, cache-gate.sh
# compares the two on greedy corpus slices, the 96k prompt and the append-only session, and defer-ppl.sh pairs their
# decode-path NLL token by token, both with the production binaries (BUILD, default build-m2). Waits for the production
# server to go idle, saves its slot and stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the
# slot). BASE=<results dir of an earlier run> runs only the Q8_0 entries and compares them with that run's production
# ones.
set -u
E=$(cd "$(dirname "$0")" && pwd)
SRC=$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF
Q=${Q:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF-headq8}
B=${BUILD:-$HOME/src/mx-llama.cpp/build-m2/bin}
O=${OUT:-$HOME/abyss-eval}/head-q8-$(date +%m%d-%H%M)
mkdir -p "$O"

if ! find "$Q" -maxdepth 1 -name '*.gguf' -type f 2>/dev/null | grep -q .; then
  echo "### making $Q $(date +%T)"
  nice -n 19 ~/src/comfyui/.venv/bin/python "$E/../convert/retype_tensor.py" "$SRC" "$Q" output.weight Q8_0 || exit 1
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

# with production stopped, neither script stops or restarts it
QM=$Q/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf
if [ -n "${BASE:-}" ]; then
  OUT=$O BUILD=$B CONFIGS="q8head:MODEL=$QM" "$E/cache-gate.sh"
  OUT=$O BUILD=$B CONFIGS="q8head:MODEL=$QM" "$E/defer-ppl.sh"
  cg=$(ls -d "$BASE"/cache-gate-* | tail -1) pp=$(ls -d "$BASE"/defer-ppl-* | tail -1)
  ncg=$(ls -d "$O"/cache-gate-* | tail -1) npp=$(ls -d "$O"/defer-ppl-* | tail -1)
  echo "### corpus slices, prod ($cg) vs q8head"
  python3 -I "$E/greedy_kl.py" compare "$cg/greedy-prod.json" "$ncg/greedy-q8head.json" | sed 's/^/  /'
  echo "### 96k prompt, prod vs q8head: $(python3 -I "$E/ctx_size.py" "$E/corpus.txt" compare "$cg/same-prod.json" "$ncg/same-q8head.json")"
  echo "### decode-path NLL, bf16 (A, $pp) vs q8head (B)"
  python3 -I "$E/nll_compare.py" "$pp/bf16/nll.txt" "$npp/q8head/nll.txt" | sed 's/^/  /'
else
  OUT=$O BUILD=$B CONFIGS="prod q8head:MODEL=$QM" "$E/cache-gate.sh"
  OUT=$O BUILD=$B CONFIGS="bf16 q8head:MODEL=$QM" "$E/defer-ppl.sh"
fi

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### head-q8 done $(date +%T), results in $O"
