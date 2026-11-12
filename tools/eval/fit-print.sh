#!/bin/bash
# What a llama.cpp build would allocate for the model on each device, without allocating it: llama-fit-params
# --fit-print loads only the metadata and simulates the buffers, in under a second per entry. Prints, per entry of
# CONFIGS, the lines "device model context compute" in MiB (context is the KV cache and the MoE cache).
#
# CONFIGS entries are "name[:VAR=VALUE,...]": CACHE=<MiB> (--moe-cache-mib, default 16000), TS=<split> (-ts, default
# 12/29), OT=<overrides> (-ot, default production's two, ';' standing for ','), CTX=<n> (default 196608), UB=<n>
# (default 2048); other VAR=VALUE go to the environment, e.g. ABYSS_DSV41_CHUNK=1024. TRACE=1 also prints what each
# compute buffer holds at its peak (GGML_ALLOC_TRACE=1 and alloc_peak.py, TOP entries, default 10).
#
# The backends are initialized, about 0.5 GiB on each visible GPU while an entry runs: with production up, run it with
# CUDA_VISIBLE_DEVICES=1 only, which simulates one device holding every layer (its compute buffer is close to GPU1's,
# the device of the host-resident layers); for the split over both GPUs stop production first.
set -u
E=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HOME/src/llama.cpp-abyss/build/bin}
MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash-1017-GGUF-upstream/DeepSeek-V4.1-Flash-1017-MXFP4-00001-of-00012.gguf}
CONFIGS=${CONFIGS:-"plain chunk:ABYSS_DSV41_CHUNK=1024"}
n_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)
[ -n "${CUDA_VISIBLE_DEVICES+x}" ] && n_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

for c in $CONFIGS; do
  IFS=: read -r name settings <<< "$c"
  extra_env=()
  [ -n "$settings" ] && IFS=, read -r -a extra_env <<< "$settings"
  cache=16000 ts=12/29 ctx=196608 ub=2048
  ot='blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU,blk\.12\.ffn_(gate|up)_exps\.=CUDA0'
  for k in "${!extra_env[@]}"; do
    case ${extra_env[$k]} in
      CACHE=*) cache=${extra_env[$k]#CACHE=}; unset 'extra_env[k]' ;;
      TS=*) ts=${extra_env[$k]#TS=}; unset 'extra_env[k]' ;;
      OT=*) ot=${extra_env[$k]#OT=}; ot=${ot//;/,}; unset 'extra_env[k]' ;;
      CTX=*) ctx=${extra_env[$k]#CTX=}; unset 'extra_env[k]' ;;
      UB=*) ub=${extra_env[$k]#UB=}; unset 'extra_env[k]' ;;
    esac
  done
  split=(-ts "$ts")
  [ "$n_gpus" -lt 2 ] && split=()
  log=$(mktemp)
  out=$(env GGML_CUDA_MMQ_PREC=q8 ${TRACE:+GGML_ALLOC_TRACE=1} "${extra_env[@]}" \
    "$BIN/llama-fit-params" -m "$MODEL" -ngl 99 -sm layer "${split[@]}" -fa 1 -c "$ctx" -b 65536 -ub "$ub" -np 1 \
    -ot "$ot" --moe-cache-mib "$cache" --moe-cache-cpu-share 0.7 --fit-print on 2> "$log")
  echo "### $name: cache $cache${settings:+, $settings}"
  echo "$out" | sed 's/^/  /'
  if [ -n "${TRACE:-}" ]; then
    python3 -I "$E/alloc_peak.py" "$log" "${TOP:-10}" | sed -n '/^buffer/,$p' | grep -v "^$" | sed 's/^/    /'
  fi
  [ -z "$out" ] && tail -5 "$log" | sed 's/^/  /'
  rm -f "$log"
done
