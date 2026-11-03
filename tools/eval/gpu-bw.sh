#!/bin/bash
# Link and bandwidth check of the two GPUs, e.g. after a cold boot (GPU1 has come up at Gen4 or Gen3 several times):
# the PCIe link each GPU trained to, nvbandwidth's copy-engine bandwidths host -> each GPU, each GPU -> host, host ->
# both GPUs at once and between the GPUs, and the host memory read bandwidth (readbw), which bounds the copies from
# host memory. Healthy: 28.9 / 28.7 GB/s per GPU, about 50 GB/s for both at once, 27.7 GB/s one way between the GPUs,
# about 54 GB/s memory read at 5200 MT/s. If the production server runs, waits for it to go idle, saves its slot and
# stops it, and restarts it at the end (~/abyss-stats/restart.sh restores the slot); a stopped server stays stopped.
set -u
NVBW=${NVBW:-$HOME/bench/nvbandwidth/build/nvbandwidth}
O=${OUT:-$HOME/abyss-eval}/gpu-bw-$(date +%m%d-%H%M).txt

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

{
  echo "### $(date '+%F %T'), BIOS $(cat /sys/class/dmi/id/bios_version 2>/dev/null), up $(uptime -p)"
  echo "### PCIe links (index, trained generation, width)"
  nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.current --format=csv,noheader
  echo "### nvbandwidth, copy engines, 512 MiB buffers"
  "$NVBW" -b 512 -t host_to_device_memcpy_ce device_to_host_memcpy_ce host_to_all_memcpy_ce \
    device_to_device_memcpy_read_ce device_to_device_bidirectional_memcpy_read_ce
  echo "### host memory read (readbw, 8 GiB)"
  "$HOME/bench/readbw"
} 2>&1 | tee "$O"

if [ -n "$pid" ]; then
  echo "### restarting production $(date +%T)"
  ~/abyss-stats/restart.sh 2>&1 | tail -3
  curl -s -m 10 http://127.0.0.1:8080/slots | grep -q '"id":0' && echo "### production answers on 8080" ||
    echo "### WARNING: production does not answer on 8080"
fi
echo "### saved $O"
