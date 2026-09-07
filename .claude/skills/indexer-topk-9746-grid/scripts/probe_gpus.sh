#!/bin/bash
# List GPUs that can actually run CUDA work. nvidia-smi is NOT sufficient:
# a GPU held by a process in another namespace shows 0 MiB / 0% yet raises
# cudaErrorDevicesUnavailable on first allocation. Prints "GPU<i>: OK|BUSY"
# and, on the last line, a space-separated list of usable indices to feed
# straight into launch_<tag>_host.sh.
set -u
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
USABLE=()
for g in $(seq 0 $((N - 1))); do
  if CUDA_VISIBLE_DEVICES=$g timeout 120 python3 -c \
      "import torch; torch.zeros(8, device='cuda'); torch.cuda.synchronize()" \
      > /dev/null 2>&1; then
    echo "GPU$g: OK"; USABLE+=("$g")
  else
    echo "GPU$g: BUSY (unusable — exclude from the shard list)"
  fi
done
echo "${USABLE[@]}"
