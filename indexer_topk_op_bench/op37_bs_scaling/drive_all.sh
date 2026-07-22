#!/bin/bash
# op37: serialized full-matrix driver (one nsys run per cell, GPU0).
# Resume: a cell is skipped when bs_data.csv already has its BS=1024 champion row.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
GPU=${1:-0}
CELLS="flash_32k flash_128k flash_1024k pro_32k pro_128k pro_1024k v32_16k v32_64k v32_256k"
for c in $CELLS; do
  if [ -f "$HERE/bs_data.csv" ] && grep -q "^[0-9]*,[0-9]*,$c,1024,champion" "$HERE/bs_data.csv"; then
    echo "[drive] $c already complete, skip"; continue
  fi
  # quiet-GPU check: no other nsys/python compute on the card
  busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU)
  echo "[drive] $c starting (gpu util pre-check: $busy%)"
  bash "$HERE/run_bs_ab.sh" "$c" "$GPU" > "$HERE/logs_$c.log" 2>&1
  echo "[drive] $c done"
done
echo "[drive] ALL DONE"
