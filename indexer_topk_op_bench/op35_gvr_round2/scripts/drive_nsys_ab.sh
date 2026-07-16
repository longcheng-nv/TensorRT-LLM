#!/bin/bash
# op35 L2 verdict driver: ONE nsys run per shard, 3 in-process rounds.
G=$1; SH=$2; NW=$3; FLAGS=$4
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/r0val/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/r0val/cutlass450:$PWD/gvrpkg_head:$PWD/variant
env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$G \
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/op35_nsys/ab_s${SH} --force-overwrite true \
  python3 scripts/nsys_ab.py $SH $NW "$FLAGS" all && echo SHARD-$SH-DONE
