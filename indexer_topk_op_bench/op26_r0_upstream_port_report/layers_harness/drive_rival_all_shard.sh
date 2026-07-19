#!/bin/bash
# All-layer external-arm sweep shard driver (radix/sglang_v2/flashinfer,
# BS=1 fp32, every captured GVR-active layer).
# Usage: drive_rival_all_shard.sh <W> <GPU> <NW>
W=$1; GPU=$2; NW=$3
WD=/tmp/gvrlayers
LH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/layers_harness
OUT="$WD/rival_all_results"
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WD/cutlass450/nvidia_cutlass_dsl/python_packages:$WD/cutlass450${PYTHONPATH:+:$PYTHONPATH}"

BATCHES=()
for i in 0 1 2 3 4 5 6 7 8; do
  for m in flash pro v32; do
    isls_flash=(4k 8k 16k 32k 64k 128k 256k 512k 1024k)
    isls_v32=(4k 8k 16k 32k 64k 128k 256k)
    if [ "$m" = "v32" ]; then
      [ $i -lt 7 ] && BATCHES+=("$m ${isls_v32[$i]}")
    else
      BATCHES+=("$m ${isls_flash[$i]}")
    fi
  done
done
i=0
for batch in "${BATCHES[@]}"; do
  if [ $((i % NW)) -eq $W ]; then
    read -r m isl <<< "$batch"
    tag="rival_seqlen_${m}_${isl}"
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP done: $tag"; i=$((i+1)); continue; fi
    echo "=== rival-all batch [$W]: $tag ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" --force-overwrite=true \
         python3 "$LH/sweep_rival_layers.py" --model $m --isl $isl --out-root "$OUT"; then
      touch "$done_m"
    else
      echo "!! batch FAILED: $tag"
    fi
  fi
  i=$((i+1))
done
echo "shard $W done ($(date -u +%T))"
