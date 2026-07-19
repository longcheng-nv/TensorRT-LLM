#!/bin/bash
# §8 per-layer rival driver: 3 batches (one per model), each its own nsys-rep.
# Usage: drive_rival_layers.sh <GPU_flash> <GPU_pro> <GPU_v32>   (parallel)
WD=/tmp/gvrlayers
LH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/layers_harness
OUT="$WD/rival_layers_results"
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WD/cutlass450/nvidia_cutlass_dsl/python_packages:$WD/cutlass450${PYTHONPATH:+:$PYTHONPATH}"
GPUS=($1 $2 $3)
MODELS=(flash pro v32)
pids=()
for i in 0 1 2; do
  m=${MODELS[$i]}; g=${GPUS[$i]}
  tag="rival_seqlen_${m}"
  [ -f "$OUT/.done_${tag}" ] && { echo "SKIP done: $tag"; continue; }
  (
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$OUT/nsys_reps/${tag}" --force-overwrite=true \
         python3 "$LH/sweep_rival_layers.py" --model $m --out-root "$OUT"; then
      touch "$OUT/.done_${tag}"
    else
      echo "!! rival batch FAILED: $tag"
    fi
  ) > "$WD/rival_${m}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "rival layers done ($(date -u +%T))"
