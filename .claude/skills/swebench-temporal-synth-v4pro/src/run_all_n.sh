#!/bin/bash
# V4 Pro convenience: synth all 3 beta cfgs × N ∈ V4-Pro production envelope.
# Recall N is POST-compress (V4 cr=4):
#   32K ISL → N=7557    64K ISL → N=14460    100K ISL → ~25100 (no real Pro 100K capture yet)
#
# When BENCH=1, after synthesising all cells run a single nsys session over
# every bundle (via bench_nsys.py) and parse → summary table. This is the
# production-grade timing path (GPU kernel duration only, no cuda.Event
# launch-tail bias). DTYPES env (default "fp32,bf16,fp16") controls the
# heuristic-path dtype sweep; radix baseline is always fp32.
set -e

OUTDIR="${1:-./synth_out_v4pro}"
BS="${BS:-1}"
SEED="${SEED:-42}"
DTYPES="${DTYPES:-fp32,bf16,fp16}"
WARMUP="${WARMUP:-3}"
REPS="${REPS:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTDIR"

# 1. Synth all cells.
for N in 4K 7557 14460 25100; do
  for CFG in beta_shallow beta_moderate beta_deep; do
    python3 "$SCRIPT_DIR/synth_temporal_data.py" \
      --N "$N" --cfg "$CFG" --bs "$BS" --seed "$SEED" \
      --outdir "$OUTDIR"
  done
done

echo
echo "All synth outputs under: $OUTDIR"
ls -1 "$OUTDIR" | grep -E "^beta_" || true

# 2. Optional: nsys-bracketed GVR_<dtype> vs Radix_fp32 + parse.
if [ "${BENCH:-0}" = "1" ]; then
  : "${LIBTH_COMMON:=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/cpp/build/tensorrt_llm/thop/libth_common.so}"
  export LIBTH_COMMON
  : "${TRTLLM_HEURISTIC_NMIN:=4096}"
  : "${TRTLLM_HEURISTIC_BSMAX:=2048}"
  export TRTLLM_HEURISTIC_NMIN TRTLLM_HEURISTIC_BSMAX
  echo
  echo "=== nsys bench across all $(ls -d "$OUTDIR"/beta_* 2>/dev/null | wc -l) cells (dtypes=$DTYPES, warmup=$WARMUP, reps=$REPS) ==="
  ( cd "$OUTDIR" && \
    nsys profile \
      -t cuda,nvtx \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite=true \
      --sample=none --cpuctxsw=none --backtrace=none \
      --cuda-memory-usage=false \
      -o nsys_sweep \
      python3 "$SCRIPT_DIR/bench_nsys.py" \
        --indir . --warmup "$WARMUP" --reps "$REPS" --dtypes "$DTYPES" )
  echo
  echo "=== export nsys → CSV ==="
  ( cd "$OUTDIR" && \
    nsys stats -r nvtx_gpu_proj_trace nsys_sweep.nsys-rep \
      --format csv -o nsys_sweep --force-overwrite=true >/dev/null 2>&1 )
  echo
  echo "=== parse → per-(cfg,N,BS,dtype) R/H summary ==="
  ( cd "$OUTDIR" && \
    python3 "$SCRIPT_DIR/parse_nsys.py" nsys_sweep_nvtx_gpu_proj_trace.csv \
      | tee summary_table.txt )
  echo
  echo "Summary written to: $OUTDIR/summary_table.txt + nsys_speedup_summary.json"
fi
