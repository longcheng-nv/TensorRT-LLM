#!/usr/bin/env bash
# Run the full GVR-vs-Radix Top-K experiment on SWE-Bench-64K and emit
# an English REPORT.md. Designed to be invoked from the skill directly.
#
# Required env / args:
#   TRTLLM_REPO            path to a built TensorRT-LLM checkout
#                          (must contain tensorrt_llm/libs/libth_common.so)
#   SWE_BENCH_PATH         directory containing Layer_{N}_pd.npy files
#                          (default: tllm_toolbox SWE-Bench-64K decode logits)
#
# CLI flags (all optional):
#   --outdir DIR           where all artifacts are written. Default:
#                          /tmp/gvr_vs_radix_swebench_<timestamp>
#   --bs_list  CSV         BS sweep, default 1,2,4,8,16,32,64,128,256,512
#   --row_stride N         BS=1 row stride, default 10 (~203 rows / layer)
#   --warmup N             warmup iterations per cell, default 3
#   --repeats N            timed reps per cell, default 5
#   --variants CSV         variant tags to bench, default "radix,gvr_K2048_fp32"
#                          (apples-to-apples GVR vs Radix at fp32 K=2048)
#   --layers "L1 L2 ..."   space-separated layer IDs, default 9-layer set
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

OUTDIR=""
BS_LIST="1,2,4,8,16,32,64,128,256,512"
ROW_STRIDE=10
WARMUP=3
REPEATS=5
VARIANTS="radix,gvr_K2048_fp32"
LAYERS="0 1 20 21 22 40 41 42 60"
GVR_BACKEND="production"
SKIP_BS1=0
SKIP_BSSCALING=0

while [ $# -gt 0 ]; do
  case "$1" in
    --outdir)         OUTDIR="$2"; shift 2 ;;
    --bs_list)        BS_LIST="$2"; shift 2 ;;
    --row_stride)     ROW_STRIDE="$2"; shift 2 ;;
    --warmup)         WARMUP="$2"; shift 2 ;;
    --repeats)        REPEATS="$2"; shift 2 ;;
    --variants)       VARIANTS="$2"; shift 2 ;;
    --layers)         LAYERS="$2"; shift 2 ;;
    --gvr_backend)    GVR_BACKEND="$2"; shift 2 ;;
    --skip-bs1)       SKIP_BS1=1; shift ;;
    --skip-bs-scaling) SKIP_BSSCALING=1; shift ;;
    -h|--help)
        cat <<EOF
Usage: $0 [--outdir DIR] [--bs_list CSV] [--row_stride N] [--warmup N]
          [--repeats N] [--variants CSV] [--layers "L1 L2 ..."]
          [--gvr_backend production|standalone]
          [--skip-bs1] [--skip-bs-scaling]

  --gvr_backend production  (default) GVR via torch.ops.trtllm.indexer_topk_decode
                            (preIdxOffset=+1 baked into the kernel).
  --gvr_backend standalone  GVR via auto_optimization_v1/topk_cuda.py JIT
                            (preIdxOffset=0 kernel + Python +1 shift -> net
                            semantics match production). BS>1 loops in Python.
EOF
        exit 0 ;;
    *) echo "[run_full.sh] unknown flag: $1" >&2; exit 2 ;;
  esac
done

case "$GVR_BACKEND" in
  production|standalone) ;;
  *) echo "[run_full.sh] --gvr_backend must be 'production' or 'standalone' (got '$GVR_BACKEND')" >&2; exit 2 ;;
esac

if [ -z "$OUTDIR" ]; then
  OUTDIR="/tmp/gvr_vs_radix_swebench_$(date -u +%Y%m%dT%H%M%S)"
fi
mkdir -p "$OUTDIR"

echo "[run_full.sh] OUTDIR     = $OUTDIR"
echo "[run_full.sh] LAYERS     = $LAYERS"
echo "[run_full.sh] BS_LIST    = $BS_LIST"
echo "[run_full.sh] VARIANTS   = $VARIANTS"
echo "[run_full.sh] GVR_BACKEND= $GVR_BACKEND"

command -v nsys >/dev/null 2>&1 || {
  echo "[run_full.sh] ERROR: nsys not on PATH. Install Nsight Systems." >&2
  exit 3
}
command -v python3 >/dev/null 2>&1 || {
  echo "[run_full.sh] ERROR: python3 not on PATH." >&2
  exit 3
}

# ---------------------------------------------------------------------------
# Step 1: preIdxOffset audit
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: preIdxOffset audit ($GVR_BACKEND) ==="
PRE_CHECK="$OUTDIR/preidx_offset_check.txt"
python3 "$SCRIPT_DIR/verify_preidx_offset.py" \
    --gvr_backend "$GVR_BACKEND" \
    --out "$PRE_CHECK" || {
  rc=$?
  echo "[run_full.sh] preIdxOffset audit failed (rc=$rc). Aborting." >&2
  echo "[run_full.sh] See $PRE_CHECK" >&2
  exit $rc
}

# ---------------------------------------------------------------------------
# Step 2: BS=1 profile + parse
# ---------------------------------------------------------------------------
BS1_RAW="$OUTDIR/bs1_results_raw.csv"
BS1_SUMMARY="$OUTDIR/bs1_results_summary.csv"
BS1_INDEX="$OUTDIR/bs1_index.json"
BS1_NSYS="$OUTDIR/nsys_bs1"
BS1_NVTX_CSV="${BS1_NSYS}_nvtx_gpu_proj_trace.csv"

if [ "$SKIP_BS1" -eq 0 ]; then
  echo ""
  echo "=== Step 2/4: BS=1 nsys profile (9 layers, stride=$ROW_STRIDE) ==="
  nsys profile --trace=cuda,nvtx --force-overwrite=true -o "$BS1_NSYS" \
      python3 "$SCRIPT_DIR/bench_bs1.py" profile \
          --layers $LAYERS \
          --row_stride "$ROW_STRIDE" \
          --warmup "$WARMUP" --repeats "$REPEATS" \
          --variants "$VARIANTS" \
          --gvr_backend "$GVR_BACKEND" \
          --output "$BS1_INDEX"

  echo ""
  echo "=== Step 2.5/4: nsys stats (BS=1) ==="
  rm -f "$BS1_NVTX_CSV" "${BS1_NSYS}.sqlite"
  nsys stats --report nvtx_gpu_proj_trace --format csv --force-export=true \
      --force-overwrite=true -o "$BS1_NSYS" "$BS1_NSYS.nsys-rep" >/dev/null

  python3 "$SCRIPT_DIR/bench_bs1.py" parse \
      --nsys_csv "$BS1_NVTX_CSV" \
      --index_json "$BS1_INDEX" \
      --output_raw "$BS1_RAW" \
      --output_summary "$BS1_SUMMARY"
else
  echo "[run_full.sh] --skip-bs1 set; expecting existing $BS1_SUMMARY."
fi

# ---------------------------------------------------------------------------
# Step 3: BS-scaling profile + parse
# ---------------------------------------------------------------------------
BSS_RAW="$OUTDIR/bs_scaling_results_raw.csv"
BSS_SUMMARY="$OUTDIR/bs_scaling_results_summary.csv"
BSS_INDEX="$OUTDIR/bs_scaling_index.json"
BSS_NSYS="$OUTDIR/nsys_bs_scaling"
BSS_NVTX_CSV="${BSS_NSYS}_nvtx_gpu_proj_trace.csv"

if [ "$SKIP_BSSCALING" -eq 0 ]; then
  echo ""
  echo "=== Step 3/4: BS-scaling nsys profile (BS in $BS_LIST) ==="
  nsys profile --trace=cuda,nvtx --force-overwrite=true -o "$BSS_NSYS" \
      python3 "$SCRIPT_DIR/bench_bs_scaling.py" profile \
          --layers $LAYERS \
          --bs_list "$BS_LIST" \
          --warmup "$WARMUP" --repeats "$REPEATS" \
          --variants "$VARIANTS" \
          --gvr_backend "$GVR_BACKEND" \
          --output "$BSS_INDEX"

  echo ""
  echo "=== Step 3.5/4: nsys stats (BS-scaling) ==="
  rm -f "$BSS_NVTX_CSV" "${BSS_NSYS}.sqlite"
  nsys stats --report nvtx_gpu_proj_trace --format csv --force-export=true \
      --force-overwrite=true -o "$BSS_NSYS" "$BSS_NSYS.nsys-rep" >/dev/null

  python3 "$SCRIPT_DIR/bench_bs_scaling.py" parse \
      --nsys_csv "$BSS_NVTX_CSV" \
      --index_json "$BSS_INDEX" \
      --output_raw "$BSS_RAW" \
      --output_summary "$BSS_SUMMARY"
else
  echo "[run_full.sh] --skip-bs-scaling set; expecting existing $BSS_RAW."
fi

# ---------------------------------------------------------------------------
# Step 4: render REPORT.md
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4/4: render REPORT.md ==="
REPORT_MD="$OUTDIR/REPORT.md"
python3 "$SCRIPT_DIR/render_report.py" \
    --bs1_summary "$BS1_SUMMARY" \
    --bs1_raw "$BS1_RAW" \
    --bs_scaling_raw "$BSS_RAW" \
    --bs1_index "$BS1_INDEX" \
    --bs_scaling_index "$BSS_INDEX" \
    --preidx_check "$PRE_CHECK" \
    --output "$REPORT_MD"

echo ""
echo "[run_full.sh] DONE."
echo "  REPORT.md   : $REPORT_MD"
echo "  BS=1 raw    : $BS1_RAW"
echo "  BS=1 summary: $BS1_SUMMARY"
echo "  BS-scaling  : $BSS_RAW"
