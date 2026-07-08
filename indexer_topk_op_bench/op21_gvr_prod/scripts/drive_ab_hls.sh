#!/usr/bin/env bash
# op21 iter13 — nsys A/B driver for the HLS log-falsi fallback.
# One nsys run per scenario (paired old/new arms inside; see
# ab_hls_logfalsi.py). Anchor cell first (axis re-baseline on a new node).
#
# Usage: GPU=0 [SCENARIOS="best worst real"] ./drive_ab_hls.sh
# nsys gotchas: env -u GITHUB_TOKEN -u HF_TOKEN (sqlite embeds process env);
# `nsys -c cudaProfilerApi` exits 143 on SUCCESS -> no `set -e`.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP21="$(cd "$HERE/.." && pwd)"
GPU="${GPU:-0}"
SCENARIOS="${SCENARIOS:-best worst real}"
# KNOB selects the A/B arm env var (iter13 OP21_FB_LOGFALSI, iter14
# OP21_FB_DIST); SUFFIX names the output dir variant.
KNOB="${KNOB:-OP21_FB_LOGFALSI}"
OUT="$OP21/results/nsys/iter13_ab_hls${SUFFIX:-}"
mkdir -p "$OUT"
cd "$OP21"

echo "### drive_ab_hls: host=$(hostname) GPU=$GPU start=$(date -u +%FT%TZ)"

# --- anchor cell: K512 fp32 262144 BS1 (iter7 axis expects ~18.0+-0.3us on
# 047-GPU0; >3% off => new-silicon axis, note it in ITERATIONS) ---
if [ ! -f "$OUT/anchor.nsys-rep" ]; then
  echo "=== anchor cell ==="
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -f true -o "$OUT/anchor" \
    python3 scripts/nsys_run_auto.py 512 fp32 262144 1 60 \
    > "$OUT/anchor.out" 2>&1
  env -u GITHUB_TOKEN -u HF_TOKEN nsys stats --report cuda_gpu_kern_sum \
    --format csv "$OUT/anchor.nsys-rep" 2>/dev/null | grep -i "gvr\|topk" \
    | head -3 | tee "$OUT/anchor.csv"
fi

# --- paired A/B per scenario ---
for scen in $SCENARIOS; do
  rep="$OUT/ab_${scen}_fp32"
  if [ -f "$rep.done" ]; then echo "SKIP $scen (done)"; continue; fi
  echo "=== A/B scenario=$scen ($(date -u +%T)) ==="
  rm -f "$rep.nsys-rep" "$rep.sqlite" "$rep.jsonl"
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -f true -o "$rep" \
    python3 scripts/ab_hls_logfalsi.py --scenario "$scen" \
    --env-knob "$KNOB" --out "$rep.jsonl" --reps 30 > "$rep.out" 2>&1
  if grep -q "AB BATCH DONE" "$rep.out"; then
    touch "$rep.done"; echo "OK $scen"
  else
    echo "FAIL $scen — tail of $rep.out:"; tail -5 "$rep.out"
  fi
done
echo "### drive_ab_hls DONE $(date -u +%FT%TZ)"
