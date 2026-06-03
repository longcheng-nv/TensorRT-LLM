#!/usr/bin/env bash
# case-1a: BS=1, N=65536 (64K), K=1024, fp32, V4 Pro typical (beta_moderate),
# preIdx target hit_rate=0.60, V4 temporal-coherence synth.
# Data is pre-generated and committed at data/beta_moderate_v4pro_N65536_bs1_hr0.6_fp32/
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python3 bench_gvr_topk.py \
    --case-dir ./data/beta_moderate_v4pro_N65536_bs1_hr0.6_fp32 \
    --bs 1 \
    --backend "${BACKEND:-local}" \
    --label case-1a_BS1_N65536_fp32 \
    --warmup "${WARMUP:-30}" \
    --reps "${REPS:-100}" \
    --l2-flush-mib "${L2_FLUSH_MIB:-128}" \
    --out results/case1a_BS1_N65536_fp32.json
