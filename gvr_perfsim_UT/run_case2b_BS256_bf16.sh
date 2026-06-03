#!/usr/bin/env bash
# case-2b: BS=256, N=65536 (64K), K=1024, bf16, V4 Pro typical (beta_moderate),
# preIdx target hit_rate=0.60. BS=256 rows literally REPLICATE the same BS=1
# logits + preIdx via --bs 256 broadcast at load time (user spec:
# "不同 BS 下直接复制相同的数据").
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python3 bench_gvr_topk.py \
    --case-dir ./data/beta_moderate_v4pro_N65536_bs1_hr0.6_bf16 \
    --bs 256 \
    --backend "${BACKEND:-local}" \
    --label case-2b_BS256_N65536_bf16 \
    --warmup "${WARMUP:-30}" \
    --reps "${REPS:-100}" \
    --l2-flush-mib "${L2_FLUSH_MIB:-128}" \
    --out results/case2b_BS256_N65536_bf16.json
