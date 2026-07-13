#!/usr/bin/env bash
# op33 iter0 CRUX driver — shard NCU attribution cells across 8 GPUs.
# Token-safe (env -u); one CSV per cell in results/crux/.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../results/crux"
mkdir -p "$OUT"

MET="gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__issue_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__grid_size,\
launch__block_size,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"

# cell = op|scen|K|dtype|N|BS   (real scenario = canonical verdict axis)
CELLS=(
  "op26_r0auto|real|512|fp32|16384|1"
  "op26_r0auto|real|512|fp32|65536|1"
  "op26_r0auto|real|512|fp32|262144|1"
  "op26_r0auto|real|2048|fp32|65536|1"
  "op26_r0auto|real|2048|fp32|65536|64"
  "op26_r0auto|real|1024|bf16|65536|1"
  "gvr_ms_auto|real|1024|fp32|65536|1"
  "gvr_ms_auto|real|1024|fp32|131072|1"
  "gvr_ms_auto|real|2048|fp32|262144|1"
  "gvr_ms_auto|real|2048|fp32|262144|64"
  "gvr_ms_auto|real|512|fp32|65536|64"
  "gvr_ms_auto|real|2048|bf16|262144|1"
)

i=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r op scen K dt N BS <<< "$cell"
  gpu=$(( i % 8 ))
  tag="${op}_${scen}_K${K}_${dt}_N${N}_BS${BS}"
  log="$OUT/${tag}.csv"
  echo "### launch gpu$gpu $tag"
  CUDA_VISIBLE_DEVICES=$gpu setsid env -u GITHUB_TOKEN -u HF_TOKEN \
    ncu --profile-from-start off --metrics "$MET" --csv --page raw \
    python "$HERE/crux_ncu.py" --op "$op" --scenario "$scen" \
      --K "$K" --dtype "$dt" --N "$N" --BS "$BS" \
    > "$log" 2> "$OUT/${tag}.err" &
  i=$(( i + 1 ))
  # 8 in flight max
  if (( i % 8 == 0 )); then wait; fi
done
wait
echo "### crux done -> $OUT"
