#!/bin/bash
cd "$(dirname "$0")/.."
OUT=results/knobs.csv; EX=results/knobs_exact.txt; : > $OUT; : > $EX
echo "cfg,K,N,mean_ns" >> $OUT
# configs: label|env
CFGS=(
 "base|"
 "d1_p4fast0|OP21_P4_FAST=0"
 "d4_p4rs0|OP21_P4_RS=0"
 "d2_qstock|OP25_QFRACS=0.75,0.5,0.25"
 "d2_qm2|OP25_QFRACS=0.85,0.35"
 "d4_slot1|OP25_SLOTCAP=1"
)
cells=(); for K in 512 1024 2048; do for N in 8192 32768 65536; do cells+=("$K:$N"); done; done
gpu=1
for cfg in "${CFGS[@]}"; do
  lab="${cfg%%|*}"; env="${cfg#*|}"
  ( for c in "${cells[@]}"; do
      K="${c%%:*}"; N="${c##*:}"; tag="${lab}_K${K}_N${N}"
      env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$gpu K=$K N=$N SCEN=real REPS=60 TAG=$tag EXFILE=$EX $env \
        nsys profile -c cudaProfilerApi -t cuda -o results/kn_$tag -f true --stats=false python3 scripts/nsys_cfg.py >/dev/null 2>&1
      m=$(env -u GITHUB_TOKEN -u HF_TOKEN nsys stats --report cuda_gpu_kern_sum --format csv results/kn_$tag.nsys-rep 2>/dev/null | grep -iE "kernel_cutlass|gvr|topk" | head -1 | awk -F',' '{print $4}')
      echo "$lab,$K,$N,$m" >> $OUT
    done ) &
  gpu=$((gpu+1)); [ $gpu -gt 6 ] && gpu=1
done
wait
find results -name "kn_*.nsys-rep" -delete 2>/dev/null; find results -name "kn_*.sqlite" -delete 2>/dev/null
echo KNOBS_DONE
