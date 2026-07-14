#!/bin/bash
# RELIABLE + RESUMABLE clean paired grid: 2-way shard (K512->GPU1, K1024->GPU2).
# Each GPU runs its cells SEQUENTIALLY, base+m3 back-to-back same GPU => clean ratio,
# no nsys concurrency flakiness. IDEMPOTENT: skips (K,N,scen) already in the CSV, so
# a re-launch on a new node resumes from the partial results.
cd "$(dirname "$0")/.."
OUT=results/reliable_grid.csv
[ -f "$OUT" ] || echo "K,N,scen,base_ns,m3_ns,ratio" > "$OUT"
NS="4096 8192 16384 32768 65536 131072 262144 524288 1048576"
done_cell () { grep -q "^$1,$2,$3," "$OUT" 2>/dev/null; }
shard () {   # $1=gpu $2=K
  local g=$1 K=$2
  for N in $NS; do
    for sc in best real worst; do
      done_cell "$K" "$N" "$sc" && continue
      local b0="" m1=""
      for m3 in 0 1; do
        local tag="rg_K${K}_N${N}_${sc}_m${m3}"
        env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g K=$K N=$N SCEN=$sc M3=$m3 \
          nsys profile -c cudaProfilerApi -t cuda -o "results/$tag" -f true --stats=false \
          python3 scripts/paired_one.py >/dev/null 2>&1
        local v
        v=$(env -u GITHUB_TOKEN -u HF_TOKEN nsys stats --report cuda_gpu_kern_sum --format csv "results/$tag.nsys-rep" 2>/dev/null \
            | grep -iE "kernel_cutlass|gvr|topk" | head -1 | awk -F',' '{print $4}')
        rm -f "results/$tag.nsys-rep" "results/$tag.sqlite"
        if [ "$m3" = "0" ]; then b0="$v"; else m1="$v"; fi
      done
      local r
      r=$(python3 -c "b='${b0}' or '0'; m='${m1}' or '0'; b=float(b); m=float(m); print(f'{b/m:.4f}' if m>0 else '')" 2>/dev/null)
      echo "$K,$N,$sc,$b0,$m1,$r" >> "$OUT"
    done
  done
}
shard 1 512 &
shard 2 1024 &
wait
echo RELIABLE_DONE
