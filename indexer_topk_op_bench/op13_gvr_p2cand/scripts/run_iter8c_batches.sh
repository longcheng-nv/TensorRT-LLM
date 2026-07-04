#!/bin/bash
# iter8c driver: wait for GPU idle, then run the 9 nsys A/B batches
# (3 K x 3 reps, rep-outer interleave, skip-if-exists => resumable).
# Re-checks co-tenancy BEFORE EVERY batch (cold-L2 timing is invalid under
# co-tenancy); nsys runs with GITHUB_TOKEN/HF_TOKEN scrubbed (sqlite embeds
# process env — feedback_nsys_sqlite_env_token_leak).
set -u
cd "$(dirname "$0")/.."
mkdir -p results/nsys

wait_free() {
  while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    if [ "$m" -lt 30000 ]; then return 0; fi
    sleep 60
  done
}

for r in 1 2 3; do
  for K in 512 1024 2048; do
    out=results/nsys/p2clog_K${K}_fp32_r${r}
    if [ -f "${out}.nsys-rep" ]; then echo "SKIP $out (exists)"; continue; fi
    wait_free
    echo "BATCH START K=$K r=$r $(date -u +%H:%M:%S)"
    env -u GITHUB_TOKEN -u HF_TOKEN nsys profile -t cuda,nvtx \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      -o "$out" -f true \
      python3 scripts/nsys_p2clog_ab.py --K "$K" --dt fp32 \
      >> results/p2clog_ab_run.log 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "BATCH FAIL K=$K r=$r rc=$rc"; else echo "BATCH DONE K=$K r=$r"; fi
  done
done
echo ALL_BATCHES_DONE
