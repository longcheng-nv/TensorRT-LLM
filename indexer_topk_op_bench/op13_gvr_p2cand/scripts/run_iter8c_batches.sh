#!/bin/bash
# iter8c driver v2 (hardened after the 2026-07-04 b200-019 contamination):
#   * wait_free now requires mem<30GB AND util<=5% on ALL GPUs, 3 consecutive
#     samples 20s apart (a compute co-tenant can have a small mem footprint).
#   * post-batch sanity gate: parse the fresh rep; base@minN<20us AND
#     base@maxN<65us (known-good B200 bands; contaminated runs showed
#     75-611us and within-run drift). Fail => quarantine rep + redo batch
#     (<=3 tries) after re-waiting for idle.
#   * nsys runs with GITHUB_TOKEN/HF_TOKEN scrubbed (sqlite embeds env).
# Resumable: skip batch if its .nsys-rep exists (only good reps survive).
set -u
cd "$(dirname "$0")/.."
mkdir -p results/nsys results/nsys/quarantine

wait_free() {
  local good=0
  while [ $good -lt 3 ]; do
    read -r maxmem maxutil <<<"$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' 'BEGIN{m=0;u=0}{if($1+0>m)m=$1+0; if($2+0>u)u=$2+0} END{print m, u}')"
    if [ "$maxmem" -lt 30000 ] && [ "$maxutil" -le 5 ]; then
      good=$((good+1))
    else
      good=0
      echo "WAIT: maxmem=${maxmem}MiB maxutil=${maxutil}% $(date -u +%H:%M:%S)"
    fi
    sleep 20
  done
}

sane() {  # $1 = rep path (without extension ok? pass full .nsys-rep)
  python3 - "$1" <<'EOF'
import sys
sys.path.insert(0, "scripts")
from nsys_p2clog_ab import parse_cells
cells = parse_cells(sys.argv[1])
if not cells:
    print("SANITY: no cells parsed"); sys.exit(1)
ns = sorted(cells)
b_lo = cells[ns[0]].get("base"); b_hi = cells[ns[-1]].get("base")
if b_lo is None or b_hi is None:
    print("SANITY: missing base"); sys.exit(1)
ok = b_lo < 20.0 and b_hi < 65.0
print(f"SANITY: base@{ns[0]}={b_lo:.2f}us base@{ns[-1]}={b_hi:.2f}us -> {'OK' if ok else 'CONTAMINATED'}")
sys.exit(0 if ok else 1)
EOF
}

for r in 1 2 3; do
  for K in 512 1024 2048; do
    out=results/nsys/p2clog_K${K}_fp32_r${r}
    if [ -f "${out}.nsys-rep" ]; then echo "SKIP $out (exists)"; continue; fi
    tries=0
    while [ $tries -lt 3 ]; do
      tries=$((tries+1))
      wait_free
      echo "BATCH START K=$K r=$r try=$tries $(date -u +%H:%M:%S)"
      env -u GITHUB_TOKEN -u HF_TOKEN nsys profile -t cuda,nvtx \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o "$out" -f true \
        python3 scripts/nsys_p2clog_ab.py --K "$K" --dt fp32 \
        >> results/p2clog_ab_run.log 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then echo "BATCH FAIL K=$K r=$r rc=$rc"; rm -f "${out}.nsys-rep" "${out}.sqlite"; continue; fi
      if sane "${out}.nsys-rep"; then
        echo "BATCH DONE K=$K r=$r"
        break
      else
        ts=$(date -u +%H%M%S)
        mv "${out}.nsys-rep" "results/nsys/quarantine/$(basename $out)_bad${ts}.nsys-rep" 2>/dev/null
        rm -f "${out}.sqlite"
        echo "BATCH CONTAMINATED K=$K r=$r (quarantined, redo)"
      fi
    done
  done
done
echo ALL_BATCHES_DONE
