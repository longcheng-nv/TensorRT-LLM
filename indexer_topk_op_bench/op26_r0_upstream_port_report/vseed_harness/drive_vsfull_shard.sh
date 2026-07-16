#!/bin/bash
# §9 rival sweep shard driver.  Usage: drive_rival_shard.sh <W> <GPU> <NW>
# Reads batches from batches_vs.py; runs batch index i where i%NW==W on GPU.
# Per batch: one whole-batch nsys-rep (all cells x arms in NVTX ranges), then
# leave the .nsys-rep for parse_rival.py. Resumable: a batch whose jsonl already
# has all cells is skipped by sweep_rival's cell-level _load_done; a batch whose
# .done marker exists is skipped entirely here.
#
# RH (harness+jsonl) lives on NFS in-place (ops_rival/sweep_rival resolve the
# op-bench root via parents[2], so must run from the NFS tree). nsys-reps go to
# /tmp (they embed env tokens -> NEVER commit; and are large).
W=$1; GPU=$2; NW=$3
WD=/tmp/gvrval1
RH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/vseed_harness
OUT="$WD/vsfull_results"
mkdir -p "$OUT/nsys_reps"
# env: bypass stale NFS userbase; cutlass 4.5.0 (make_fragment) + flashinfer 0.6.11
export PYTHONNOUSERSITE=1
C450=$WD/cutlass450/nvidia_cutlass_dsl/python_packages
[ -d "$C450" ] && export PYTHONPATH="$C450:$WD/cutlass450:$WD/fi_clean${PYTHONPATH:+:$PYTHONPATH}"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"

mapfile -t BATCHES < <(python3 "$RH/batches_vs.py")
i=0
for batch in "${BATCHES[@]}"; do
  if [ $((i % NW)) -eq $W ]; then
    read -r fam sw a b c <<< "$batch"
    if [ "$fam" = "synth" ]; then
      tag="synth_${sw}_${a}_K${b}_${c}"                  # a=scen b=K c=dtype
      args="--family synth --sweep $sw --scenario $a --K $b --dtype $c"
    else
      tag="real_${sw}_${a}_${b}"                          # a=model b=dtype
      args="--family real --sweep $sw --model $a --dtype $b"
    fi
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP done: $tag"; i=$((i+1)); continue; fi
    echo "=== refresh batch [$W]: $tag -> $rep.nsys-rep ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 "$RH/sweep_vsfull.py" $args --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM" > "$OUT/${tag}.log" 2>&1; then
      touch "$done_m"
    else
      echo "!!! batch $tag FAILED (see $OUT/${tag}.log; left un-marked for resume)"
    fi
  fi
  i=$((i+1))
done
echo "VSFULLWORKER${W}DONE"
