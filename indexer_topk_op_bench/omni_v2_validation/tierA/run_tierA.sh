#!/usr/bin/env bash
# Tier A: omni-kernel v2 scripts validation (GAPS P4 — scripts robustness).
# Each test has an EXPECTED outcome; a gate that cannot fail is not a gate.
# Usage: bash run_tierA.sh   (from tierA/; writes results/tierA_log.txt)
cd "$(dirname "$0")"
SK=../../gvr_agent_retrospective/skill_v2_draft/scripts
OPS=ops
RES=results
mkdir -p "$RES"
LOG="$RES/tierA_log.txt"
: > "$LOG"
export PYTHONPATH="$OPS:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

pass=0; fail=0
check() {  # check <test-id> <expected: 0|nonzero> <actual-rc>
  local id=$1 expected=$2 rc=$3 verdict
  if { [ "$expected" = 0 ] && [ "$rc" = 0 ]; } || { [ "$expected" != 0 ] && [ "$rc" != 0 ]; }; then
    verdict="OK"; pass=$((pass+1))
  else
    verdict="UNEXPECTED"; fail=$((fail+1))
  fi
  echo "[$verdict] $id (expected rc ${expected}, got ${rc})" | tee -a "$LOG"
}

run() {  # run <test-id> <expected-rc-class> <cmd...>
  local id=$1 expected=$2; shift 2
  echo "=== $id: $* ===" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  check "$id" "$expected" $?
}

# --- verify_exact.py: dense track, positive + negative ---
run A1_dense_pass 0 python3 "$SK/verify_exact.py" --impl "$OPS/rmsnorm_triton.py" --mode dense --dtype bf16 --trials 3
run A2_dense_broken_must_fail 1 python3 "$SK/verify_exact.py" --impl "$OPS/rmsnorm_broken.py" --mode dense --dtype bf16 --trials 3

# --- verify_exact.py: select track (tie-aware multiset), positive + negative ---
run A3_select_tie_pass 0 python3 "$SK/verify_exact.py" --impl "$OPS/topk_tie.py" --mode select --trials 3
run A4_select_broken_must_fail 1 python3 "$SK/verify_exact.py" --impl "$OPS/topk_broken.py" --mode select --trials 3

# --- bench_cold.py: cold-L2 + graph A/B ---
run A5_bench_cold_ab 0 python3 "$SK/bench_cold.py" --impl "$OPS/rmsnorm_triton.py" --baseline "$OPS/rmsnorm_torch.py" --reps 20 --label tierA

# --- nsys_verdict.py: L2 ship arbiter, plus anchor-drift rejection ---
run A6_nsys_ab 0 python3 "$SK/nsys_verdict.py" --impl "$OPS/rmsnorm_triton.py" --baseline "$OPS/rmsnorm_torch.py" --batches 3 --launches 20
run A7_anchor_drift_must_reject 1 python3 "$SK/nsys_verdict.py" --impl "$OPS/rmsnorm_triton.py" --anchor-impl "$OPS/rmsnorm_torch.py" --anchor-expected 1.0 --anchor-tol 0.03 --batches 1 --launches 10

# --- ncu_attrib.sh: with and without INPUT_BYTES (set -u robustness probe) ---
run A8_ncu_with_input_bytes 0 env INPUT_BYTES=234881024 bash "$SK/ncu_attrib.sh" "$OPS/ncu_runner.py"
run A9_ncu_without_input_bytes 0 bash "$SK/ncu_attrib.sh" "$OPS/ncu_runner.py"

echo "" | tee -a "$LOG"
echo "TIER A SUMMARY: $pass OK, $fail UNEXPECTED (details: $LOG)" | tee -a "$LOG"
exit $fail
