# op32 — GVR short-row register-resident path (INSIGHTS P0)

## Objective triple
- **incumbent**: `op26_r0auto` (fp32 N<65536 → `gvr_cutedsl_op26`, base GvrOp26Kernel:
  M2 P2 log-secant + fb_fix + rank-scatter P4). A/B ALWAYS vs this.
- **rivals (track)**: sglang_v2 (short-row leader 1.7-2.3× on synth, op28), op27_hls.
- **envelope**: N∈{4096,8192,16384(,32768)} · K∈{512,1024,2048} · dtype fp32(primary), bf16/fp16(secondary) · BS=1 (short-row single-CTA). N≥65536 = out of scope (op29 HBE owns it).
- **verdict_axes**: [worst, real, best] — report all three.
- **ship_rule**: worst improves AND real/best regression-free AND exactness green (tie-aware vdiff=0, 3 tracks) AND dispatch rules ≤3.

## Hard constraints (user-mandated)
1. KEEP the multi-threshold(M=2) + secant(log-falsi fb_fix) skeleton. Do NOT rewrite
   into a sglang histogram-only single-pass framework. Do NOT copy sglang v2 code/skeleton.
2. New behavior via gated flag / subclass only; baseline byte-identical, one-revert recoverable.
3. Distinguish from FALSIFIED op15 smem-resident: target = eliminate PASSES+barriers at
   register level (issue-bound lever), NOT save DRAM re-read traffic. Must show a
   warm-L2 A/B win (op15 died on warm-L2 parity).

## Direction
Base at BS=1 short-N does ~1.46 P2-secant + 1 P3-collect ≈ 2.46 full-N GMEM/L2 load passes.
When the row fits in block registers (N ≤ num_threads×kItems), load ONCE → register-resident
count_ge (secant iters compare register values, zero re-load) + P3 scatter from registers.
Aligns with the OPEN ledger lever "intra-CTA warp pipelining / reduce issue" (Q3'/Q4', not run;
25% occ / 10% issue-rate → ~90% idle cycles at BS=1).

## Red lines (falsification ledger — do NOT re-propose)
- op15 smem-resident (warm-L2 parity → dead). Register ≠ smem: no staging pass, 0-latency reuse.
- Opt-L (fuse P3 into count via online slot-reserve ≈ full scan). Register path keeps P3 SEPARATE
  (ballot-free, uses R0 cached offsets), just reads registers not GMEM → NOT Opt-L.
- P4-internal reseed (dead). M≥4 explicit multi-threshold (issue tax). qfracs UH4.
