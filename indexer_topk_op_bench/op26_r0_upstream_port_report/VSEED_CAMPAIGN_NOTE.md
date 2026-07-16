# vseed campaign — flash-1M big-BS regression root cause CORRECTION + P1 estimator study
(2026-07-16, umbriel-b200-072, /tmp/gvrval1 staging)

## Mechanism CORRECTION (kernel-sim on the ACTUAL bench layer L22)
The §7b story "admission miss → extra full-N fallback scans" is WRONG for the
flash-1M cell. Simulating the exact kernel rung placement (256-bin hist over
[pmin,pmax], qneeds crossing, thr = bin lower edge):

| cell (L22) | hit | rung q.85 cnt | rung q.35 cnt | verdict | base pmean cnt |
|---|---|---|---|---|---|
| flash 128k | 0.70 | 894* | 194 | admit slim | 235 (<K, miss) |
| flash 256k | 0.28 | 11622 | 878* | admit slim | 1717* |
| flash 512k | 0.06 | 79977 | 9580 | **MISS both** | 17752 (miss too → base sick) |
| flash 1024k | 0.42 | **4408*** | 385 | **FAT admit** | **633* (slim 1-pass)** |
| pro 1024k | 0.27 | 36690 | 1956* | admit slim | 2354* |

flash-1M: R0 admits the COARSE q.85 rung with 4408 candidates (near kC=5120);
base's pmean lands 633 → P3 collect + P4 rank-scatter carry ~7x more candidate
work in PR/op26. At BS>=128/cs=1 (saturated) that IS the 0.71-0.79 regression.
512k is the opposite regime: BOTH arms miss (base pmean cnt 17752 >> kC) →
base secant walks many passes (43us BS=1, undershoot x36) → PR/base 2-3x.

## P1 estimator study (real captures, all rungs)
True threshold = rank-(hit·K) order statistic of gathered prev-topK values
(exact identity: #gathered >= true_thr = hit·K). Verified: pro hit .86-.94 →
count(>= q-th gathered) ≈ q·K. mean & median are biased HIGH (undershoot side):
flash cells hit .5-.8 give mean/median counts 213-472 < K=512 (inadmissible).
No fixed q works for all rows (hit varies .06-.94) — that's WHY the rung
LADDER exists. Best cheap adaptive point = pmean is wrong; the right insurance
is ADDING pmean (and/or more rungs) as extra columns in the one M-ary pass.

## Fix under test: r0_vseed (virtual seed rung)
Insert P1's pmean as an extra rung in the M-ary count pass (const-folded flag,
+1 column, same memory traffic):
- flash-1M: pmean rung (cnt 633) is admissible AND tighter than q.85's 4408 →
  admission rule (last m in window) picks it → P3/P4 back to base cost.
- true-miss cells (512k): pmean's measured count improves the fallback bracket.
Scripts: study_p1_estimator.py (+_out), study_p1_L22.txt. A/B pending below.

## Round 1 A/B (nsys cold-L2, b200-072 GPU1, 25 cells x {base,pr,vseed-v1}, all exact)
Regression cells FIXED: flash-1M fp32 BS128/256/512/1024 vs/base = 1.01/1.02/
1.01/1.00 (vs/pr 1.29-1.42); 16-bit BS1024 0.70 -> 0.96-0.97; fp32 BS1 1.10
(beats base); v32-256k 1.41/1.20/1.17 (BS 1/128/1024). Guard tax: R0-win cells
(flash 128k/64k, pro 128k/1024k, v32 64k @BS1) vs/pr 0.95-0.97 (~3-5%).
16-bit 1M BS1/BS64 only partially recovered (0.83-0.86 vs base) — separate
16-bit mechanism, not fat-admission.

## m3 control (config-only qfracs=(0.85,.50,.35), 10 cells, 4 arms, all exact)
m3 == vseed on the regression cells (fp32 1M BS128/1024: 1.03/1.00 vs 1.01/1.00)
AND same guard tax (0.95-0.96) -> the tax is the EXTRA COUNT COLUMN itself, not
v1's insert step. vseed beats m3 on v32-256k BS1 (1.40 vs 1.25 over base; pmean
adapts to the value distribution, fixed rank-quantile rungs don't). vseed >= m3
everywhere -> config-only M3 is a viable zero-code fallback but strictly weaker.

## v2 (implemented): pmean parked in last rung column BY P1 (zero extra sync),
admission = explicit argmin count in window (unsorted-safe), fallback bracket =
explicit max/min threshold. Smoke exact. Round 2 pending: {base, pr, vseed-v2,
vs2 = qfracs=(0.85,)+vseed} — vs2 REPLACES the q.35 rung with pmean (2 columns
total = zero column tax) since the estimator study shows pmean covers q.35's
admission region on all observed cells.

## Round 2 (25 cells x {base, pr, vseed-v2, vs2}, nsys cold-L2, all exact)
vs2 (qfracs=(0.85,)+pmean, 2 columns) kills the guard tax (flash 128k/512k,
pro 128k/1024k: 0.99-1.01 vs pr) and fixes MORE than v2 on the regression
cells (16-bit 1M BS1 1.16-1.17x, bf16 BS1024 1.41x, fp32 BS1 1.16x). BUT
v32-64k BS1 vs2/pr = 0.86: pmean count 4007 is admissible-but-FAT (kC/K=3 at
K2048), while pr's miss->refine converges to a slimmer threshold — a fat admit
can lose to a good 2-pass miss. K2048 keeps the q.35 rung.

## WINNER: per-K hybrid (zero new kernel code beyond the vseed flag)
  K512/K1024: r0_qfracs=(0.85,) + r0_vseed  (pmean replaces q.35; 2 columns)
  K2048:      r0_qfracs=(0.85,0.35) + r0_vseed  (3 columns)
Full-envelope validation (REPORT grid: synth seqlen+BS x 3K x 2scen x 3dtype +
real 3-model all-ISL seqlen+BS x 3dtype, 54 nsys batches, 8 GPUs b200-072)
launched -> vsfull_results; aggregate_vsfull.py emits vsfull.csv + regression
list (vs/pr < 0.98). REPORT new-chapter update pending sweep completion.

## Full-envelope audit round 1 (54 batches, 2772 cells) — three findings
1) AGGREGATOR BUG (fixed): NVTX range names lack the scenario -> best/worst
   reps collided on merge; aggregate now joins each batch's rep with its own
   jsonl (_rep_for name-order remap).
2) EXACTNESS: 12 fails, ALL pro/512k fp32 (hit .23, N=131075). CONTROL-PROVEN
   PRE-EXISTING: pristine snapshot kernel + qfracs=(0.85,) fails identically
   (|miss|=1, picks -0.288984 for -0.288981, diff 3e-6) — the P4 rank-scatter
   "exact" one-level fine recursion resolves ~range/1024^2 ≈ 5e-6; adjacent
   values 3e-6 apart in the straddling bin are BELOW its resolution. Same
   class as the op22 §9 "2.7e-6 boundary defect" follow-up. vseed only shifts
   the admitted threshold and thereby which pair straddles — it does not
   introduce the defect. Fix = second recursion level (separate follow-up).
3) PERF TAIL: SEVERE (<0.90 vs pr) concentrated in K2048 16-bit big-BS
   (0.72-0.89, cs1/T512/mb3) and K1024 16-bit large-N BS>=128 — root cause =
   the +1 smem_ptcnt_multi column (+2-4KB) pushed high-occupancy 16-bit
   configs over an smem occupancy cliff (mb probe: vs mb3 68.3us -> mb2
   60.5us vs pr 54.3us).
## v3 (occupancy fix): vseed column's per-thread counts REUSE the existing
smem_ptcnt buffer -> zero smem growth. Bad cell 68.3 -> 56.9us (pr 54.5);
flash-1M BS128 win intact (51.7us vs pr ~80). vsfull2 re-audit launched.

## Full-envelope audit round 2 (v3, RE-RUN on umbriel-b200-049 2026-07-16 —
## b200-072 became unreachable mid-sweep; env re-staged via setup_bs_env.sh +
## vseed_v3.diff onto pristine snapshot; smoke reproduced round-1/2 numbers;
## cross-node base anchor drift 049/072 med 1.001 p95 1.044 = clean)
54 batches / 2772 cells, all-idle node, 0 batch errors.
- vs/pr geomean 0.9963 (r1 v2: 0.9897); vs/base geomean 1.1343.
- SEVERE (<0.90): 105 -> 16, now ONE coherent cluster: K1024 16-bit BS=128
  large-N (262k-1M, 0.80-0.89; only 6 cells inside the N<=256K envelope, all
  at the 262144 boundary) + 2x K2048 16-bit 1M BS256. Same residual 16-bit
  mechanism flagged in round 1 (not fat-admission, not smem growth).
- Envelope N<=256K: geomean 0.9960, min 0.890; outside: 0.9977, min 0.803.
- EXACTNESS: vs fails = exactly the 12 known pre-existing pro/512k fp32
  3e-6 boundary cells (pr: 0 fails); nothing new introduced.
- flash-1M regression FIXED where diagnosed: fp32 BS128-1024 vs/base
  1.02-1.03; 16-bit BS128 0.85-0.87 rising to 0.98-1.00 by BS1024.
- Verdict: per-K hybrid vseed v3 holds the full envelope at ~0.4% mean tax
  with the 16-bit BS128 large-N cluster as the single disclosed residual.

## SHIPPED to PR branch + P4 exact-tail fix + final re-measure (2026-07-16, b200-049)
- vseed + per-K rung defaults -> PR branch @88a563b145 (r0_vseed default ON with
  enable_r0; K512/K1024 qfracs=(0.85,) iff vseed, K2048 keeps (0.85,0.35); explicit
  vseed-off retains the old ladder for all K).
- NEW: P4 exact-tail fix @eae374554c closes the 3e-6 boundary defect (audit
  finding b): ambiguity-gated (tie set overfills remaining slots) MSB-first
  8-bit-digit radix select over order-preserving int keys (4 levels = bit-exact
  fp32), warp-parallel digit scan, all scratch reused (zero SMEM growth);
  fp32-only default (16-bit kernels byte-identical; forcing it on 16-bit hits
  every plateau row ~1.4x - do not). Unit test plants 5e-8 + 1-ulp tie bands.
- vsfull3 re-measure (54 batches, 2772 cells x {base, OLD head @018251950f,
  NEW head @eae374554c}): NEW exact 2772/2772 (12 pro/512k fp32 now exact;
  base keeps its 36 known flash/512k undershoot fails). NEW/OLD geomean 0.9943
  (envelope 0.9937); NEW/base 1.1322 vs OLD/base 1.1388, but worst tail
  0.676->0.790 and <0.90-vs-base count 174->106; real axis NEW 1.196 >= OLD
  1.190. Severe NEW/OLD <0.90 = 25: 14 = known vseed K1024 16-bit BS128
  large-N residual, 9 = pro/512k fp32 repair-active rows paying the
  correctness price (previously WRONG results), 2 = K2048 16-bit 1M BS256.
- REPORT.html new §9b (auto-fills from vseed_harness/vsfull3.csv).
- Neutrality A/B harness: vseed_harness/ab_tail_neutrality.py.
