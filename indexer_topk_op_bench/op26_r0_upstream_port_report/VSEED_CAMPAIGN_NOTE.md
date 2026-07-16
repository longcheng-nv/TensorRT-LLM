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
