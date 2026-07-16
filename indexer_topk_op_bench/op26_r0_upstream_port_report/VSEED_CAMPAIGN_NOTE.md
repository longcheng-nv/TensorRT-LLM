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
