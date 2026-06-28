# op14 — Learnings

## Ceiling (measured)
- Baseline = 3 full-N HBM reads at large N (2 P2 evals + 1 P3). cand_final ≈4.3-4.6×K
  ≪ N. P2+P3 ≈80% of total @262K ⇒ compaction (3N→~1N) can give ~40-50% at large N.
- Small N (≤32K): launch floor + rank-scatter P4 floor (~14-20us, op12) ⇒ ~neutral.

## Design
- Fast path: pass-1 fused count+compact survivors(≥pmin) to global scratch (atomic
  slot); exact iff K≤c0≤cap; secant+P3 then run on scratch (≪N). Fallback to
  baseline on undershoot/overflow.

## Iter 1
- Impl works + exact. Scratch P2 reuses the SAME s_thr=[pmean,pmin,pmax] state from
  P1 untouched — counting >=thr over scratch (survivors >= pmin) == counting over
  full N for any thr>=pmin, so bracket seeds carry over (only reset done/cnt_lo/
  cnt_hi/out_count). t0=pmin is the conservative seed.
- bf16 boundary-tie defect is PRE-EXISTING in the rank-scatter baseline (K1024 bf16
  N262144 beta_shallow s1: 27 elems tie at K-th bf16 value → uniq<K). 1pass inherits
  it identically (same uniq, different arbitrary tie-break since survivors collected
  in scratch order). Not a 1pass bug; fp32 is clean.
- atomic-per-survivor (NOT per-N) is cheap: only c0≈4-5×K qualify >=pmin.
