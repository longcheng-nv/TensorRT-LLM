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

## Iter 2-3 — PREMISE FALSIFIED (the decisive finding)
- The "≈1-HBM-pass" win cannot exist on B200 at N≤262144 because the BASELINE is
  ALREADY ≈1 HBM pass. B200 L2 = 126.5 MB. The fp32 input is N*4 bytes: 0.25/0.5/
  1.0 MB at N=65536/131072/262144 — fits in L2 by 100-500×. ncu (even with
  --cache-control all flush) shows dram__bytes_read.sum = 1.11 MB for BOTH base
  and 1pass at N=262144 = ONE input read; baseline's P2-secant + P3-collect
  re-reads are served from L2 (L2 hit-rate ~80% on re-reads), NOT HBM.
- ⇒ compaction saves ZERO HBM traffic. It only ADDS: (a) a full-N pass-1 with a
  per-element warp-collective (vote_ballot+popc+shuffle) emit, (b) a global scratch
  write of c0 survivors, (c) scratch P2/P3 re-reads from (cold) global. Net loss
  scales with N: +68%/+93%/+106% (nsys ×3, cold-L2, K512 fp32).
- c0 = #{v >= pmin} GROWS with N (not survivor-bounded by a few×K): 6.8k/9.2k/10.9k
  at 65k/131k/262k for K512. cap must be ≥~24×K for the fast path to fire at 262k;
  even so it loses.
- GENERAL LESSON for indexer top-K micro-opt on Blackwell: any "reduce #passes over
  the input" lever is moot while the input fits L2 (true for ALL DSv4 decode N, max
  262144 ⇒ 1 MB ≪ 126.5 MB). The real levers are the P4 floor + launch floor
  (op12/op13), NOT HBM passes. This retroactively explains why op13's pass-count
  changes were ~washes. ncu dram__bytes_read is the one-line test: base==1× input.
- per-element warp-collective ops (ballot/shuffle) over all N are themselves a
  multi-pass-equivalent cost — never put them in the hot streaming loop.
