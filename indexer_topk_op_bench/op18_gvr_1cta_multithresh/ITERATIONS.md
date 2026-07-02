# op18 single-CTA multi-threshold GVR top-K — iteration log

**Goal (user 2026-07-02):** implement + optimize a SINGLE-CTA multi-threshold GVR
top-K operator, based on the count_ge_multi_bench block_count_ge_multi design and
op17's threshold-selection method (band [pmin,pmax] from preIdx, tightest count>=K),
with parameter re-tuning (M, rounds, accept window replacing kFTarget targeting,
kC, CTA thread count). Deliverable: bilingual HTML report.

HW: B200 sm_100 (148 SMs). Protocol: cold-L2 (512MB evict) + CUDA-graph + cudaEvent
median (harness/sweep.py); nsys pure-kernel validation for positive claims (repo rule).
Baseline: single-CTA gvr_cutedsl on report synth bundles (seed=42).

Priors (falsification-aware):
- op17 iter4: single-CTA M=16 Triton multi-threshold P2 → 0.41-1.00x (ALU tax). But
  M-sweep found K512/N16K M=4 → 1.20x; count_ge_multi_bench (cuteDSL-faithful CUDA)
  measured M=2 ~x1.02 mean, M=4 ~x1.25 mean, M=6 x1.46, M=8 x1.77 per pass.
- op16: tight threshold via SERIAL secant is tax-bound (extra full-N passes);
  "only real lever = cheaper P2" — M-ary parallel evaluation IS that lever.
- op17 iter1b: P4 is cand-linear (0.42-0.83x shrink tight-vs-loose) + ~7500cyc floor.

Design (src/gvr_mt_op.py):
- block_count_ge_multi<M>: ONE full-N scan, M sorted thresholds, M static register
  counters, same vec/4-way-unrolled memory path; caches ALL M per-thread count
  columns (column-major smem) so the winning threshold seeds P3 with ZERO recount
  (closes the count_ge_multi report's "1 extra scan" gap).
- Adaptive rounds: round 0 places M points on [pmin,pmax] (mode 0 uniform /
  1 dyadic-low / 2 pmean-anchored); rounds >=1 refine (thr_best, thr_above)
  uniformly (M+1 divisions). Effective resolution (M+1)^R.
- Accept when best count <= c_accept = accept_mult*K (replaces kFTarget).
- done=1 (count in [K,kC]) → P3 reuses cached column; done=2 → baseline retry-shrink.

---

## Iter 0 — 2026-07-02 — first compile: EXACT everywhere

M=4 R=2 acc=2.0 place=uniform. All 14 fp32 cells (K512/1024/2048 × N4K..262K):
uniq=K, valdiff=0 → exactness gate passed. Next: A/B grid vs gvr_cutedsl.

---

## Iter 1 — 2026-07-02 — first A/B (M4 R2 u a2.0): avg 0.93x, large-N tax-bound

Full fp32 grid, cold-L2 event median (`scripts/ab_grid.py`): min 0.694x (K512/262K),
avg 0.930x, max 1.268x (K512/16K). Wins only at scattered small/mid-N; loses
0.69-0.85x at N>=131K — the unconditional round-2 pass (x1.46 M4 tax at 262K)
cannot be paid back by the P4-shrink (~2µs) at large N.

**Offline policy simulator** (`scripts/simulate_placement.py` — replicates P1 band
incl. cr=1 +1 preIdx offset, baseline secant w/ kFTarget, M-ary rounds, done=2
retry-shrink passes; validated cand/passes on the real bundles):
- Baseline burns 2 full-N passes at EVERY K512/K1024 cell (count(pmean) always
  outside [K,kC]); K2048: 2-3 passes. Final cand: K512 1.3-2.3K, K1024 2.4-4.6K,
  K2048 2.2-5.2K.
- M4 R1: ONE pass, cand 0.9-1.8K (K512) / 2.3-4.5K (K1024) — replaces baseline's
  2 passes with one x1.25-1.46 pass -> the LARGE-N play (saves ~0.5-0.7 pass).
- M4/M8 R2 + accept: 2 passes, cand 0.5-0.7K (K512) / 1.0-1.5K (K1024) /
  2.1-2.6K (K2048) -> the SMALL-N play (P4-shrink, passes are latency-cheap).
=> per-(K,N) config dispatch needed (mirrors op17 pick_G). Running 9-config
empirical sweep (`scripts/config_sweep.py`) to build the dispatch table.

---

## Iter 2 — 2026-07-02 — 9-config sweep + phase localization: the L2 trap, again

**9-config full-grid sweep** (results/config_sweep_fp32.jsonl): best fixed config
M4R1u avg 0.993x (min 0.856, max 1.230); ORACLE per-cell best only avg 1.023x.
Wins concentrate at N<=16K (1.10-1.24x) + 65K (1.06-1.11x); all configs lose at
131K/262K. M6/M8 R2 disasters at large N (0.49-0.66x).

**Phase localization** (`scripts/measure_mt_phases.py`, clock64, K512/262K, cyc):
- baseline: P1 3.5K | P2 27.3K (2 evals: COLD ~22K + L2-WARM ~5K) | P3 29K | P4 10.2K
- mt M4R1u: P1 3.6K | P2 33K (ONE cold M4 pass) | P3 28K | P4 15.3K (cand 1821)
- ROOT CAUSE of the sweep miss vs sim: **the L2 trap (op14/15/17-iter0 pattern)**.
  The sim counted baseline's 2nd pass as a full pass; it is L2-resident (~5K cyc,
  row 1MB << 50MB L2). So "pass-collapse" reclaims only ~5K cyc while the M4
  compare tax on the COLD pass costs +10K cyc at 262K -> structural large-N loss.
- In-kernel M-scaling == count_ge_multi microbench (M2 22K free, M4 32K x1.46,
  M8 59K x2.7 at 262K); branchless Int32(cmp) rewrite and unroll 8/16: NO effect
  (the pass is latency-exposure-bound, not branch- or unroll-bound).
- **P4 snap is placement-sensitive, not cand-linear alone**: P4 = 15.3K @cand
  1821 (uniform thr) but 7.2K @cand 669 (M8 uniform) and 12.6K @cand 532
  (R2-refined thr) — same-cand configs differ ~2x by where thr lands.

**PIVOT (op16's 'cheaper P2' lever, CDF-aware form):** one free-ish pass (M2/M3
large-N, M4-M8 small-N) with **offline-tuned CDF-aware thresholds**:
`scripts/optimize_fracs.py` (5 seeds) picks per-(K,N,M) fracs on [pmin,pmax]
targeting a count ladder; fracs[0]=0 anchors exactness. Multi-seed tightest
count: K512 ~520-840 (vs baseline cand 1327-2680), K1024 ~1.0-2.2K (vs
2.4-4.9K), K2048 ~2.1-3.5K (parity — baseline already tight there). M=2 is
seed-fragile (done2 1-2/5 seeds); M>=3 safe (done2=0 everywhere).
place_mode=3 (compile-time frac table) implemented; EXACT. f3 config sweep
running (M2R2f3_a2, M3R1f3, M3R2f3_a13, M4R1f3, M6R1f3, M8R1f3).

---

## Iter 3 — 2026-07-02 — CDF-aware sweep: BROAD WIN, dispatch table built

f3 sweep (results/config_sweep_f3.jsonl, fp32, cold-L2 event, exact everywhere):
| config | min | avg | max |
|---|---|---|---|
| M2R2f3_a2 | 0.880 | 1.099 | 1.349 |
| M3R1f3 | 0.892 | 1.096 | 1.285 |
| M4R1f3 | 0.969 | 1.081 | 1.284 |
| ORACLE per-cell | 1.004 | 1.143 | 1.349 |

CDF-aware placement turned the large-N regime from 0.70-0.99x into 1.06-1.35x:
M2 (free pass) + R2 accept-guard wins every N>=131K cell (K512 1.06/1.11,
K1024 1.09/1.10, K2048 65K 1.34 / 262K 1.35). Small/mid-N: M3R1/M4R1 1.10-1.29x.
Weakest cells: K1024/4K (1.004), K2048/16K (1.014).

**Dispatch table** `_DISPATCH` + `gvr_mt_auto()` added to src/gvr_mt_op.py
(per-(K, N-bucket) -> (M, R, accept_mult), place_mode=3 everywhere).
Next: x3-median validation (all dtypes) + nsys pure-kernel on positives +
BS sweep guard.

---

## Iter 4 — 2026-07-02 — FINAL VALIDATION: no-regression broad win, all dtypes

**x3-median cold-L2 event, full grid, gvr_mt_auto vs gvr_cutedsl
(results/validate_x3.jsonl), 60/60 cells EXACT:**
| dtype | min | avg | max |
|---|---|---|---|
| fp32 | 1.010 | 1.144 | 1.344 |
| bf16 | 1.000 | 1.114 | 1.244 |
| fp16 | 1.012 | 1.143 | 1.364 |
Zero cells < 0.99 in ANY dtype (fp32 fracs table generalizes to bf16/fp16).

**BS sweep (K512 N65536 fp32, M3R1, results/bs_sweep_k512_n65536.csv):**
1.076-1.164x at BS 1..128, win GROWS with BS — no high-BS guard needed
(unlike op17's cluster which degenerates at BS>=32).

**nsys pure-kernel (100 iters, cold-L2 in cudaProfilerApi window, median,
results/nsys_summary.csv):**
| cell | base us | mt us | nsys spdup | event spdup |
|---|---|---|---|---|
| K512  N16384  | 15.31 | 10.59 | 1.446x | 1.289x |
| K512  N65536  | 18.82 | 16.77 | 1.122x | 1.113x |
| K512  N262144 | 38.46 | 35.10 | 1.096x | 1.080x |
| K1024 N32768  | 21.92 | 16.19 | 1.354x | 1.297x |
| K1024 N262144 | 41.47 | 36.38 | 1.140x | 1.130x |
| K2048 N65536  | 25.06 | 18.37 | 1.364x | 1.313x |
| K2048 N262144 | 51.07 | 36.93 | 1.383x | 1.344x |
nsys >= event on every cell (repo rule satisfied). COMPLETE.
