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
