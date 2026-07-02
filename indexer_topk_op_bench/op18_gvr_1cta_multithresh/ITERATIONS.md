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
