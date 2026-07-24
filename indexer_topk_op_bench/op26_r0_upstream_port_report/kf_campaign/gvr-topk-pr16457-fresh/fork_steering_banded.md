# RE-SCOPED TARGET (banded bar) + steering from rounds 1-3

## New acceptance bar (replaces the flat gm>=1.60 / 0-regression goal)

Split the workload envelope into two ISL bands and satisfy BOTH:

- **Band A (PRIMARY, where the money is): ISL 32K-1M** (i.e. n = seq_len >=
  8192 for V4 Flash/Pro, >= 32768 for V3.2 — every workload whose uuid names
  isl 32k/128k/512k/1024k, and the bs-extension workloads at those sizes):
  **geomean >= 1.60x vs baseline, every case >= 0.95x.**
- **Band B (guard): ISL 4K-32K** (uuid isl 4k):
  **geomean >= 1.00x, every case >= 0.95x.**

Regression tolerance is now 5% per case (was: zero). Use it: a <=5% loss on a
small-n case that buys a big Band-A win is a GOOD trade. Priority within Band
A: V4-Pro (k=1024) > V4-Flash ~= V3.2; larger n first.

## Hard lessons from rounds 1-3 (do not re-learn these with GPU time)

1. **Stop optimizing the 4k cells.** Platform timing on small-n BS=1 is
   harness-floor-bound (~15-23us measured; the true kernel is <1us) — wins
   there are mispriced by the floor and are Band-B guard cases now, not
   targets. Round-3's champion won mostly 4k cells and still scored 0.84
   overall. Every hour spent there is wasted.
2. **The baseline is strong at large n.** It streams near DRAM roofline for
   its 2-pass structure. To clear 1.6x in Band A you must CUT ROW TRAFFIC,
   not just tune: the baseline reads ~2.05x the row; a true single-read
   pipeline (threshold estimate + candidate collect fused into ONE pass over
   the row, second pass only over candidates) is the only lever with 1.6x+
   headroom at DRAM-bound sizes. Think: per-CTA streaming histogram/quantile
   sketch of the row WHILE collecting above a conservative floor, then refine
   from the collected superset — never touch the full row again.
3. Confirmed-productive primitives from your own rounds (reuse, don't
   rediscover): two-level radix histogram for k=2048 heavy tails;
   1024-thread coop blocks on bandwidth-bound batched cases; runtime
   candidate-cap smem + batch-aware occupancy; shape-dispatched hybrid;
   measured prior-min endpoint to cut undershoot scans; smem-cached exact
   radix ONLY for n<=8192 (fails beyond).
4. Confirmed dead ends: warp-aggregated histogram atomics at k=2048;
   multi-probe secant (2-point wins); cooperative-groups grid.sync (judge
   non-compliant) — use plain multi-launch or manual barriers;
   redundant cluster reductions; whole-row smem caching past 8192.
5. GVR skeleton stays mandatory (preIdx prior -> threshold estimate ->
   secant/log-style refine on exact counts -> exact refine of candidates),
   but P1/P4 are free: radix/histogram primitives inside phases are
   encouraged. Exactness (tie-robust value multiset, every row) is
   non-negotiable — an inexact 2x is worth nothing.
6. BS>1 rows are identical copies: per-row work is identical, so batch wins
   come from occupancy/wave shaping, not per-row algorithm changes. Band-A
   bs-extension workloads (bs32-1024 at large n) reward kernels whose grid
   scales with b without occupancy collapse.
