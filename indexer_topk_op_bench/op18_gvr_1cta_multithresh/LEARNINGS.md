# op18 single-CTA multi-threshold GVR top-K — LEARNINGS

## Current best
- **File:** `src/gvr_mt_op.py` (`gvr_mt_auto(...)` — tuned per-(K,N) dispatch).
- **What:** single-CTA kernel (baseline grid, no cluster). P2's secant is replaced
  by ONE adaptive M-ary pass: `block_count_ge_multi<M>` evaluates M sorted
  thresholds per full-N scan (same vec/unroll memory path; M static register
  counters, branchless predicated adds) and caches ALL M per-thread count
  columns in smem — the tightest count>=K column seeds P3 with ZERO recount.
  Round-1 thresholds are **CDF-aware**: per-(K,N,M) fracs on [pmin,pmax] fit
  offline on 5 seeds to a count ladder (fracs[0]=0 anchors count>=K exactness).
  Optional round 2 (accept guard c_accept) refines inside the chosen bracket.
- **Result:** x3-median cold-L2 event, vs single-CTA gvr_cutedsl, EXACT 60/60:
  fp32 1.010-1.344 (avg 1.144), bf16 1.000-1.244 (avg 1.114), fp16 1.012-1.364
  (avg 1.143). nsys pure-kernel 1.10-1.45x on the 7 spot cells (>= event).
  BS 1..128: 1.08-1.16x, win grows with BS (no guard needed).

## Effective techniques (by impact)
- **CDF-aware threshold placement (the decisive lever):** offline multi-seed
  frac table targeting a count ladder — turned large-N from 0.70-0.99x (uniform/
  dyadic placement) into 1.06-1.35x. This is op16's hypothesized "CDF-aware
  cheaper P2", realized. M>=3 with a pmin anchor is seed-safe (done2=0 on 5
  seeds); M=2 alone is fragile (1-2/5 seeds blow past kC).
- **Regime dispatch (M,R,acc per K,N):** M4-M6 R1 at N<=8K (latency-bound, taxes
  hidden), M3 R1 at 16-65K, M2 R2 acc=2K at >=131K (M2 is the only tax-free
  compare width on a cold bandwidth-exposed pass; R2's warm refine pass is the
  safety net that lets M stay at 2).
- **Cached M count columns (smem_ptcnt_multi):** chosen column copied to
  smem_ptcnt -> P3 collect with done=1, zero recount. Closes the count_ge_multi
  report's "P3 cache" gap without M x SMEM blowup concerns (M*threads*4B <= 32KB).
- **kFTarget/kC acceptance replaced by c_accept ladder:** the M-ary round makes a
  tight accept window affordable; kC only remains as the exactness/overflow bound.

## Ineffective / falsified this bucket
- **Uniform & dyadic placement** (iter1-2): avg 0.93x, large-N 0.49-0.99x.
- **Branchless Int32(cmp) rewrite:** zero effect — the M-pass is memory-latency-
  exposure-bound, not branch-bound (cuteDSL if-increment was already fine).
- **Deeper unroll (8/16):** 33.1K -> 31.9K cyc only. Outstanding-load count is
  not the limiter; in-kernel M-scaling == the count_ge_multi CUDA microbench
  (M2 free, M4 x1.46, M8 x2.7 at 262K cold).
- **M>=6 at N>=65K:** compare tax on the cold pass 0.56-0.92x. Exactly the
  count_ge_multi report's "do NOT push M to 6-8 at large N".

## Architecture notes (B200 sm_100)
- **The L2 trap governs single-CTA multi-threshold:** baseline's 2nd+ P2 passes
  are L2-resident (~5K cyc vs ~22K cold at K512/262K) — "pass collapse" only
  reclaims warm passes; the M compare tax rides the COLD pass. That is why
  M2/M3 (free width) + better placement wins, and M4-collapse-everything loses.
- **P4 histogram-snap is placement-sensitive, NOT cand-linear alone:** measured
  7.2K vs 15.3K cyc at same-order cand, same cell, different thresholds. Tune
  by measured P4, not by cand count. (Corrects op17-iter1b's linear model.)
- clock64 phase stamps (measure_mt_phases.py) — decisive tool: localized the
  gap to {cold-vs-warm pass accounting, P4 snap} in one afternoon.

## vs op17 (cooperative-cluster portfolio)
- op17 nsys BS=1: 1.21-1.67x but needs a 16-CTA cluster, degenerates BS>=32,
  G=2 unstable. op18: 1.10-1.45x nsys, single CTA, wins at ALL BS (grows to
  1.16x at BS=128), no cluster machinery, no dispatch cliff. Complementary:
  low-BS peak -> op17; robust all-BS single-CTA -> op18.

## Open follow-ups
- Refit fracs on real captures (report bundles are synth; real DSv4 CCDF tails
  differ per synth-vs-real memory) + per-dtype refit (bf16 avg 1.114 < fp32
  1.144 suggests headroom).
- P4 snap placement pathology: root-cause why some thresholds double snap time;
  a snap-aware placement objective could lift the weak cells (K1024/4K 1.02).
- Fold the M2R2 CDF machinery into the production kernel behind a flag; A/B on
  real-loop captures (dsv4-indexer-capture) before any PR.
