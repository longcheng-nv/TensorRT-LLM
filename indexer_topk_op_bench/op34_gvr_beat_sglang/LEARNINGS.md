# op34 LEARNINGS — primitive inventory (search compositions before inventing)
Verified GVR primitives (domain in parens):
- block_count_ge / _multi (M static counters, vectorized 4-unroll, cached per-thread col) — op18/op26
- P1b 256-bin smem hist over K hint values -> warp-0 h-quantile rung scan (no serial walk) — op26 iter6
- fused rank-and-scatter P4 (hist->prefix->per-cand scatter, 1 pass, barriers 14->7); EXACT variant
  = +256-bin fine level on straddling bin; P4 is BARRIER/OCC-bound not pass-bound — op#7
- R1 inline log-falsi shot between two measured rungs (1 extra pass on miss) — op26
- guaranteed-fill-K fallback (fb_fix) = correctness safety on threshold miss — op26

Live levers (not falsified on real v4cap):
- intra-CTA warp pipelining (deepest untouched; 25% occ / 10% issue at BS=1 single-CTA)
- shrink P4 cand_count (P3 over-collects 6xK@K512) — iter1 attacks this (band 0.29K)
- fast-write certain-winners (count<K short-circuit) — iter1
- multi-CTA-per-row for BS=1 to match sglang's 8-CTA parallelism (careful: op#3 cluster loses to radix)
