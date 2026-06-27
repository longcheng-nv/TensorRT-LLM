# op12_gvr_lp — Learnings

## Architecture facts (from reading gvr_topk_decode_lp.py, 2139 lines)
- Single CTA/row. Phases: P1 preIdx-stats → P2 secant (full-N count_ge ×iters)
  → P3 collect (v≥thr → smem_keys[fp32]+smem_vals[idx]) → P4 select top-K of cand.
- `smem_keys` is **already fp32** (candidate values); P4 already exact-fp32.
  So opt-1 "fp32→fp16 for P4 refine in fp32" is already the state — no change needed there.
- Input load already casts self.dtype→fp32 (`_load_fp32`, block_count_ge), so bf16/fp16
  *input* already halves load traffic. But SGLang comparison fixes fp32 input → opt-1
  (truncate to cut traffic) needs an fp32→fp16 pre-pass that ~cancels savings unless
  secant iters are high. LOW priority.
- P4 is **barrier-bound**, not work-bound: snap ~14 barriers, rank_scatter ~7,
  rank_scatter_exact adds a fine-hist recursion (extra pass + barriers) → that's why
  rs_exact is *slower* than snap at small N in the report, yet best overall here.
- kFTarget controls cand_count (~3.2×K by default). kC=5120 candidate cap.

## Effective so far
- Config selection matters: rs_exact/512 best all-rounder (median 1.0 vs report's stored 0.868).

## Ineffective / dead ends
- `rs` non-exact (no fine recursion): fails exactness on continuous Beta data (straddle
  bin mis-ordered). Needs the straddle resolved → can't just drop the recursion.

## Open hypotheses
- H1 (opt-2): tight kFTarget (cand≤num_threads) + SGLang-style 1-elem/thread 4-round
  8-bit radix P4 → lean refine, attacks small-N wall. Risk: tighter target → +1 secant
  iter (cheap at small N, costlier at large N → dispatch kFTarget by N).
- H2: finer coarse histogram in rank_scatter (more bins) to resolve b* in one pass,
  dropping the fine recursion. Risk: continuous data may keep distinct values in b*.

## Floor caveat
- SGLang at N=4K ≈ 12.7µs event-timed; event includes ~4µs CUDA-graph launch overhead
  (report nsys/event ≈ 0.88). Any kernel pays the same launch floor → "50% faster
  everywhere" at N=4K (≤8.5µs event) needs ~4µs pure kernel — near the small-CTA floor.
  Track whether small-N target is physically reachable.
