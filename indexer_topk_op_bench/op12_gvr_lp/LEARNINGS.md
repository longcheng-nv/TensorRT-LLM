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

## RESULTS (decisive)
- **P4 is barrier/latency-floor bound, NOT candidate-count bound.** snap/rs/rs_exact
  all cluster at the same P4 cost; tightening cand_count via `kc_accept` (opt-2) did
  NOT shrink P4 and *added* secant passes → worse at N≥16K. **opt-2 rejected by data.**
- **opt-1 (fp16 traffic) is a no-op for the fp32-vs-SGLang comparison**: candidate keys
  are already fp32 in smem; input is fixed fp32 in HBM; cutting traffic needs an
  fp32→fp16 pre-pass that ~cancels its own savings (helps only at very high secant iters).
- **P4 = 45–50% of GVR time** (4–11µs); P1+P2+P3 floor alone ≈1.2× SGLang at small N.
- **50%-everywhere is physically infeasible at N≤16K**: shared ~4µs CUDA-graph launch
  + GVR's intrinsic secant put the floor above SGLang/1.5×. Best small-N ceiling ≈1.2×.
- **Best achievable op = regime dispatch** (`p4_mode="dispatch"`, the new default):
  rs_exact/512 for N<131072, snap/1024 for N≥131072. Seqlen sweep (BS=1): wins large N
  1.2–1.9×, parity/slight-loss small N; median ≈1.05, big large-N gains over any single config.

## Floor caveat
- SGLang at N=4K ≈ 12.7µs event-timed; event includes ~4µs CUDA-graph launch overhead
  (report nsys/event ≈ 0.88). Any kernel pays the same launch floor → "50% faster
  everywhere" at N=4K (≤8.5µs event) needs ~4µs pure kernel — near the small-CTA floor.
  Track whether small-N target is physically reachable.
