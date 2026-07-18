# Track B — small-N 8-CTA exact path (design scoping, iter3)

Target: the 99-cell ISL 4-16k hole (gvr_pr vs sglang gm 0.599; absolute
4k BS=1: pr 8-11µs vs sglang 4.7-6.7µs; GVR 1-CTA skeleton floor ~9.7µs
sits ABOVE sglang — op32/op34 structural verdict, unreachable in-skeleton).

**Build decision: port + exactness-fix, NOT from-scratch** (op35-apex: own
sampling-filter build landed at gm ~0.51). Source already vendored + benched:
`indexer_topk_op_bench/ops/sglang_v2/` (sglang@main 2026-07-13 device headers
verbatim, tvm-ffi host layer replaced; the harness sglang_v2 arm runs it).

**Exactness fix (the moat)**: sglang v2 is conditionally exact — the tie
collect truncates at `kMaxNumTie = 2048`:
- `sgl_kernel/deepseek_v4/topk_impl.cuh:593/673` — `count_eq < kMaxNumTie`
  guards; `:616` — `tie_count = min(equal_count, kMaxNumTie)` SILENT truncation.
- Real V3.2 L52 @128k/256k already crosses the cap (measured, §8.1); V4 margin
  ≥5.2× at the measured cells, but unconditional exactness requires a guard.
Fix design (ε-cost, exact-by-construction):
1. kernel: when `equal_count > kMaxNumTie`, set a per-row overflow flag in
   gmem (one st.global.u8; the count already exists at :616).
2. host wrapper: rare flagged rows re-run through the exact in-tree radix
   path (unconditionally exact, 2245/2245). Overflow is data-rare and
   never observed at the small-N dispatch window — the re-run is a
   correctness escape hatch, not a perf path.
3. gate: adversarial tie battery (the 2245-check §8 battery + synthetic
   all-tie rows that force equal_count >> 2048) must pass 100% incl. the
   forced-overflow path.

**Dispatch**: shape-keyed at the production op level (allowed): route
K512/K1024/K2048 rows with N below a threshold (start: N < 32768, tune on
the 7b grid) to the ported path; preIdx ignored there (hint-free is FASTER
at small N — op34 hint arm 4-8× slower). Large N keeps GVR PR + A0 table.

**Expected effect (from baseline arithmetic)**: parity at the 99-cell hole
lifts composite from 0.738 toward ~D=1.03 territory combined with mid-band
dispatch; still below the 1.10 target — feeds the PLAN feasibility gate.

**Perf watch-items**: PDL 2-kernel launch overhead vs per-layer plan reuse
(plan() untimed in the report protocol — replicate the production reuse
pattern in the integration); BS>64 mid-valley where GVR already wins 2.2×
(dispatch must NOT route those to the ported path).
