# F1 deviations from TRACKF1_DESIGN.md

## v4 (final structure) notes

- **CAP-overflow fallback is an integer-bisection threshold select, not the
  suggested chunked moving-upper-bound collect.** The chunked sketch is not
  exact: with cnt_straddle > CAP + need, a round's CAP-bounded collect yields
  an ARBITRARY subset of the below-bound elements, and an element at subset
  rank j is only guaranteed inside the global top-need when
  j + (n_below - CAP) <= need — i.e. ZERO safe picks per round (no progress,
  or wrong picks if forced). The bisection select is the same cost class
  (<= 40 bounded rounds of one straddle-membership count pass each, one
  final write pass), ~90 source lines, and exact: it bisects on the
  order-preserving key `skey(b) = b ^ ((b >> 31) & 0x7FFFFFFF)` (signed-int
  compare == fp32 value order), so threshold ties are bit-identical values —
  interchangeable for value-set exactness (the idx tiebreak the sketch asked
  for is unnecessary). Overflow-safe signed midpoint:
  `(lo & hi) + ((lo ^ hi) >> 1)`; termination when mid == lo.
- **Deep recursion deleted** (v2/v3 ~266-line block); v1-era level-loop
  metadata slots (256+4k..) are no longer used beyond insertion-1's staging
  at 258. Bisection state lives at smem_hist[260..263].
- v3's insertion 3 (scratch store inside the original scatter) is REMOVED;
  the scatter and pad are the original text, unconditional. The collect pass
  (need_more rows only) recomputes cb/sb with the identical expressions.
- Scratch ring stays at smem_hist[4..259]; in v4 the collect runs after the
  post-scatter barrier so slots 2/3 are dead anyway, but keeping the v3 slots
  avoids re-deriving the layout.
- PTX line delta (K=512, N=4096, T=512 variant): ON = 6132 vs OFF/base =
  5759, +373 lines (+6.5%). OFF is byte-identical to the snapshot modulo the
  mangled kernel name.

## v3 (F2' scatter-integrated tail-select) notes

- **Scratch ring slot choice**: `smem_hist[4..259]` (pairs, CAP=128), NOT the
  suggested 288..543 and NOT 0..255. Rationale: 288+2*CAP=544 exceeds the
  512-bin (bf16/fp16) specs; 0..255 would include fine bins 2/3 which stay
  LIVE until every thread has read sb_star0/rank_above_fine after the
  pre-scatter barrier — a fast scatter thread writing entry o=1 (slots 2/3)
  races slow readers. Slots 4..259 are dead at scatter time: fine bins 4..255
  dead after the search, 256..259 (= level-0 metadata slots) are only reused
  by the deep publish, which runs exclusively in the cnt>CAP fallback branch
  that never reads scratch. Mid counters (272..275), need_more (276), L (277)
  and deep-done (278) are all above 259.
- **Selection**: serial by thread 0 (coordinator-sanctioned "simplest correct
  wins"): `need` passes over `cnt_strad <= 128` scratch entries, direct fp32
  compares of the bit-cast values (no order-key transform), consumed entries
  overwritten with NEG_FLT_MAX bits. Residual edge: a genuine candidate value
  exactly equal to -FLT_MAX inside the straddle bin would collide with the
  consumed marker (never occurs for finite real logits; NaN logits are
  out-of-contract everywhere in this kernel).
- **Original scatter + pad now run unconditionally** (hot path falls through
  with no taken branch); the deep fallback rewrites ALL output positions so
  the earlier scatter writes are superseded, and the original pad is a no-op
  whenever need_more fired (filled == kK there).
- **Gate-C attribution finding (silicon, 25/25 bench cells)**: need_more
  fires on 0/25 real bench rows (every cell ra_fine = K-1, cnt_straddle = 1);
  host replay agrees (2/25 marginal with the top-kC proxy). The v2 bench
  regression (gm 1.0254) is therefore NOT deep-pass cost — it is
  icache/code-size plus the tiny level-0 insertions. v3 keeps the fix shape
  and bounds worst-case cost, but cannot by itself recover an icache-driven
  tax; if Gate C still fails, the lever is code-size (e.g. moving the deep
  fallback out of line), not algorithm.

## v2 restructure (Gate-C perf fail on v1: flag-ON hot path paid loop scaffolding)

- Flag-ON level 0 is now the ORIGINAL one-shot code text (fine hist + 3-step
  search + ORIGINAL 3-class scatter) with exactly two marked insertions:
  (1) target-warp lane0 stages `smem_hist[sb_star]` into `smem_hist[256+2]`
  BEFORE the pre-existing hist[2]/[3] scratch writes clobber fine bins 2/3
  (otherwise `cnt_straddle` is unreadable when sb_star∈{2,3});
  (2) thread0 computes the ONE runtime `need_more = (ra_fine + cnt_straddle
  > kK) AND (width > ulp_floor)` inside the existing counter-zero block and
  publishes it to `smem_hist[276]`, read block-uniformly after the existing
  barrier. need_more==0 (hot path) runs the original scatter verbatim;
  need_more==1 enters the deep section (fine levels 1..3 + chain scatter).
- Deep-loop done flag lives at `smem_hist[278]`, NOT reusing 276: rewriting
  276 inside the deep branch could race a late reader of the uniform
  need_more test (read-write on the same slot with no intervening barrier)
  and diverge barriers.
- The deep section re-publishes level-0 state (f_lo/finv bits, sb_star_0,
  ra_0) into the per-level metadata slots on entry; levels 1..3 and the
  chain scatter are the v1 code restricted to `range_constexpr(1, MAXL)`.

- The deep-entry publish must SEED level-1's grid params (f_lo_1/finv_1 bits
  into slots 256+4/256+5): in v1 the level-0 loop iteration published them,
  but the one-shot level 0 does not. Missing this made level 1 bin with
  garbage (stale coarse-hist counts bit-cast as floats) and collapse the
  chain into bin 0 (found via cute.printf: lvl=1 sb_s=0 cnt_str=chain-size).
- DSL naming constraint discovered: the AST preprocessor promotes a variable
  to function scope when it is assigned in BOTH const_expr branches; the
  original elif's "first-assign `sb_star` inside a dynamic if" then becomes
  an illegal None->Int32 update AT THE UNTOUCHED ORIGINAL TEXT. The v2 copy
  therefore renames its level-0 `sb_star` to `sb_star0` (rename-only, no
  logic change); names shared at region level (f_lo, finv, pre_f, twf2, ...)
  are safe (v1-proven).

## v1 deviations (still applicable)

- **Loop form**: used the design's sanctioned fallback `for lvl in
  cutlass.range_constexpr(MAXL)` + block-uniform `done` skip-body guard
  (smem flag read after a barrier) instead of a dynamic `while True/break` —
  the DSL AST preprocessor does not support `break` out of a dynamic while,
  and the design names the unrolled form as the safest.
- **L / continue-flag slots**: design suggested `smem_hist[271]/[270]` as the
  s_iscalars fallback, but those collide with the per-level metadata block
  (`256+4k+3` for k=3 is 271). Used `smem_hist[276]` (done flag) and
  `smem_hist[277]` (L) instead — still above the 256 fine bins, below the
  kNumBins>=512 floor (asserted in the ctor). s_iscalars was not used for
  L/cont because slot [5] is documented as live cluster-handoff state.
- **Branch structure for byte-identical OFF**: instead of wrapping the
  original one-shot body in an `else:` (would re-indent ~140 lines), the new
  code is a preceding `if cutlass.const_expr(exact and p4_finebin_loop):`
  branch and the original `if cutlass.const_expr(exact):` became `elif
  cutlass.const_expr(exact):` — the original body text is untouched; with the
  flag OFF the first branch const-folds away and the elif is equivalent to
  the original if.
- **Battery case 1 (flag OFF == baseline) check form**: raw output
  bit-equality is unachievable — the SNAPSHOT baseline itself is run-to-run
  nondeterministic in output ORDER (atomicAdd rank-scatter order; measured:
  base-vs-base torch.equal False, sorted index sets equal). Replaced with the
  design's actual contract "same code emitted", verified as (a) per-row
  sorted index-set bit-equality vs the snapshot AND (b) byte-equality of the
  emitted PTX modulo the mangled kernel name (which embeds the package name
  gvrpkg vs gvrpkgf1). Verified OFF norm-PTX == snapshot norm-PTX.
- **Battery case 3/4 pass criterion vs design's "must be exact"**: the design's
  correctness contract assumes "cand set ⊇ top-K by construction" — falsified
  on some planted adversarial rows: the P1-P3 ADMISSION (untouched by F1) can
  give up in the huge count-plateau between the high cluster (0.9+) and the
  planted pair (~0.45-0.75), leaving cand_count = K-1 (undershoot; output gets
  one -1 pad and the boundary value is missing BEFORE P4 runs). This is
  hint-dependent (pre_idx RNG) and reproduces IDENTICALLY with the flag OFF
  and on the pristine snapshot build. The battery therefore classifies a
  flag-ON planted-row failure as an F1 failure ONLY if the snapshot baseline
  is exact (or fails differently) on the same row; identical baseline failure
  = pre-existing admission miss, counted as pass with an explicit
  `admission-miss(pre-exist)` tally. A `base-defect-fixed` tally counts rows
  where the baseline one-shot is inexact but flag-ON is exact (the fix in
  action).
- **Level exercise instrumentation (battery §5)**: inferred from constructed
  data (planted same-fine-bin pairs require >=2 levels; 1-ULP pairs terminate
  via the ULP floor) rather than a device debug counter, per the design's
  "else infer from constructed data" allowance. Additionally cross-checked on
  the host by replaying the published-level recurrence in numpy for the
  planted cases.
