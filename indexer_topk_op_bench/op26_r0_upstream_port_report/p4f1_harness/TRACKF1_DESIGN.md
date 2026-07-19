# F1 — iterative fine recursion with ULP floor (design spec, pinned)

Target file: `p4f1_harness/gvrpkgf1/top_k/gvr_topk_decode.py` (copy of the
PR#16457 snapshot). All edits marked `# [p4f1]`. Everything below refers to
`phase4_rank_scatter` (def at ~line 1877) — its `enable_p4_rank_scatter_exact`
branch (`if cutlass.const_expr(self.enable_p4_rank_scatter_exact):`, ~line 2008).

## Flag (constructor)

Add `p4_finebin_loop: Optional[bool] = None` to `GvrTopKKernel.__init__`
(next to `p4_fuse_mmz`/`dist_p4`-era flags). Resolve:
`self.p4_finebin_loop = bool(p4_finebin_loop) if p4_finebin_loop is not None else False`.
Require: if set, `enable_p4_rank_scatter_exact` must be True (assert like
dist_p4 does). **Default OFF must be byte-identical**: wrap ALL new device
code in `if cutlass.const_expr(self.p4_finebin_loop):` with the ORIGINAL
code in the else (or structure so that OFF compiles the original text).
Also thread the flag through the compile key in `launch`/cache-key code
(find where flags like `enable_p4_rank_scatter_exact` enter the JIT cache
key at ~line 3714 `key = (logits.dtype, ...) + tuple(sorted(cfg.items()))` —
pass via cfg the same way existing flags are passed; mirror exactly how
`skip_h1`-era flags were threaded in gvrpkg35/36 if visible, else how
`enable_p4_rank_scatter_exact` reaches the constructor).

## Algorithm (replaces the ONE-shot fine recursion when flag ON)

Definitions from the existing code (keep identical expressions — bin
classification MUST be bit-reproducible across passes):
- coarse: `cb = Int32((v - bmin_r) * inv1)` clamped to [0, kBins-1]
- level-k fine: `sb_k = Int32((v - f_lo_k) * finv_k)` clamped to [0, 255]
- level 0: `f_lo_0 = bmin_r + Float32(b_star)/inv1`, `finv_0 = (255.99)*inv1`
  (== existing `f_lo`, `finv`)
- level k+1: `f_lo_{k+1} = f_lo_k + Float32(sb_star_k)/finv_k`,
  `finv_{k+1} = Float32(255.99) * finv_k`

Constants: `MAXL = 4` fine levels total (level 0 .. 3). Effective resolution
2^10·2^32 — combined with the ULP floor, termination is unconditional.

SMEM: reuse `smem_hist` slots ABOVE the 256 fine bins (kNumBins >= 512 in
every spec, see comment at ~line 783). Layout (Int32 slots, floats stored
as bits via `float_as_uint32` and read back with the existing
`llvm.bitcast` idiom used at ~line 1922):
- `smem_hist[256 + 4k + 0]` = f_lo_k bits
- `smem_hist[256 + 4k + 1]` = finv_k bits
- `smem_hist[256 + 4k + 2]` = sb_star_k
- `smem_hist[256 + 4k + 3]` = ra_k  (count of candidates STRICTLY above the
  level-k straddling bin, i.e. the existing `rank_above_fine` of that level)
- `smem_hist[272 + k]` = per-level mid scatter counters (zeroed before scatter)
- `s_iscalars[5]` (or another free slot) = L = number of fine levels actually
  used (1..MAXL); `s_iscalars[6]` = loop-continue flag (block-uniform).
  CHECK s_iscalars capacity first; if only 5 slots exist, keep L and the
  continue-flag in smem_hist[271]/[270] instead.

Loop (all threads execute together; every barrier is reached by all
threads — conditions are read from SMEM scalars AFTER a barrier so they are
block-uniform):

```
lvl = 0; publish f_lo_0/finv_0 (thread0) ; barrier
while True:                                  # at most MAXL iterations
    zero smem_hist[0..255]; barrier
    build 256-bin hist over candidates in the CHAIN:
        cb == b_star AND sb_j == sb_star_j for all j < lvl
        (recompute sb_j from published f_lo_j/finv_j — identical exprs)
    barrier
    fine 3-step search seeded at ra_{lvl-1} (level0: seeded at rank_above)
      -> sb_star_lvl, ra_lvl   (this is EXACTLY the existing fine-search
         code; parameterize the seed value)
    thread0: cnt_straddle = smem_hist[sb_star_lvl]  (hist intact after search)
             need_more = (ra_lvl + cnt_straddle > kK)
             width = 1.0 / finv_lvl
             ulp_floor = fmax(|f_lo_lvl|, 1e-30) * 2^-23
             cont = need_more AND (width > ulp_floor) AND (lvl+1 < MAXL)
             publish sb_star_lvl, ra_lvl, cont, and (if cont) f_lo_{lvl+1},
             finv_{lvl+1}
    barrier
    if not cont: L = lvl+1; break
    lvl += 1
```

Scatter (single pass over candidates, replaces the existing 3-class pass):

```
for each candidate v (grid-stride as existing):
    cb = coarse bin
    if cb > b_star:
        pos = atomicAdd(cnt_above); if pos < kK: write        # class A
    elif cb == b_star:
        placed = False
        for k in 0..L-1:                                       # unrolled to MAXL,
            sb = level-k bin of v                              # guarded by k < L
            if sb > sb_star_k:
                o = atomicAdd(mid_counter_k)
                pos = base_k + o                                # base_0 = rank_above
                if pos < kK: write                              # base_k = ra_{k-1} (k>=1)
                placed = True; break
            elif sb < sb_star_k:
                placed = True; break                            # below boundary: drop
            # sb == sb_star_k -> descend
        if not placed:                                          # survived all L levels
            o = atomicAdd(straddle_counter)
            pos = ra_{L-1} + o
            if pos < kK: write
```

`filled` for the tail padding = min(kK, ra_{L-1} + cnt_straddle_{L-1}) —
same as existing but with the final level's values.

DSL notes (match existing idioms exactly):
- dynamic `while` with SMEM-scalar conditions is used elsewhere in this
  file; if the DSL fights the `while True/break` form, unroll with
  `for lvl in cutlass.range_constexpr(MAXL):` + a block-uniform `done`
  guard (skip-body pattern), which is the safest form.
- inner per-element level walk: unroll `for k in range_constexpr(MAXL)`
  with `if k < L_dyn and not placed:` guards (Int32 flags, no python bool).
- no new barriers inside per-element loops; barriers only between phases.
- do NOT touch the APPROX branch or the flag-OFF exact branch.

## Correctness contract

- Flag OFF: byte-identical SASS-level behavior (same code emitted).
- Flag ON: for every row, output is a valid top-K value-multiset of the
  candidates (== torch.topk value-set of the row, since P1-P3 admission is
  unchanged and cand set ⊇ top-K by construction). The ULP floor makes any
  residual straddle bit-identical values — any order is value-set exact.

## Battery (implementer runs; must pass before handing back)

`p4f1_harness/battery_f1.py` (write it; mirror op36 variant/battery_a2.py
style: build kernel via the package's own launch/pick_config path):
1. flag OFF == baseline: same inputs (random, 3 K's × N∈{4096, 65536,
   262144}), outputs bit-equal to snapshot gvrpkg build.
2. flag ON random: value-set exact vs torch.topk, same shapes, cs∈{1, >1
   if launch picks it}.
3. planted adversarial: rows where positions K-1 and K hold values with gap
   = (candrange/(kNumBins·256))/2 (same fine bin) — must be exact; sweep
   the pair location across coarse-bin edges.
4. deep-tie: 64 values within one fine bin straddling K (forces ≥2 extra
   levels); 1-ULP pairs; all-equal row (ULP floor path); K exact-count row
   (cand_count == kK early path untouched).
5. Report: pass counts + which levels were exercised (instrument via a
   debug counter if trivial, else infer from constructed data).

Env for running (this node): PYTHONNOUSERSITE=1, PYTHONPATH=
/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450,
CUDA_VISIBLE_DEVICES=<one idle GPU>.
