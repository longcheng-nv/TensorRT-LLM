# P4 fine-bin boundary-tie inexactness — root cause + kernel fix plan

Status: PROPOSED (2026-07-19). Affects the shipped PR#16457 GVR kernel
(`enable_p4_rank_scatter_exact` path, `phase4_rank_scatter` in
`gvr_topk_decode.py`; snapshot lines ~2010-2140). Found by the all-layer
real-data backfill (op26 REPORT §4b): 9/865 per-layer BS=1 cells inexact
(4 pro + 5 v32), always exactly ONE boundary element swapped.

## 1. What the earlier fixes fixed — and what they did not

| fix | class it fixed |
|---|---|
| `enable_p4_rank_scatter_exact` (PR#15709/#16457) | replaced the APPROX single-pass rank-scatter (arbitrary order in the straddling COARSE bin) with one fine-histogram recursion |
| `p4_exact_tail` (@eae374554c) | exact-tail scratch/rank handling on the recursion path |
| base secant undershoot repair (R0) | `gvr_base`'s <K-unique-indices failure on low-hit rows (flash 512k class) |

None of these addressed the case where the **fine** recursion itself cannot
separate two distinct fp32 values. The docstring assumption — "256 sub-bins
over bin b* gives kNumBins×256 effective resolution, enough to resolve the
straddling bin to ≤1 distinct value" — is falsified by real captures.

## 2. Root cause (code-verified, numerically confirmed)

`phase4_rank_scatter` exact path:

1. coarse LINEAR histogram: `bin = (v - cmin) * inv1`, kNumBins (1024/2048)
   bins over the candidate range `[cmin, cmax]`;
2. ONE fine recursion: 256 LINEAR sub-bins over the straddling coarse bin
   `b*`;
3. elements with `sb == sb_star` (straddling FINE bin) are written to the
   remaining output slots **in atomicAdd arrival order** (arbitrary).

Resolution floor = `range / (kNumBins·256)`. Measured on the failing rows:
candidate range 3.0–10.7 → fine-bin width **1.2e-5–3.0e-5**, while the true
K-th value and its lower neighbor differ by only **6e-6–1e-5** — both land
in the same fine bin, and step 3 sometimes emits the wrong one. Exactly one
element is affected because `rank_above_fine + cnt_straddle` overshoots K by
the tie multiplicity (here 2). Frequency on real data: ~1% of layer-cells;
never triggered on the single-layer §4 set, the 2245-class battery, or the
synthetic envelopes (their boundary gaps are wider).

Note: an fp16/bf16-collision explanation was floated first (the pairs do
collide in both) — that is correlation, not mechanism: this path has no
16-bit keys; the fine-bin width just happens to be of the same order as
16-bit ULPs at these magnitudes.

## 3. Fix options

### F1 (recommended hotfix): iterate the fine recursion to a ULP floor

Wrap the existing fine-recursion block in a loop:

```
after each fine pass over the current straddling bin:
  need = kK - rank_above_fine
  if cnt_straddle <= need:              # bin fully consumed -> done (exact)
      break
  width = 1 / finv                       # current sub-bin width
  if width <= ulp(f_lo):                 # all values in bin bit-identical
      break                              # true fp32 ties -> any order exact
  # recurse: b* <- sb_star; f_lo/finv re-derived; rebuild 256-bin sub-hist
```

- Termination: each pass gains 8 bits of resolution; from 2^18 (coarse×fine)
  two extra passes reach 2^34 divisions of the candidate range — below fp32
  ULP for any range these rows produce. The `width <= ulp` guard makes
  exactness **unconditional** (bit-identical values are genuine ties; any
  selection is value-set exact).
- Cost: the extra pass executes only when the straddling bin is unresolved
  (~1% of rows, data-dependent); one 256-slot zero + one candidate scan +
  2 barriers per extra pass — ~1µs-scale on the affected rows, byte-identical
  behavior elsewhere. No new SMEM (reuses the fine-hist slots).
- Structure: the fine block is already self-contained; the loop re-enters it
  with `(f_lo, finv, rank_above) := (f_lo + sb_star/finv, finv*256*, rank_above_fine)`.

### F2: exact tail select on the straddling fine bin

When `cnt_straddle > need`, compact the straddle elements (typically 2–32)
into scratch and let one warp select top-`need` by full fp32 order-key.
Simple and bounded, but needs a cap + fallback for pathological rows where
thousands of near-identical values straddle (all-equal-ish rows) — F1's
recursion handles those naturally, so F2 alone is not sufficient.

### F3 (structural, follow-up PR): radix-select over u32 order keys

Replace linear-value binning with monotonic `f32→u32` order-key radix
digits (8 bits, MSB-first): exact by construction in ≤4 passes, immune to
FP rounding at bin edges and to outlier-skewed ranges. Bigger diff; the
right shape if P4 is touched again (also composes with op36's dist_p4,
whose rare-ambiguity gather fallback already covers its own boundary case).

## 4. Validation plan (gates before any ship)

1. **Fixtures**: the 9 failing real cells (pro 64k/L22, 128k/L6, 512k/L48,
   512k/L60; v32 8k/L8, 8k/L39, 16k/L38, 64k/L16, 128k/L25) — must go
   value-set exact.
2. **Adversarial synth**: pairs planted at gap = fine_width/2 straddling the
   K boundary, swept over K∈{512,1024,2048} × kNumBins × cand-range scales;
   plus all-equal and 1-ULP-apart rows (ULP-floor guard).
3. **Regression batteries**: op26 2245-class full grid + the 865-cell
   per-layer grid + 275-cell BS grid (exactness folded).
4. **Perf gate**: nsys ≤2-way on the op26 §6 real+synth cells — expected
   WASH (extra pass fires on ~1% rows); any cell >2.5% regression blocks.

## 5. Scope note

`gvr_base` (secant path) and `sgl_bx` (op36) are unaffected (different
selection mechanics). op36's `dist_p4` distributed P4 carries its own
gather fallback for boundary ambiguity; port F1 there for symmetry when
PR-B lands.
