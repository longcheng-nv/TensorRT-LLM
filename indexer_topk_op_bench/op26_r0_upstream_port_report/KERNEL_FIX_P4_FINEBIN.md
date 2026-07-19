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

## 6. F4 (evaluated, REJECTED): log-transform before the fine recursion

Proposal (2026-07-19 review): map candidates v → log-domain before the
coarse+fine linear binning, keeping the existing single 256-sub-bin refine.

**Verdict: does NOT guarantee exactness, and does not even fix the observed
cells.** Measured on the 4 diagnosed failing rows (log-domain pair distance
= |ln(v_kth/v_neighbor)| vs log-domain fine-bin width = ln(seg_max/seg_min)
/ (kNumBins·256), sign-segmented):

| cell | pair log-gap | log fine-bin | verdict |
|---|---|---|---|
| pro/128k/L6 | 1.26e-05 | 1.66e-05 | SAME BIN — still inexact |
| pro/512k/L48 | 3.74e-05 | 4.75e-05 | SAME BIN — still inexact |
| v32/8k/L8 | 1.96e-06 | 1.08e-05 | SAME BIN — still inexact (5.5× short) |
| v32/64k/L16 | 2.38e-06 | 7.17e-06 | SAME BIN — still inexact (3× short) |

Why it cannot work in principle:

1. **Pigeonhole**: ANY fixed monotone transform + a fixed 2^18–2^19-bin
   two-level histogram partitions fp32's ~2^30 candidate-representable
   values into ≤2^19 classes; some class contains ≥2 distinct values, and
   real/adversarial data can (and does) place the K boundary inside it.
   A static re-measure (linear→log→anything) only moves WHICH pairs
   collide; it cannot eliminate collisions.
2. **Log specifically trades small-|v| for large-|v| boundaries**: log gives
   uniform RELATIVE resolution. The v32 failing pairs sit at |v|≈4–5 where
   their relative gap (2e-6) is far below the relative bin width — log makes
   those strictly worse than linear would need. Meanwhile candidates near 0
   (pro/512k has |v| down to 3e-5) inflate ln(max/min) to ~12, widening
   every log bin.
3. **Engineering cost is not free either**: logits are signed (v32 boundary
   values are negative) → sign-split segments; log/ex2 = MUFU per candidate
   × 2 passes (hist + scatter) over kC≈5–6K candidates — a real add on the
   hot path, paid on 100% of rows to not fix the 1%.

Note the "log-like binning done right" is exactly binning by the top bits of
the monotonic u32 order key (sign+exponent+mantissa-high) — i.e. option F3's
first radix digit. It inherits the same pigeonhole limit per pass; what makes
F3 exact is the guaranteed ≤4-pass recursion, not the transform.

## 7. On the performance concern with F1/F2

Historical regressions in this area (p4_fused_hist −15%, global kNumBins
512, kNumBins=256) were UNCONDITIONAL structural changes paid on every row.
F1/F2 as specified here are **block-uniform conditional** paths: the extra
recursion/tail-select fires only when `cnt_straddle > need` — a whole-block
branch on an SMEM scalar that is false on ~99% of real rows (and ~100% of
synth rows). The 99%-path cost is one extra SETP/branch on a value already
in SMEM + icache footprint; the 1%-path pays ~1–2µs. If even that footprint
measures as a regression (op32 showed small-N cells are icache/issue-bound),
the fallback is F2 with the tail-select hoisted into a separate __noinline__
device function (cold section), or a compile-time flag default-ON only for
K∈{1024,2048} where the failures occur.

## 8. F1 implementation campaign — measured verdict (2026-07-19, b200-027)

Implemented behind default-OFF flag `p4_finebin_loop` in `p4f1_harness/gvrpkgf1`
(agent-implemented from this doc + iterated v2→v4; battery grew to 164 cases).
OFF is PTX-byte-identical to the snapshot (modulo mangled kernel name);
baseline output order on straddle rows is atomic-arrival nondeterministic
(proven base-vs-base), so bit-equality is asserted on deterministic rows +
PTX identity.

Gates (all on real captures, launch contract):
- A: 9 fixtures ON exact 5/5 each, OFF negative control 0/5 each — PASS.
- B: battery 164/164 (planted same-fine-bin pairs, 1-ULP ladders forcing
  deep levels, all-equal ULP-floor, CAP=128 boundary + CAP+1 fallback) — PASS.
- D: full 865-cell all-layer grid ON: 865/865 exact (9/865 -> 0) — PASS.
- C (nsys x3-round median, paired same-process A/B): FAIL at the
  "no cell > 1.025" bar in every variant:

| variant | structure | bench (25 cells, 0 fire) | fixture (9 cells, fire) |
|---|---|---|---|
| v2 | compile-time iterative fine recursion (replaces one-shot) | gm 1.0254, max 1.049 | gm 1.169 |
| v3 | + scatter-integrated scratch (store in hot branch) | gm 1.046 (hot-path store tax) | gm 1.067 |
| v4 | zero hot-path deltas; separate collect pass; deep section DELETED (PTX +6.5%) | gm 1.0335, max 1.061 | gm 1.104 |

Attribution (silicon printf): 0/25 real bench rows fire need_more (every
real row has exactly 1 straddle element) — the bench tax is NOT tail work.
NCU: registers 86(OFF)->80(ON), occupancy limits unchanged — not registers.
v4 vs v2 PTX delta (6.5% vs ~24%) did not move the tax — not primarily
icache either. Residual attribution: wrapping the original scatter in a
dynamic (block-uniform) branch changes codegen of the hot loop + BS=1
issue-bound layout sensitivity (op32). ~2-5%/cell is the irreducible cost
of RUNTIME tie detection in this structure.

**Conclusion**: exactness is fully achieved (gates A/B/D), but not for free
(gate C). Decision forked to the user: (a) accept ~3% BS=1 tax, default-ON;
(b) merge default-OFF as opt-in exactness + document the baseline contract
("exact up to same-fine-bin boundary ties, ~1% of real layer-cells, error
magnitude one boundary element at ~1e-5"); (c) F3 radix-select P4 rewrite
(structural; potentially tax-free but unproven — a new campaign).

## 9. RESOLUTION (2026-07-19): already fixed upstream — no PR needed

Port preparation exposed the decisive fact: the defect does NOT exist at the
shipped PR#16457 head. `@eae374554c` ("[None][fix] GVR top-K decode: exact
tie resolution in the P4 straddling fine bin", 2026-07-16) added
`p4_exact_tail` — an **ambiguity-gated MSB-first 8-bit radix select over
f32 order keys** on the straddling-fine-bin tie set (rewrites
[rank_above_fine, kK); default ON for fp32 rank-scatter-exact; hot path
pays two scalar compares). That is precisely this doc's option F3, gated
the way v2-v4 tried and failed to make cheap — and its cost was already
absorbed in the §9b NEW/OLD re-measure (gm 0.9943, no regression flag).

Verified on silicon today (standalone `p4f1_harness/gvrpkgprod` build of the
shipped file): the 9 fixtures exact 5/5 each; the FULL 865-cell all-layer
grid 865/865 exact.

Scope correction to §0/§2 of this doc and to REPORT §4b: the 9/865
inexactness belongs to the **pre-vseed measurement arm** (@018251950f — the
bench snapshot the per-layer sweeps ran, as the §4 headnote states), NOT to
the shipped kernel.

F1 campaign disposition: v2-v4 + 164-case battery + gates stand as an
independent re-derivation and third-party validation of the upstream fix
(and of why the naive structures cost ~3% — the shipped ambiguity-gate is
the right shape). Nothing to merge; no follow-up PR; F3 project moot
(p4_exact_tail IS F3-lite).

Process lesson (recorded to memory): a defect diagnosed on a pinned bench
snapshot must be REPRODUCED ON THE SHIPPED HEAD before any fix campaign —
one 10-minute fixture run would have saved the whole F1 implementation
loop. The snapshot's very first "earlier fixes" table in §1 mislabeled
`p4_exact_tail` as "exact-tail scratch/rank handling" instead of reading
its diff.
