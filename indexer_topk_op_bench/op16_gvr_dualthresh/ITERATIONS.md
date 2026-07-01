# op16_gvr_dualthresh — Iteration log

Base op: **GVR cuteDSL rank-scatter P4** (op#7) =
`p4_recursive_digit/src/gvr_topk_decode_p4.py` → `src/gvr_topk_decode_dt.py`.
Wrapper mirrors `harness/gvr_cutedsl_rs_op.py`. HW: **B300 sm_103**.

User goal: two-threshold P2 (add `threshold_1` with count `M<K` ⇒ those M are
definite top-K winners), so P3 collects only the band `[threshold, threshold_1)`
and P4 refines only `M0-M` elements with a cheap select (bitonic/insertion).
Target: >95% of report cases beat Radix-cuteDSL AND SGLang, +40% avg per seqlen.

---

## Iter 0 — Baseline characterization + decisive P2-tax measurement (B300)

**Strategy**: before building, measure whether the mechanism can pay. Used the
clock64-instrumented rank-scatter kernel (`harness/measure_cute_phases_rs.py`)
with a `kC`/`kFTarget` override subclass (`scripts/p4_scaling.py`) to isolate
P1–P4 CYCLES as the candidate count shrinks (smaller kC ⇒ tighter threshold ⇒
fewer candidates). clock64 tot ≈ cold wall at these sizes (single-CTA,
18464 cyc ≈ 10µs ≈ the 9.7µs cold sweep number), so cycle deltas ≈ wall deltas.

### Results — K=512 fp32 cr=4, P4 cycles vs (kC, eval-optimal kFTarget)

| N | (kC,kFT) | P1 | P2 | P3 | P4 | tot | vs base |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | (5120,512) base | 3491 | 3874 | 1741 | 7050 | 16349 | — |
| 4096 | (1536,1280) | 3163 | 5152 | 1632 | 6078 | 16107 | −1.5% (wash) |
| 4096 | (1024,1024) | 3114 | 5132 | 1736 | 5959 | 16124 | −1.4% (wash) |
| 4096 | (768,640) | 3124 | 6881 | 1671 | 5997 | 17732 | +8% loss |
| 16384 | (5120,512) base | 3682 | 4425 | 3196 | 7120 | 18567 | — |
| 16384 | (1024,1024) | 3258 | 6085 | 3142 | 5823 | 18530 | −0.2% (wash) |
| 65536 | (5120,512) base | 3656 | 11030 | 9208 | 8656 | 32735 | — |
| 65536 | (1024,1024) | 3816 | 15999 | 8880 | 6972 | 35863 | +9.5% loss |

All variants EXACT (value-equiv torch.topk, uniq==K).

### Decisive findings
1. **P4 is only ~15% count-reducible at small N** (7050→5959 as cand 2K→1K);
   it only collapses (→~260) when cand→K, which needs kC≈K.
2. **Tightening the threshold explodes P2 by 1.3–2.5×** — the eval tax. With
   eval-optimal kFTarget the tax ≈ the P4 saving at small N (WASH) and EXCEEDS
   it at N≥65536 (LOSS). Exactly reproduces op13 on the rank-scatter P4.
3. The **band cannot be shrunk without the tax**: band = cand − c_hi ≥ K − c_hi,
   and c_hi < K, so band ≤ 256 requires cand ≤ c_hi+256 < 768 ⇒ kC≈K ⇒ max tax.

### Theoretical ceiling of the FULL two-threshold idea (free band-select P4)
Even assuming the band-refine P4 is FREE (replace rank-scatter with ~300cyc
single-warp select) at the tightest measured kC=768:

| N | tot with free select | vs baseline |
|---:|---:|---:|
| 4096 | 3124+6881+1671+300 = 11976 | **−27%** |
| 16384 | 3366+9757+3166+300 = 16589 | **−11%** |
| 65536 | 3729+20087+8393+300 = 32509 | **−0.7%** |
| ≥131072 | P2 tax explodes, P4 already ~small | **LOSS** |

⇒ Best-case ceiling: mild mid-N gain, decaying to zero by 65K, loss beyond.

### Why the +40%-everywhere TARGET is structurally unreachable (independent of P4)
- **Small N (4K):** best-case −27% → gvr 9.7→7.1µs, but Radix=6.45µs. Radix's
  floor is BELOW GVR's floor; cannot beat Radix here, let alone by 40%.
- **Large N (131K/262K):** GVR is P2+P3 (full-N streaming) bound (60–70%); at
  262K gvr=35µs vs Radix flat 19µs; even P4→0 leaves ≫Radix. A threshold-
  streaming kernel cannot beat flat radix-select here. (~26/60 cases.)
- The two-threshold idea attacks P4, which only dominates at small/mid N.

### Verdict
The user's specific mechanism (add threshold_1 in P2 via secant) is **tax-bound**
like op13/op14: net wash at small/mid N, loss at large N. The stated
+40%-over-Radix-at-95%-of-cases target is **not physically reachable** by any
P4-side optimization. The only lever that could unlock the P4 collapse is a
**cheaper P2** (pin a tight threshold in far fewer full-N passes — sampling /
better init / better root-finder), op13's unbuilt H2.

### Next action
Report to user with data + options: (A) build the achievable mid-N band-select
version (~10–25% mid-N, sub-target); (B) pursue cheaper-P2 lever (uncertain,
larger); (C) accept target unreachable and stop. Await direction.

---

## Iter 1 — PIVOT to cheaper-P2 (sampling-quantile init); host validation (B300)

User chose: pursue the cheaper-P2 lever (the only path with real upside).

**Idea**: replace P2's iterative full-N secant (2–3.67 passes) with a SMEM
histogram of a small strided subsample (n_s=4096, N-independent) → estimate the
K-th-value quantile → ONE full-N confirm pass; existing secant is a rare
correction fallback. Attacks P2 *compute* (fewer element-comparisons), the real
large-N cost (op14: HBM is moot, input fits L2). Distinct from op14 compaction.

**Host prototype** (`scripts/host_sample_p2.py` + histogram check), searchsorted
counts, across full grid × 3 beta cfgs × 3 seeds:

| metric | baseline secant | sampled-hist init |
|---|---|---|
| full-N passes (P2) | 2.0–3.67 | **1.00** everywhere |
| cand entering P4 (K512) | 1706–2627 (~3–5×K) | **513–589 (~1.1×K)** |
| cand entering P4 (K1024) | 2659–4137 | **~1090–1180 (~1.1×K)** |
| exactness (value-equiv topk) | — | **9/9 all cells, fp32/bf16/fp16** |
| compute_ratio (samp/base) | 1.0 | **0.51–0.63 (K512), 0.28–0.38 (K1024/2048)** |

Histogram-of-sample t0 (kernel-realistic, B=512/1024/2048 bins) reproduces
1-pass + cand~1.1×K + 9/9 exact — the kernel design is de-risked. aim_mult 1.10
stays exact everywhere (undershoots self-correct, never inexact).

**Two wins at once, no tax:**
- P2: 2–3.67 passes → 1 (large-N compute saving).
- cand ~1.1×K (vs 2–5×K) → triggers the P4 collapse (iter-0: cand→K makes
  rank-scatter P4 ~260 cyc vs ~7000–8600) — the tight threshold now costs 1 pass,
  not the op13 tax.

**Projected (from iter-0 cycles, to be confirmed by nsys):** ~−40% to −60%
small/mid N, ~−40% to −45% large N. This would flip many large-N Radix losses to
wins. (Corrects the iter-0 "structurally unreachable" verdict, which was for the
*two-threshold* mechanism; sampling breaks the tax.)

### Next action
Build sampled-histogram P2 in the kernel (minimal: replace P2 init threshold with
sampled t0; secant stays as fallback; reuse smem_hist for the sample histogram;
gate behind `enable_sampled_init`). Then nsys cold-L2 A/B (report protocol).

---

## Iter 3 — nsys cold-L2 A/B (REPORT protocol) — PREMISE FALSIFIED again (L2)

Ran `scripts/nsys_ab.py` under nsys (512MB evict + eager + NVTX-range sync +
nvtx_kern_sum, report-identical). ANCHOR check: re-measured gvr_cutedsl_rs vs
report gvr_cutedsl_rs_cold_us = 0.976–1.020 ⇒ protocol comparable to report.html.

### K512 fp32 cr=4 (B300, pure-kernel cold-L2 µs)

| N | rs(base) | op16 | radix | sglang | op16/rs | op16 vs radix |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 9.81 | 11.26 | 6.47 | 7.64 | **0.87** | 0.57 |
| 8192 | 9.48 | 12.06 | 7.66 | 8.35 | **0.79** | 0.64 |
| 16384 | 10.12 | 14.08 | 11.83 | 11.03 | **0.72** | 0.84 |
| 32768 | 12.76 | 15.68 | 18.66 | 15.46 | 0.81 | 1.19 |
| 65536 | 16.97 | 17.21 | 18.76 | 24.17 | 0.99 | 1.09 |
| 131072 | 23.02 | 21.99 | 18.80 | 41.46 | 1.05 | 0.86 |
| 262144 | 35.44 | 33.09 | 18.65 | 75.81 | 1.07 | 0.56 |

op16 beats BOTH baselines: 1/7 (only N=65536, same as baseline's mid-N win).

### Why the host projection (−40 to −60%) was WRONG — the op14 L2 trap, again
- host counted LOGICAL full-N passes; but on B300 the fp32 input (≤1MB) fits L2
  (126MB) ⇒ the baseline's 2–3 secant `count_ge` re-reads are **L2 hits, cheap**.
  Cutting them saves little (op14's decisive lesson, re-confirmed).
- the sampling init is **pure ADDED work**: 4096 strided global reads + 4096 SMEM
  atomic incs + suffix scan. At small/mid N this exceeds the P2 saving → NET
  SLOWER (0.72–0.87×). At large N the sample amortizes → small net win (+5–7%),
  but that is the residual P2-compute saving, not the projected 3×-pass cut.
- P4 collapse (cand 2K→1.1K) is muted: rank-scatter P4 is floor-bound at small N
  (iter-0), so the tighter cand barely helps where the sampling costs most.

### Structural wall stands (iter-0), independent of P2 method
Even at large N where op16 nets +5–7% over its own baseline, it still LOSES to
radix (0.56–0.86×): radix is flat ~18–19µs while op16's P3 full-N collect alone
is ~17µs @262K. No threshold-streaming GVR variant can match flat radix-select at
large N. And small-N radix floor (6.5µs) is below GVR's ~9µs floor.

### Verdict
op16 sampled-init: **NO-SHIP**. It is net SLOWER at small/mid N (added sampling
cost > L2-cheap pass saving) and only +5–7% at large N (still far behind radix).
The +40%-over-Radix / 95%-of-cases target is confirmed **physically unreachable**
for any GVR threshold-streaming variant in the DSv4 decode regime (N≤262144,
input ⊂ L2). Both attack surfaces (P4 via two-threshold; P2 via sampling) are now
empirically exhausted on B300.

### Next
Run full nsys grid (other K/dtype) to confirm the trend is universal, then final
report + HEAD at baseline (no ship).

---

## Iter 4 — PIVOT to secant-framework opt (user constraint) — host analysis

User constraint: stay in the "secant bracket-then-refine" structure; optimize
the ITERATION only (no sampling replacement). Also clarified algo-A: don't shrink
the band; get threshold_1 in the SAME secant run — either (1) single-CTA, reuse
the iteration path's intermediate thresholds, or (2) a 2nd CTA searching
threshold_1 in parallel, write partial top-K, sync to the main CTA.

### (a) Secant convergence-acceleration variants (`scripts/secant_variants.py`)
Mean full-N count_ge evals across grid x 3 cfgs x 3 seeds (all EXACT 216/216):
- base (linear secant): 2.602
- **quad (inverse-quadratic on last 3 pts, secant fallback): 2.509** ← best, stable
- illinois: 2.889 (regresses K2048), pquantile init: 3.940 (backfires, per op13)
quad helps K1024 large N (3.67->2.89, 2.78->2.44); K512/K2048 ~flat. Small (-3.6%
evals overall) but NO regression. => adopt quad as the interpolation.

### (b) FREE threshold_1 from the single-CTA secant path (`scripts/host_thr1_free.py`)
threshold_1 = largest path threshold with count<K (=> M definite winners peeled
for FREE — the secant already evaluated these points):

| K | M(free) | M/K | band=M0-M | band/M0 |
|--:|--:|--:|--:|--:|
| 512 | ~360 | 0.70 | 1.4-2.3K | 0.80-0.86 |
| 1024 | ~750 | 0.71-0.81 | 2.0-3.4K | 0.71-0.83 |
| 2048 | ~1.5-1.9K | 0.74-0.92 | 0.4-1.0K | 0.19-0.38 |

- Free peel gives M~0.7-0.9xK definite winners at ZERO extra pass.
- BUT band = M0-M stays LARGE for K512/K1024 (wide accept window kCC=10K => M0
  2-4K). rank-scatter P4 is floor-bound (iter-0) => peel saves only ~13% of P4.
- K2048: M0 small + M large => band 0.4-1K (band/M0 0.2-0.4) => real P4 headroom.

### Projected (free peel + quad, NO tax, NO regression)
K2048 large N ~-15%, K512/K1024 ~-5%. Shrinking K512/K1024 band needs a small M0
(tight threshold) = the tax; only the 2-CTA path can hide threshold_1's search,
but not the main CTA's tightening tax.

### Next action
Build "Scheme X" = quad interpolation + single-CTA free threshold_1 peel + band
via existing rank-scatter (P3 classifies >=thr1 -> direct output[0:M], [thr,thr1)
-> smem band; P4 selects top-(K-M) on band). All free/no-tax. nsys-confirm K2048
gain + K512/K1024 no-regression, then decide on the 2-CTA path.

---

## Iter 5 — BUILD Scheme X (secant framework, user constraint) — EXACT

Free threshold_1 peel + band rank-scatter (`enable_dual_thresh`):
- P2 secant records free threshold_1 (s_thr[3], M=s_iscalars[5]) = largest path
  threshold with count<K (definite winners, zero extra pass).
- `phase4_dual` dispatcher → `phase4_partition` (register-staged: winners >=thr1
  → output[K-M:K]; band <thr1 → smem_keys[0:band]) → `phase4_rank_scatter`
  (target_k=K-M) fills output[0:K-M]. All P4 methods @cute.jit (required).
- **Exactness: baseline 3/3 + dual 81/81 OK** (fp32/bf16/fp16 × K512/1024/2048 ×
  N × 3 cfgs). smem s_thr[4]/s_iscalars[6]. Baseline byte-identical (flag off).
- CuTe DSL decorator gotcha cost 5 turns (see LEARNINGS / learnings yaml).

## Iter 6 — nsys A/B (report protocol) — Scheme X NET NEUTRAL

K2048 fp32 (anchor 0.99–1.02 comparable to report):
| N | rs | op16X | X/rs | X vs radix |
|---:|---:|---:|---:|---:|
| 8192 | 11.51 | 12.12 | 0.95 | 0.65 |
| 32768 | 16.08 | 15.72 | 1.02 | 1.29 |
| 65536 | 21.01 | 21.59 | 0.97 | 0.92 |
| 262144 | 48.96 | 49.60 | 0.99 | 0.40 |

**Scheme X ≈ baseline (X/rs 0.95–1.02×).** The free-peel P4-collapse saving is
eaten by phase4_partition overhead (M0-wide 2-pass register staging + smem-atomic
slot contention on 2 counters over ~M0 threads). vs radix still lost at large N
(structural). NO-SHIP as-is.

### K512 fp32 (resumed run — anchor 0.939–1.019 comparable to report)
| N | rs | op16X | radix | sglang | X/rs | X vs radix |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 9.13 | 10.22 | 6.08 | 7.15 | **0.89** | 0.60 |
| 8192 | 9.38 | 10.85 | 7.73 | 8.28 | **0.86** | 0.71 |
| 16384 | 10.18 | 11.22 | 11.87 | 10.93 | **0.91** | 1.06 |
| 32768 | 12.74 | 14.02 | 19.77 | 15.37 | **0.91** | 1.41 |
| 65536 | 17.20 | 18.01 | 19.57 | 23.97 | **0.96** | 1.09 |
| 131072 | 23.40 | 23.90 | 19.11 | 41.15 | **0.98** | 0.80 |
| 262144 | 35.34 | 37.03 | 19.19 | 75.46 | **0.95** | 0.52 |

**K512 Scheme X is NET SLOWER at EVERY N (X/rs 0.86–0.98×)** — worse than K2048,
as predicted for the larger K512 band. Root: at K512 M0≈1.8–2.6K but M(free)≈360,
so P4 still rank-scatters a ~1.4–2.3K band (only cand-count ~15% less than baseline
M0) — the rank-scatter floor barely moves — while phase4_partition adds a full
M0-wide 2-pass + atomic contention on top. The overhead is a larger fraction at
small/mid N (0.86–0.91 @ 4K–32K), decaying toward 1.0 only as N grows and P2+P3
dominate. Data: `results/nsys_ab/abX_K512_fp32.jsonl` + `nsys_reps/abX_K512_fp32.nsys-rep`.

### Full K×dtype grid (all 9 configs, X/rs = Scheme X vs own baseline; anchors 0.94–1.03)
Per-N X/rs (`<1` = slower). Data: `results/nsys_ab/abX_K{512,1024,2048}_{fp32,bf16,fp16}.jsonl`.

| N | K512 f32 | K512 bf16 | K512 f16 | K1024 f32 | K1024 bf16 | K1024 f16 | K2048 f32 | K2048 bf16 | K2048 f16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 0.89 | 0.89 | 0.89 | 0.88 | 0.92 | 0.91 | — | — | — |
| 8192 | 0.86 | 0.87 | 0.87 | 0.86 | 0.84 | 0.86 | 0.95 | 0.92 | 0.91 |
| 16384 | 0.91 | 0.88 | 0.91 | 0.87 | 0.85 | 0.86 | 0.97 | 0.92 | 0.91 |
| 32768 | 0.91 | 0.91 | 0.90 | 0.96 | 0.97 | 0.95 | **1.02** | 0.94 | 0.94 |
| 65536 | 0.96 | 0.95 | 0.95 | 0.91 | 0.91 | 0.90 | 0.97 | 0.95 | 0.94 |
| 131072 | 0.98 | 0.96 | 0.97 | 0.94 | 0.93 | 0.94 | 0.98 | 0.95 | 0.95 |
| 262144 | 0.95 | 0.96 | 0.96 | 0.96 | 0.95 | 0.95 | 0.99 | 0.97 | 0.96 |

**Universal: X/rs ∈ 0.845–1.022; exactly ONE cell >1.0 (K2048 fp32@32K, 1.02×), all
others <1.** (1) smaller K worse (larger band ⇒ rank-scatter floor barely collapses);
(2) bf16/fp16 consistently slightly worse than fp32 (partition overhead dtype-independent,
baseline a touch cheaper); (3) large N → 1.0 (P2+P3 dominate) but never wins. Confirms
the iter-6 fp32 read is not a dtype/K artifact.

### FINAL VERDICT (iter 6 complete)
Every op16 lever (P4 two-threshold band-shrink, P2 sampling, secant accel, free-peel
Scheme X) nets neutral-to-NEGATIVE on B300: theoretical gains eaten by L2-fit +
rank-scatter/launch floor + implementation overhead. K2048 neutral (0.95–1.02×),
K512 regresses (0.86–0.98×). The >95%-cases / +40%-over-Radix target is
**structurally unreachable** in the DSv4 decode regime (N≤262144, input ⊂ L2).
Scheme X is EXACT + no-regression on the baseline path (flag off) but a net loss
with the flag on ⇒ **NO-SHIP**. **HEAD stays at baseline.**

One untried refinement (not pursued — ceiling too small to matter): replace
phase4_partition's smem-atomic slot allocation with a warp-aggregated / prefix-sum
compaction. Expected to help K2048-large-N only; cannot fix the K512 regression
(K512's large band barely collapses P4 even with a *free* partition — iter-0
free-select ceiling was only mild mid-N), and does not change the structural wall
vs flat radix-select. Recommendation: close op16, keep the exact dual-thresh code
gated-off for reference, spend effort on op13's cheaper-P2 lever instead.
