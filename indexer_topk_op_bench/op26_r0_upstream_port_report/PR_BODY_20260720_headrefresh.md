# PR #16457 body — 2026-07-20 refresh at head e6fdbfac3d
# (fresh canonical re-measure: drive_canonical.sh 77 cells x 3 arms b200-027;
#  full-grid stats: headfull sweep 2772 cells, HEADFULL_VERDICT.md)
---
## Summary
Adds an **R0 histogram-ladder admission** fast path plus a fused **rank-and-scatter** Phase-4 writeback to the production Blackwell cuTe-DSL GVR top-K decode kernel (`GvrTopKKernel`) and makes R0 the **default** (`enable_r0=True`).

R0 replaces the Phase-2 **secant** threshold search with a single-pass multi-threshold "rung ladder" admission, seeded by a 256-bin histogram over the prev-topK gathered values (P1b), and a fused rank-and-scatter Phase-4 writeback (barriers ~14 → ~7). Hardening/tuning pieces included:

- **Virtual seed rung (`r0_vseed`, default on with R0).** P1's mean probe (the secant init) is folded into the R0 count pass as one extra virtual rung column, reusing the existing per-thread count buffer (zero SMEM growth, no extra traffic or synchronization). Admission picks the tightest admissible column, and on a full miss the measured count donates an interior bracket point to the fallback refine. This adapts the ladder to each row's value distribution and fixes the cold-hint **fat-admission** regime (details under *Known limitation*). With the seed rung, K512/K1024 drop to a single static rung (`qfracs=(0.85,)`, zero column tax); K2048 keeps a two-rung ladder.
- **Exact boundary-tie resolution in Phase-4 (`p4_exact_tail`, default on for fp32).** The P4 fine recursion resolves values to the fine-bin width; distinct fp32 values spaced below that resolution which straddle the top-K boundary landed in one fine sub-bin and were kept in arbitrary arrival order — observed on real DSv4-Pro 512k-ISL captures as a 1-element miss with |dv| ≈ 3e-6. Now the only ambiguous case (straddling-bin tie set overfilling the remaining slots) is gated and re-ranked bit-exactly with an MSB-first 8-bit-digit radix select over order-preserving integer keys; all scratch is reused (zero SMEM growth). A **tiny-tie fast path** resolves the common few-element ambiguity without the full radix pass, so repair-active rows return to base parity (see *Performance*). Unambiguous rows pay two scalar compares. 16-bit kernels are byte-identical (their upconverted keys are already fully resolved; 16-bit tie plateaus are bitwise-equal, so arrival order is value-exact).
- **K2048 tuning on real-content distributions.** The K2048 low rung is recalibrated (0.85 → 0.6) against real capture value distributions, and the K2048 R0 Phase-4 histogram is trimmed 2048 → 512 bins — together lifting the real V3.2-capture axis from ~1.35× to 1.44× without touching other K.

The classic **secant path is retained verbatim** and stays reachable via `enable_r0=False`.

This PR lands the kernel capability + flips the default only. It does **not** touch the call site / dispatch (the custom op does not plumb `enable_r0`); the activation policy / dispatch guard is a follow-up PR (see *Known limitation*).

## Performance

**Methodology.** nsys pure-kernel, **cold-L2** (512 MB L2 evict outside the timed NVTX range, per timed call), single-GPU same-run A/B (R0 vs the retained secant baseline on the identical inputs), 20 cold reps. B200 (SM100). **All numbers below re-measured 2026-07-20 at the current head** (seed rung + exact-tail + tiny-tie fast path + K2048 recalibration included) with both arms driven through the kernel's own launch-shape contract (`GvrTopKKernel.launch` → `pick_config`): cluster_size 8/4/2/1 by (BS, N) — incl. the cs=8 tiny-grid pick at N≥128K — T=512/1024, mbpm tiers, fp32-only 256-bit loads; i.e. the shapes the production runner picks. All performance numbers below are on per-layer indexer top-K **captured from production DeepSeek-V4 Flash / Pro and V3.2 decode runs** (a calibrated synthetic envelope is additionally used for exactness/audit coverage; see *Correctness* and *Known limitation*).

### Data source
**Real** (production decode capture). Per-layer `(indexer logits, indexer top-K)` dumped from real **BS=1 greedy end-to-end decode** runs of DeepSeek-V4 Flash (K=512), DeepSeek-V4 Pro (K=1024) and DeepSeek-V3.2 (K=2048), across 9 input-sequence-length rungs (ISL 4k–1024k) × 3 GVR-active layers each. `N` = post-compress indexer length (V4 compress_ratio=4, V3.2 cr=1); `hit` = |preIdx ∩ topK| / K (0.27–1.00 measured across the sweep). For V3.2 (no warm-start) `preIdx` is reconstructed from the previous decode step's top-K. V3.2 spans 7 rungs (4k–256k) rather than 9: DeepSeek-V3.2's maximum context is 160K (163,840 tokens), so longer prompts truncate to that window — its ISL-256k row's N=163,775 is the exact valid kv length at the benched decode step (verified from the capture: the newest kv position is selected every step, walking the top-K max index +1 per step). Correctness is validated against each capture's own recorded top-K reference.

### Real decode-capture (production DSv4 / V3.2 indexer top-K, BS=1, fp32) — µs and speedup per seq-len

Per-layer top-K captured from real decode runs; `N` = post-compress indexer length, `hit` = preIdx∩topK overlap. Launch shapes per `pick_config` (cs=8 on the large-N rungs). One GVR-active bench layer per model (flash L22 / pro L30 / v32 L34).

**V4 Flash (K512)**

| ISL | N | hit | base (µs) | R0 (µs) | speedup | exact |
|--:|--:|--:|--:|--:|--:|:-:|
| 4k | 1,027 | 0.64 | 8.97 | 7.88 | 1.14× | ✓ |
| 8k | 2,051 | 0.33 | 8.82 | 8.38 | 1.05× | ✓ |
| 16k | 4,099 | 0.34 | 11.78 | 9.02 | 1.31× | ✓ |
| 32k | 8,195 | 0.69 | 11.74 | 9.37 | 1.25× | ✓ |
| 64k | 16,387 | 0.08 | 14.71 | 12.35 | 1.19× | ✓ |
| 128k | 32,771 | 0.70 | 16.30 | 12.03 | 1.36× | ✓ |
| 256k | 65,538 | 0.28 | 17.26 | 13.91 | 1.24× | ✓ |
| 512k | 131,075 | 0.06 | 43.46 | 20.74 | 2.10× | ✓ |
| 1024k | 262,127 | 0.42 | 20.84 | 18.02 | 1.16× | ✓ |

*V4 Flash (K512) geomean speedup: 1.285×*

**V4 Pro (K1024)**

| ISL | N | hit | base (µs) | R0 (µs) | speedup | exact |
|--:|--:|--:|--:|--:|--:|:-:|
| 4k | 1,027 | 1.00 | 19.21 | 10.05 | 1.91× | ✓ |
| 8k | 2,051 | 0.46 | 9.30 | 9.21 | 1.01× | ✓ |
| 16k | 4,099 | 0.74 | 15.35 | 10.08 | 1.52× | ✓ |
| 32k | 8,195 | 0.53 | 18.66 | 11.56 | 1.61× | ✓ |
| 64k | 16,387 | 0.31 | 12.33 | 11.73 | 1.05× | ✓ |
| 128k | 32,771 | 0.33 | 16.87 | 13.61 | 1.24× | ✓ |
| 256k | 65,539 | 0.36 | 17.01 | 15.16 | 1.12× | ✓ |
| 512k | 131,075 | 0.23 | 21.59 | 17.64 | 1.22× | ✓ |
| 1024k | 262,127 | 0.27 | 25.03 | 20.11 | 1.25× | ✓ |

*V4 Pro (K1024) geomean speedup: 1.299×*

**V3.2 (K2048)**

| ISL | N | hit | base (µs) | R0 (µs) | speedup | exact |
|--:|--:|--:|--:|--:|--:|:-:|
| 4k | 4,111 | 0.73 | 16.06 | 10.86 | 1.48× | ✓ |
| 8k | 8,207 | 0.84 | 15.46 | 11.90 | 1.30× | ✓ |
| 16k | 16,399 | 0.53 | 28.00 | 13.76 | 2.03× | ✓ |
| 32k | 32,783 | 0.43 | 20.62 | 16.78 | 1.23× | ✓ |
| 64k | 65,551 | 0.41 | 31.96 | 20.68 | 1.55× | ✓ |
| 128k | 131,087 | 0.62 | 21.56 | 18.22 | 1.18× | ✓ |
| 256k | 163,775 | 0.87 | 28.38 | 19.11 | 1.48× | ✓ |

*V3.2 (K2048) geomean speedup: 1.443×*

**Real overall (25 cells): geomean 1.333×, 25/25 ≥ 1.0, exactness 25/25.**

The former sub-1.0 real cell (Pro 512k, 0.959× at an earlier revision) is a **repair-active row** — the exact boundary-tie repair triggers there, and the pre-fix kernel returned a wrong top-K set on exactly these rows (see *Correctness*). The tiny-tie fast path now recovers that re-rank cost: the cell measures **1.22×** at the current head.

### Batch-size scaling + launch-shape policy (`pick_config` / `launch`)

The BS=1 tables above raise the natural question of batch scaling. Two findings (same nsys cold-L2 protocol, BS ∈ {1…1024}, capture inputs replicated to BS independent rows):

- **R0 itself is BS-invariant.** Re-measured 2026-07-20 at the current head over the full real-capture BS grid (all captured ISL rungs × 3 dtypes × 11 BS points = 825 cells, R0 exact **825/825**): the R0/base geomean is **1.257× at BS=1 → 1.262× at BS=1024** and never drops below 1.215× (BS=128) at any point of the 11-point BS grid (per-model BS=1 → BS=1024: Flash 1.272→1.314×, Pro 1.234→1.214×, V3.2 1.267→1.258×). R0 changes Phase-2/Phase-4 arithmetic only — grid shape, smem budget, and cluster semantics are identical to base — so it introduces no batch-size coupling.
- **Launch-shape choice dominates at large BS — and it is a launch-time decision, not a kernel property.** `cluster_size` / `num_threads` / `min_blocks_per_mp` are compile-time constructor knobs: a compiled kernel cannot change its own grid or cluster shape, so batch adaptation must pick a different compiled variant per call. Driving the kernel with a config frozen at the BS=1 optimum (cluster_size = N≥64K ? 4 : 1, T=1024, mbpm=1) is geomean **2.38× slower (max 5.8×)** than per-(BS, N) picks across BS ∈ {64, 256, 1024} — e.g. K512 bf16 N=65536 BS=1024: 419 µs frozen vs 72 µs picked. Multi-CTA splitting only pays while the grid is a single wave (`num_rows × cluster_size ≤ num_SMs`); past that, row parallelism already saturates the SMs and per-row splitting is pure overhead. The production custom op already makes these picks; the risk was only ever to direct-drive users.

To make that policy part of the kernel's contract (and un-mismeasurable), this PR adds two host-side helpers to `GvrTopKKernel` — no device-code change:

- **`pick_config(dtype, num_rows, num_candidates, max_seq_len=None)`** — the (dtype, BS, N) → ctor-kwargs launch-shape policy as a pure classmethod colocated with the kernel: cluster_size 8/4/2/1 by (BS, N) (8 = tiny-grid large-N: BS ≤ 4, N ≥ 128K), threads/occupancy/vec-width tiers, and the CUDA-graph contract (`max_seq_len` = peak runtime N so variants are picked for the replay shape, not the capture shape). Mirrors the production runner's inline policy; the follow-up dispatch-guard PR unifies the runner onto it.
- **`launch(logits, pre_idx, seq_lens, out_indices, top_k, ...)`** — a thin compiled-variant cache (same sym-int + tvm-ffi compile contract as the runner) so tests and benchmarks get production-equivalent shapes by default; `**kernel_overrides` (e.g. `enable_r0=False`, `cluster_size=8`) force any knob and participate in the cache key.

**cluster_size = 8 validated** (previously untested anywhere): 78/78 standalone exactness cells green (R0 **and** secant, dtype ∈ {fp32, bf16, fp16} × K ∈ {512, 1024, 2048} × N ∈ {131072, 262144} × warm/mid/cold hint × BS ∈ {1, 4}, fp32 R0 = secant identical index sets), and the cs=8 pick beats a forced cs=4 on 8/8 nsys cells (geomean 0.943×, up to −12 % at N=262144).

## Known limitation (disclosed) + follow-up
R0 is **not** a uniform win on the cold-hint tail, but the seed rung + K2048 recalibration tightened the envelope materially:

- **Previously disclosed fat-admission regression: fixed.** An earlier revision of this PR regressed to **0.68–0.79× of base** on the real Flash 1M-ISL capture at BS ≥ 128 (hit ≈ 0.42): the coarse static rung admitted ~4400 candidates where the row needed ~630 (7× Phase-3/4 work). The virtual seed rung adapts the admission to the row's values and returns those cells to base parity (fp32 **1.01–1.02×**, up to 1.43× vs the previous revision).
- **fp32 (the dtype the production indexer runs)**: over the full fp32 dtype × BS × N audit grid (which includes a calibrated synthetic cold-hint stress axis, hit ≈ 0.05, alongside the captures) there is **no cell below 0.915×** of base; the residual sub-1.0 cells are cold-hint stress rows, a regime not observed in the production captures.
- **16-bit inputs carry the remaining tail.** Over the full 2772-cell dtype × BS × N audit grid (re-measured at the current head, 2026-07-20), R0/base geomean is **1.190×** with min **0.887×** and only **4 cells < 0.90×** — all four are bf16/fp16 (synthetic cold-hint N=1M and real Pro 8k-ISL big-BS). At the previous revision this tail was 106 cells with min 0.790×; the K2048 recalibration + tiny-tie fast path closed most of it.

The real decode captures do not exhibit the cold-hint distribution (measured hit-rates 0.27–1.00; 25/25 real cells ≥ 1.0), which is why the default is on. A **follow-up PR** adds a dispatch guard routing the residual regimes to the retained secant path. Because the hint hit-rate is **not observable at inference time** (it is the overlap of the *current* top-K with the hint), the guard is designed as an **in-kernel admission escape** (R0 already counts seed admissions — bail out to the secant path when the count signals a cold seed) and/or **trailing-step feedback** (the kernel emits its measured admission counter; the host uses the previous step's value as this step's predictor). The `enable_r0=False` fallback in this PR is exactly what that guard dispatches to.

## Correctness of the results
Every timing cell above is **value-set-exact vs `torch.topk`** and the exactness is asserted on the same run that produces the time:

- **Tie-aware value-set check.** top-K is order-independent, so correctness is by set, not position. The check gathers the kernel's selected values and asserts the sorted value-multiset equals `torch.topk`'s (rtol/atol 1e-5), plus in-range + no-duplicate + `n_below`=0 guards. This is robust to value ties (a different index with an equal value is still correct). **Real 25/25** (vs each capture's recorded reference) plus a 52-cell calibrated synthetic envelope **52/52**; over the full dtype × BS audit grid, R0 is exact on **all 2772** cells (re-verified at the current head, 2026-07-20).
- **Boundary-tie defect found by a bit-exact audit and fixed in this PR.** On real Pro 512k fp32 captures, the pre-fix Phase-4 kept an arbitrary member of a sub-resolution tie set straddling the top-K boundary (1-element miss, |dv| ≈ 3e-6 — inside the 1e-5 tolerance above, but not the true set; data-dependent, does not reproduce on tie-free synthetic rows). The `p4_exact_tail` radix re-rank makes all 12 affected audit cells **bit-exact** (validated at BS 1/128/1024), with paired cold-L2 A/B on unaffected cells at 0.998 geomean (noise) and 16-bit arms byte-identical.
- **Base-secant undershoot repaired.** On real Flash 512k (hit-rate ≈ 0.06) the upstream **base** secant is itself inexact (returns < K unique) *and* ~2.1× slower; the full dtype × BS grid exposes this same rung as **36 base-inexact cells (base 2736/2772)** — R0 is exact on all of them.
- **New unit tests.** `test_cute_dsl_gvr_topk_decode_r0_equivalence` (SM100-gated) drives `GvrTopKKernel` directly (the custom op does not expose `enable_r0`), runs **both** `enable_r0=True` and `enable_r0=False`, and asserts (1) both arms are a valid top-K vs `torch.topk` (tie-aware value-multiset) and (2) **identical index sets** on tie-free fp32. Grid: dtype ∈ {bf16, fp16, fp32} × K ∈ {512, 1024, 2048} × N ∈ {8192, 65536} × BS ∈ {1, 16} × warm/cold hint × cluster_size ∈ {1, 4, 8} (cs=8 gated to N ≥ 65536, its production regime). Additional tests: `..._r0_equivalence_bigbs` (multi-wave grids: BS=256 single-CTA, BS=64 cluster_size=4), `..._pick_config_policy` (locks the (BS, N) → shape map and the big-BS occupancy knobs), `..._launch_autoconfig` (drives `launch()` across all four cluster regimes incl. a forced-secant override arm), and `..._p4_exact_tail_ties` (plants adversarial **5e-8-spaced and 1-ulp bitwise** tie bands across the top-K boundary and asserts value-multiset exactness across K × N × cluster_size). This is also the sole coverage of the secant fallback now that op-level tests inherit the new default. A standalone exactness harness independently confirms **186/186** across dtype/K/N/BS + adversarial quantized tie plateaus + cluster.

## Scope / risk / rollback
- Touches **only** `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py` and its unit-test file. No call-site, dispatch, or config-schema changes.
- `enable_r0=False` restores the pre-R0 upstream kernel **byte-for-byte** (all R0 fields are `const_expr`-folded). Full rollback = flip one default back.
- `r0_vseed=False` retains the previous static `(0.85, 0.35)` ladder for all K; `p4_exact_tail` defaults off for 16-bit (byte-identical there).

## Test plan
```
pytest tests/unittest/_torch/attention/sparse/test_cute_dsl_gvr_topk_decode.py
```
on SM100 (B200 / B300).
