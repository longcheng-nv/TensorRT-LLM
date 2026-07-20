# PR #16457 body — 2026-07-20 concise rewrite at head e6fdbfac3d
# (real-only perf; data: results_canonical_seqlen_fp32.jsonl + results_canonical_bs_real.jsonl
#  + headfull 2772-cell audit, HEADFULL_VERDICT.md)
---
## Summary

Adds an **R0 histogram-ladder admission** fast path + a fused **rank-and-scatter Phase-4** writeback to the production Blackwell cuTe-DSL GVR top-K decode kernel (`GvrTopKKernel`), and makes R0 the **default** (`enable_r0=True`).

**Headline (real production decode captures, B200): 1.33× geomean over the shipped kernel, faster on 25/25 cells, exact on the full 2772-cell audit grid — and it repairs two correctness defects of the shipped kernel** (a low-hit undershoot and a boundary-tie miss; see *Correctness*).

What's inside:

- **R0 admission**: replaces the Phase-2 secant threshold search with a single-pass multi-threshold rung ladder seeded by a 256-bin histogram over the prev-topK gathered values; fused rank-and-scatter Phase-4 (cluster barriers ~14 → ~7).
- **Virtual seed rung (`r0_vseed`)**: P1's mean probe is folded into the count pass as a free extra rung (zero SMEM growth), adapting admission per row — fixes the cold-hint fat-admission regime.
- **`p4_exact_tail`** (fp32 default on): bit-exact boundary-tie resolution (ambiguity-gated radix select + tiny-tie fast path); 16-bit kernels byte-identical.
- **K2048 tuning**: low rung 0.85 → 0.6 recalibrated on real captures + Phase-4 histogram 2048 → 512 bins.
- The classic **secant path is retained verbatim** (`enable_r0=False`); this PR flips the kernel default only — call-site/dispatch is untouched (guard = follow-up PR).

## Performance

**Methodology.** nsys pure-kernel, **cold-L2**, single-GPU same-run A/B (R0 vs retained secant on identical inputs), 20 cold reps, B200. Both arms launched via the kernel's own `launch`/`pick_config` contract (the shapes the production runner picks). Re-measured **2026-07-20 at the current head**. Inputs = per-layer indexer top-K **captured from production DeepSeek-V4 Flash / Pro / V3.2 BS=1 greedy decode** (9 ISL rungs 4k–1024k; `N` = post-compress indexer length, V4 cr=4 / V3.2 cr=1; `hit` = preIdx∩topK / K; V3.2 `preIdx` = previous step's top-K). V3.2 has 7 rungs: its 160K (163,840-token) max context truncates longer prompts — the 256k row's N=163,775 is the exact valid kv length at the benched step. A calibrated synthetic envelope is used only for exactness/audit coverage (*Correctness*, *Known limitation*).

### BS=1 fp32, per seq-len — **geomean 1.333×, 25/25 ≥ 1.0, exact 25/25**

**V4 Flash (K512) — geomean 1.285×**

| ISL | N | hit | base (µs) | R0 (µs) | speedup |
|--:|--:|--:|--:|--:|--:|
| 4k | 1,027 | 0.64 | 8.97 | 7.88 | 1.14× |
| 8k | 2,051 | 0.33 | 8.82 | 8.38 | 1.05× |
| 16k | 4,099 | 0.34 | 11.78 | 9.02 | 1.31× |
| 32k | 8,195 | 0.69 | 11.74 | 9.37 | 1.25× |
| 64k | 16,387 | 0.08 | 14.71 | 12.35 | 1.19× |
| 128k | 32,771 | 0.70 | 16.30 | 12.03 | 1.36× |
| 256k | 65,538 | 0.28 | 17.26 | 13.91 | 1.24× |
| 512k | 131,075 | 0.06 | 43.46 | 20.74 | **2.10×** |
| 1024k | 262,127 | 0.42 | 20.84 | 18.02 | 1.16× |

**V4 Pro (K1024) — geomean 1.299×**

| ISL | N | hit | base (µs) | R0 (µs) | speedup |
|--:|--:|--:|--:|--:|--:|
| 4k | 1,027 | 1.00 | 19.21 | 10.05 | **1.91×** |
| 8k | 2,051 | 0.46 | 9.30 | 9.21 | 1.01× |
| 16k | 4,099 | 0.74 | 15.35 | 10.08 | 1.52× |
| 32k | 8,195 | 0.53 | 18.66 | 11.56 | 1.61× |
| 64k | 16,387 | 0.31 | 12.33 | 11.73 | 1.05× |
| 128k | 32,771 | 0.33 | 16.87 | 13.61 | 1.24× |
| 256k | 65,539 | 0.36 | 17.01 | 15.16 | 1.12× |
| 512k | 131,075 | 0.23 | 21.59 | 17.64 | 1.22× |
| 1024k | 262,127 | 0.27 | 25.03 | 20.11 | 1.25× |

**V3.2 (K2048) — geomean 1.443×**

| ISL | N | hit | base (µs) | R0 (µs) | speedup |
|--:|--:|--:|--:|--:|--:|
| 4k | 4,111 | 0.73 | 16.06 | 10.86 | 1.48× |
| 8k | 8,207 | 0.84 | 15.46 | 11.90 | 1.30× |
| 16k | 16,399 | 0.53 | 28.00 | 13.76 | **2.03×** |
| 32k | 32,783 | 0.43 | 20.62 | 16.78 | 1.23× |
| 64k | 65,551 | 0.41 | 31.96 | 20.68 | 1.55× |
| 128k | 131,087 | 0.62 | 21.56 | 18.22 | 1.18× |
| 256k | 163,775 | 0.87 | 28.38 | 19.11 | 1.48× |

All cells exact vs the captured reference. Flash 512k: the base kernel is additionally **inexact** there (undershoot) — R0 is exact *and* 2.10× faster.

### BS scaling — R0/base geomean per BS (all captured ISL rungs × 3 dtypes; 825 cells, exact 825/825)

| | BS 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Flash (K512) | 1.272 | 1.263 | 1.264 | 1.261 | 1.258 | 1.254 | 1.248 | 1.240 | 1.278 | 1.295 | 1.314 |
| Pro (K1024) | 1.234 | 1.236 | 1.234 | 1.232 | 1.230 | 1.227 | 1.214 | 1.195 | 1.204 | 1.204 | 1.214 |
| V3.2 (K2048) | 1.267 | 1.274 | 1.270 | 1.264 | 1.259 | 1.264 | 1.236 | 1.207 | 1.227 | 1.260 | 1.258 |
| **all** | **1.257** | 1.256 | 1.254 | 1.251 | 1.248 | 1.247 | 1.232 | 1.215 | 1.237 | 1.252 | **1.262** |

- **R0 is BS-invariant** (flat 1.257× → 1.262×, floor 1.215× at BS=128): it changes Phase-2/4 arithmetic only — grid shape, smem, and cluster semantics are identical to base.
- **Launch shape, not the kernel, dominates at large BS.** A config frozen at the BS=1 optimum is geomean 2.38× (max 5.8×) slower than per-(BS, N) picks. This PR therefore ships the policy as part of the kernel: **`pick_config`** (the (dtype, BS, N) → launch-shape classmethod, incl. the CUDA-graph `max_seq_len` contract) and **`launch`** (compiled-variant cache; `**kernel_overrides` for forcing knobs). cluster_size=8 (previously untested) validated: 78/78 exactness cells, beats forced cs=4 on 8/8 nsys cells.

## Known limitation + follow-up

- The residual R0 losses are confined to the **cold-hint stress tail** (synthetic hit ≈ 0.05), a regime not observed in any production capture (measured hit 0.27–1.00): full 2772-cell audit grid geomean **1.190×**, fp32 never below **0.915×**, and only **4 cells < 0.90×** (min 0.887×, all bf16/fp16). An earlier revision's fat-admission regression (0.68–0.79× on Flash-1M big-BS) and 106-cell 16-bit tail are **fixed** by the seed rung + K2048 recalibration.
- **Follow-up PR**: dispatch guard routing residual regimes to the retained secant path. Hit-rate is not observable at inference time, so the guard is an **in-kernel admission escape** and/or **trailing-step feedback** — `enable_r0=False` is exactly what it dispatches to.

## Correctness

- Every timing cell asserts a **tie-aware value-set match vs `torch.topk`** on the same run (plus in-range / no-duplicate / n_below=0 guards). Real 25/25, synthetic envelope 52/52, full audit grid **2772/2772** (re-verified at the current head).
- **Fixes a real boundary-tie defect**: the pre-fix Phase-4 kept an arbitrary member of a sub-resolution tie set straddling the top-K boundary (real Pro-512k captures, 1-element miss, |dv| ≈ 3e-6). `p4_exact_tail` makes all 12 affected audit cells bit-exact; unaffected-cell cost is noise (0.998 geomean); the tiny-tie fast path returns repair-active rows to a win (Pro 512k: 1.22×).
- **Repairs the base undershoot**: on real Flash-512k (hit ≈ 0.06) the shipped secant returns < K unique indices on **36 audit cells** — R0 is exact on all of them.
- **New SM100 unit tests**: R0/secant equivalence (both arms valid top-K + identical index sets on tie-free fp32) across dtype × K × N × BS × hint × cluster_size ∈ {1,4,8}; big-BS multi-wave grids; `pick_config` policy lock; `launch()` autoconfig incl. forced-secant arm; adversarial 5e-8 / 1-ulp tie bands (`p4_exact_tail`). A standalone harness independently confirms 186/186.

## Scope / risk / rollback

- Touches **only** `gvr_topk_decode.py` + its unit-test file; no call-site, dispatch, or config-schema changes.
- `enable_r0=False` restores the pre-R0 kernel **byte-for-byte** (all R0 fields const-folded) — rollback = flip one default.
- `r0_vseed=False` keeps the static ladder; `p4_exact_tail` defaults off for 16-bit (byte-identical there).

## Test plan

```
pytest tests/unittest/_torch/attention/sparse/test_cute_dsl_gvr_topk_decode.py
```
on SM100 (B200 / B300).
