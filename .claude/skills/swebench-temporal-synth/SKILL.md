---
name: swebench-temporal-synth
description: >
  Generate temporally-coherent synthetic decode logits + preIdx for GVR
  Top-K benchmarks, mimicking SWE-Bench-64K decode-step prev/current
  correlation. Row sampled from one of 3 beta cfgs of
  `_DECODE_DIST_CONFIGS` (≈ SWE-Bench L20 / L22 / L42 fits). preIdx built
  from topK(row + c·σ·noise) with c binary-searched per (cfg, N) to give
  realised hit_rate ≈ 0.50 — produces boundary-band miss distribution
  matching real decode-step preIdx behavior. K=2048 fp32 BS=1 default.
  Optional follow-up: run GVR vs Radix nsys speedup measurement on the
  generated data (validated upper bound on production: 2.08-2.25× at
  N=128K; closest synthetic match to Q9d-04b real-data 1.94×). Trigger
  keywords: "生成SWE-bench随机合成时间相关数据", "synthesize temporal
  coherent SWE-bench logits", "temporal preIdx benchmark data",
  "GVR boundary-band miss synth", "decode 时间相干 合成 logits",
  "generate prev-step coherent preIdx", "boundary-miss preIdx synth",
  "SWE-bench-aligned beta logits with temporal preIdx".
license: LicenseRef-NvidiaProprietary
metadata:
  author: loncheng@nvidia.com
  parent_experiment: Q19-tempC (ablation_study/gvr_phase_timing/19_seq_bs_sweep_beta/REPORT_temporal.md)
  hardware: NVIDIA Blackwell (B200 sm_100 / B300 sm_100a)
  validated_against: Q9d-04b real SWE-Bench (production 1.94× ↔ skill upper bound 2.08-2.25×)
---

# SWE-Bench-aligned temporal-coherence synthetic data — skill

## What this skill does

Given a target sequence length `N` (e.g. 64K, 128K), a beta distribution
cfg matching a SWE-Bench layer fit, and `K=2048` Top-K, this skill emits
two tensors ready for the production `torch.ops.trtllm.indexer_topk_decode`
op:

1. **`logits`** — `[BS, N_padded]` fp32, sampled from one of 3 beta
   cfgs (mean ≈ −0.75 / −2.96 / −4.51, full_range ≈ 11–14, clipped). All 3
   cfgs are direct copies of beta entries in
   `tests/unittest/_torch/thop/parallel/test_indexer_topk.py::_DECODE_DIST_CONFIGS`.
2. **`preIdx`** — `[BS, K]` int32, built via the temporal-coherence
   procedure (Q19 Design C):
   - sample Gaussian noise `c · σ · N(0, 1)` with `σ = row.std()` and `c`
     binary-searched (20 iters, tolerance ±0.005) to give a realised
     **hit-rate ≈ 0.50**;
   - `prev_topk = topK(row + noise, K, sorted=True)`;
   - replace `prev_topk[-1]` with `current_argmax` if missing (kernel
     argmax invariant);
   - `preIdx = prev_topk − 1` (caller-side −1; kernel will add +1 per
     `heuristicTopKDecode.cu:89,145`).

The misses (positions in `prev_topk \ current_topk`) naturally cluster
at the threshold boundary because Gaussian noise pushes ex-topK positions
from above-threshold to just-below-threshold. This boundary-band miss
distribution tightens GVR's P1 (pmin, pmax, pmean) bracket and is the
mechanism by which real production preIdx matches Q9d-04b's 1.94×
speedup.

## When to invoke

Triggered when the user asks for SWE-Bench-aligned synthetic data with
temporal correlation between consecutive decode steps. Sample phrasings:

- "生成 SWE-bench 随机合成时间相关数据，长度为 64K"
- "synthesize temporal-coherent SWE-bench-style decode logits"
- "generate prev-step coherent preIdx with 50 % hit rate"
- "boundary-band miss synth at N=128K"
- "give me logits + preIdx tensors matching SWE-Bench L42 with realistic
  prev-step correlation"

Do **not** invoke for:
- Uniform random `torch.rand` synth → use the simpler
  `tests/unittest/_torch/thop/parallel/test_indexer_topk.py::create_random_logits`.
- Random-position-from-true_topK preIdx → use Q19 main bench script with
  `--preidx_mode topvalue`.
- Real SWE-Bench data → use the `gvr-vs-radix-swebench` skill (cluster
  `computelab-sc-01` only).

## Parameters

| Flag | Default | Description |
|---|---|---|
| `--N` | `65536` | Sequence length (`int`). User says "64K" → `65536`, "128K" → `131072`, etc. Must satisfy `N > 2·K = 4096`. |
| `--cfg` | `beta_moderate` | Beta cfg. Choices: `beta_shallow` (≈L20 m=−0.75), `beta_moderate` (≈L22 region m=−2.96), `beta_deep` (≈L42 m=−4.51), or `all` (emit 3 separate file triplets). |
| `--bs` | `1` | Batch size (`int`). Rows are replicated, not independently sampled. |
| `--target_hr` | `0.50` | Calibration target for `|prev_topk ∩ current_topk| / K`. Realised hit rate is reported per cell. |
| `--seed` | `42` | torch / numpy random seed for row sampling. Noise seed is `seed + 1000` (separate, fixed across calibration + bench). |
| `--K` | `2048` | Top-K. Fixed in production but exposed for ablation. |
| `--max_c` | `5.0` | Upper bound for noise coefficient binary search. Larger N may converge below 1.0; small N (where K/N is close to 0.5) may saturate. |
| `--outdir` | `./synth_out/` | Output directory. Created if missing. |
| `--bench` | (false) | If set, also run a quick GVR vs Radix speedup measurement on the generated data and emit `speedup.txt`. Requires `$TRTLLM_REPO/cpp/build/tensorrt_llm/thop/libth_common.so`. |

## How to run

One-shot CLI (synthesis only):

```bash
python3 ${SKILL_DIR}/src/synth_temporal_data.py \
    --N 65536 --cfg beta_moderate --bs 1 \
    --outdir /tmp/swebench_synth_64k
```

With benchmark:

```bash
python3 ${SKILL_DIR}/src/synth_temporal_data.py \
    --N 65536 --cfg all --bs 1 --bench \
    --outdir /tmp/swebench_synth_64k_bench
```

All-cfg sweep example (3 beta cfgs × 6 N from Q19-tempC):

```bash
bash ${SKILL_DIR}/src/run_all_n.sh /tmp/swebench_synth_full_sweep
```

## What this skill outputs

For each invocation, the output directory contains:

```
synth_out/
  {cfg}_N{N}_bs{BS}/
    logits.pt          # fp32 [BS, N_padded], with -inf padding at >N
    preIdx.pt          # int32 [BS, K=2048] (kernel reads logits[preIdx[i]+1])
    seq_lens.pt        # int32 [BS] = N (valid range)
    meta.json          # noise_c, realised_hit_rate, row stats, kernel-side hit_rate
    [speedup.txt]      # if --bench: GVR/Radix wall in µs + speedup ratio
```

Loading from PyTorch:

```python
import torch
logits   = torch.load("synth_out/beta_moderate_N65536_bs1/logits.pt")
preIdx   = torch.load("synth_out/beta_moderate_N65536_bs1/preIdx.pt")
seq_lens = torch.load("synth_out/beta_moderate_N65536_bs1/seq_lens.pt")

# preIdx already has the -1 offset applied; just pass through:
indices = torch.empty((logits.shape[0], 2048), dtype=torch.int32, device="cuda")
scratch = torch.empty((logits.shape[0] * 2048,), dtype=logits.dtype, device="cuda")
torch.ops.trtllm.indexer_topk_decode(
    logits, seq_lens, indices, 1, 2048, preIdx, scratch
)
```

## Key invariants enforced

| Invariant | Where enforced |
|---|---|
| Row stats match target cfg moments (mean ± 0.05, std ± 0.05) | `sample_beta_row()` uses analytical moment-match `_fit_beta_params`, then clips to `[mean − fr/2, mean + fr/2]` |
| `preIdx` is in **previous-step** coordinates (caller −1) | `prev_topk - 1` applied unconditionally; kernel adds +1 per `heuristicTopKDecode.cu:89` |
| Current `argmax` is readable via some preIdx slot | If `current_argmax ∉ prev_topk`, replace `prev_topk[-1]` with it before −1 |
| Realised hit rate ≈ 0.50 (or `--target_hr`) | 20-iter binary search on `c`; reports realised hit rate in `meta.json` |
| `N_padded = (N + 3) & ~3` for fp32 float4 alignment | `_pad_align_inf()`, padding cols set to `-inf` |

## Failure modes

| Failure | Behaviour |
|---|---|
| `N <= 2*K` | Aborts: miss pool would be empty / inverse-cdf saturates. Error message names the K/N ratio. |
| `scipy` missing | Aborts (logistic / lognorm / weibull future-extensions need scipy; beta alone uses NumPy only — script falls back if `scipy.stats` is unavailable, but the framework keeps the import for cross-family parity). |
| `--bench` set but `$TRTLLM_REPO` libth_common.so missing | `--bench` exits with explicit ImportError naming expected path. Synthesis itself still succeeds. |
| Calibration cannot reach `target_hr` (K/N=0.5 floor) | Reported in `meta.json` as `"calibration_saturated": true`; realised hit rate clamped to the achievable floor. Synthesis proceeds. |
| `--cfg=all` × `--N` × `--bs` produces existing directory | Each output goes to its own `{cfg}_N{N}_bs{BS}/` subdir; no in-place overwrite. |

## Validated context (from Q19-tempC)

Anchors for what speedup the generated data should produce when fed
through `torch.ops.trtllm.indexer_topk_decode` (production path with
preIdxOffset=+1 internally) on a B200 sm_100:

| N | beta_shallow | beta_moderate | beta_deep | (production reference) |
|---:|---:|---:|---:|---:|
|  4 K | 1.02× | 1.00× | 1.01× | — |
|  8 K | 1.01× | 1.00× | 1.01× | — |
| 16 K | 1.28× | 1.40× | 1.24× | — |
| 32 K | 1.55× | 1.47× | 1.51× | — |
| 64 K | **1.85×** | **1.88×** | 1.76× | — |
|128 K | **2.25×** ⭐ | **2.08×** | **2.18×** | — |
| (Q9d-04b real SWE-Bench L0–L60, K=2048 fp32 BS=1) | — | — | — | **1.94×** |

The skill's temporal-coherence preIdx OVERSHOOTS production by 7–16% at
N=128K because real prev-step preIdx has additional looseness sources
(KV slot shifts, non-Gaussian decode drift, indexer-vs-attention layer
mismatch) that pure i.i.d. Gaussian noise cannot reproduce. Treat the
skill's output as an **upper bound** on GVR vs Radix speedup for
synthetic-data benchmarks.

## Notes on methodology

- **Why beta only**: the original ablation (Q19 main) showed all 4 dist
  families share Radix wall behavior (~77 µs at N=128K) but differ in
  GVR wall. Beta covers L20/L22/L42 SWE-Bench layers — the dominant
  fit family per CLAUDE.md §"Real Data". For lognorm/logistic/weibull
  variants, use Q19 main bench script with `--preidx_mode topvalue` (the
  temporal mode is not extended to non-beta families per user spec
  2026-05-12).
- **Why per-(cfg, N) calibration**: noise coefficient `c` must adapt to
  K/N density. At K/N = 0.5 (N=4K), c saturates at the asymptotic floor;
  at K/N = 0.0156 (N=128K), c ≈ 0.45. Hardcoded sigma would only work
  for one N.
- **Why hit_rate=0.50 default**: Q-temp study (`03_root_cause_analysis/
  05_temporal_threshold/REPORT.md`) measured prod-decode hit rate at
  ≈0.45-0.55 across layers; 0.50 is the consensus midpoint.
- **Noise seed fixed at `seed + 1000`**: separating row seed from noise
  seed makes the calibration deterministic; same noise realisation used
  during binary search and final preIdx construction.
- **argmax replacement at slot K-1, not slot 0**: kernel doesn't require
  argmax at slot 0 specifically — only that argmax is reachable via
  *some* preIdx slot. Replacing the lowest-value prev slot preserves
  the boundary-band character of the other 2047 slots.

## See also

- **Parent experiment**: `ablation_study/gvr_phase_timing/19_seq_bs_sweep_beta/REPORT_temporal.md`
- **Sibling skill** (real data, not synthetic): `gvr-vs-radix-swebench`
- **Kernel invariant verification**: `heuristicTopKDecode.cu:89,145` +
  `heuristic_topk.cuh:597`
- **F006 / F007 / F008 / F009 rules** (local CLAUDE.md):
  production path requirements, dist-family coverage, preIdx quality
  non-monotonicity, and 3-mode bracket on production speedup.
