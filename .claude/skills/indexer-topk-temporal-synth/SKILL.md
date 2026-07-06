---
name: indexer-topk-temporal-synth
description: >
  UNIFIED generator of realistic synthetic decode logits + temporally-coherent
  preIdx for DeepSeek V3.2 (K=2048, cr=1), V4-Flash (K=512, cr=4) and V4-Pro
  (K=1024, cr=4) indexer top-K benchmarks. Supersedes swebench-temporal-synth,
  swebench-temporal-synth-v4flash and swebench-temporal-synth-v4pro.
  Marginal = per-layer empirical inverse-CDF + GPD tail calibrated from the
  real 64K production captures (fixes the single-Beta top-K-tail collapse:
  synth mass at the real selection boundary was 0.00x at N>=128K, now
  1.00-1.14x); rows can mix over the real layer family ("aggregate" cfg).
  Temporal = rank-conditional retention curve + real miss-depth samples +
  per-row hit-rate sampled from the real per-step distribution + optional
  Gaussian-copula AR(1) multi-step chain (lag-1 rho from real consecutive
  decode steps). Trigger keywords: "synthesize DSv3.2/V4 Flash/Pro indexer
  logits", "temporal-coherent synthetic decode logits", "GVR top-K synth
  data", "empirical-CDF indexer synth", "generate preIdx with real hit-rate",
  "unified temporal synth", "multi-step decode chain synth".
license: LicenseRef-NvidiaProprietary
metadata:
  author: loncheng@nvidia.com
  supersedes: swebench-temporal-synth, swebench-temporal-synth-v4flash, swebench-temporal-synth-v4pro
  motivating_study: synth_vs_real_validation/SYNTH_VS_REAL_VALIDATION.html (2026-06-29)
  validated: assets/validation_gates.json — all 5 gates PASS for all 3 models (2026-07-06)
  hardware: NVIDIA Blackwell (B200 sm_100 / B300 sm_100a)
---

# Unified DSv3.2 / V4-Flash / V4-Pro temporal indexer synth

## Why this replaces the three legacy skills

`synth_vs_real_validation/` falsified the legacy moment-matched single-Beta
generators against the real 64K production captures:

| defect (legacy Beta synth) | consequence | fix here |
|---|---|---|
| bounded Beta fit to (mean, σ) flattens the real heavy positive tail (real max ≈ +5..+10, synth ≈ +1..+3) | synth mass above the REAL top-K boundary → **0.00×** at N≥128K (V3.2/Flash) — near-threshold candidate/tie work understated | per-layer **empirical quantile table** (tail-densified p-grid) + **GPD** peaks-over-threshold extrapolation |
| single `beta_moderate` curve vs the real all-layer mixture (agg KS 0.19–0.20 for Flash/V3.2) | benchmarked curve unrepresentative of what the kernel sees across the model | `--cfg aggregate` (default): each row samples a real layer's marginal |
| iid-Gaussian noise + scalar `c` binary-searched to ONE fixed hit-rate | wrong rank-retention profile; no step-to-step hr variance; no undershoot | **rank-conditional retention curve** + per-row hr sampled from the **real per-step hr distribution** + real **miss-depth** samples + V4 sentinel (`n_valid<K`) distribution |
| no multi-step structure | EMA/A1-style multi-step benches impossible | `--steps T` Gaussian-copula AR(1) chain, exact closed loop `preIdx_t = topK(row_{t-1})` |

Validation gates (`src/validate_against_real.py`, results in
`assets/validation_gates.json`, all PASS 2026-07-06):

| gate | v32 | v4flash | v4pro | limit |
|---|---|---|---|---|
| G1 per-layer KS max | 0.005 | 0.003 | 0.002 | ≤0.05 |
| G2 aggregate KS | 0.021 | 0.018 | 0.021 | ≤0.05 |
| G3 boundary mass @16K/64K/256K | 1.01/1.10/1.14 | 1.11/1.12/1.11 | 1.03/1.03/1.01 | 0.80–1.25 |
| G4 retention-curve max err | 0.030 | 0.015 | 0.021 | ≤0.05 |
| G5 realised-vs-target hr err | 0.000 | 0.000 | 0.000 | ≤0.03 |

Kernel end-to-end: bundles from all 3 models run through the production
`torch.ops.trtllm.indexer_topk_decode` (GVR + Radix paths) with **exact**
top-K output (smoke-tested on B200).

## How to run

```bash
SKILL=<repo>/.claude/skills/indexer-topk-temporal-synth   # any TensorRT-LLM checkout carrying this skill

# V4-Flash, real layer-mixture marginal, BS=8 independent rows
python3 $SKILL/src/synth_temporal_data.py --model v4flash --N 64K \
    --cfg aggregate --bs 8 --outdir /tmp/synth_out

# Legacy bucket names still work (now = real-layer terciles by mean)
python3 $SKILL/src/synth_temporal_data.py --model v32 --N 128K \
    --cfg beta_moderate --bs 1 --outdir /tmp/synth_v32

# Single real layer; fixed hit-rate override
python3 $SKILL/src/synth_temporal_data.py --model v4pro --N 64K \
    --cfg L22 --target_hr 0.75 --outdir /tmp/synth_l22

# 4-step temporal chain (adds step{1..4}_logits.pt / step{t}_preIdx.pt)
python3 $SKILL/src/synth_temporal_data.py --model v4flash --N 32K \
    --steps 4 --outdir /tmp/synth_chain
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | `v32` \| `v4flash` \| `v4pro` — sets K, compress_ratio, preIdx offset, padding, seq_lens semantics |
| `--N` | `65536` | post-compress seq len; accepts `64K`, `131072`, … Must be > 2·K |
| `--cfg` | `aggregate` | `aggregate` (real per-layer mixture — recommended) \| `beta_shallow/moderate/deep` (legacy names → real-layer mean-terciles) \| `L<layer>` (single calibrated layer) \| `all` |
| `--bs` | `1` | batch size |
| `--row_mode` | `independent` | each row draws its own layer + sample; `replicate` = legacy broadcast of one row |
| `--target_hr` | (sampled) | fixed hit-rate; default samples the real per-step hr distribution per row |
| `--sentinel_mode` | `real` | V4: `n_valid<K` undershoot sentinels (-1) sampled from capture (captures at hand are all-valid, so usually = `full`); `full` forces K valid |
| `--steps` | `1` | >1 emits a copula AR(1) chain with exact `preIdx_t = topK(row_{t-1})` |
| `--dtype` | `fp32` | `fp32` \| `bf16` \| `fp16` logits |
| `--K` | model-native | ablation override |
| `--seed` | `42` | reproducible; per-row substreams derived internally |

### Output layout (drop-in compatible with the legacy skills)

```
{cfg}_N{N}_bs{BS}/
  logits.pt      [BS, N_padded]  (-inf padded; align 4 for v32, 8 for v4)
  preIdx.pt      [BS, K] int32   (v32: caller -1 applied; v4: raw positions)
  seq_lens.pt    [BS]  int32     (v32: N; v4: N*cr + next_n - 1)
  meta.json      model, per-row layer/hr/n_valid, calib provenance
  step{t}_*.pt   (only --steps > 1)
```

### nsys benchmark (GVR vs Radix)

Same workflow as the legacy skills; `bench_nsys.py` is model-aware via
`meta.json` (compress_ratio; radix_aux always pre-allocated per post-#14297):

```bash
cd /tmp/synth_out && nsys profile --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o nsys_sweep --force-overwrite=true \
    python3 $SKILL/src/bench_nsys.py --indir . --warmup 3 --reps 10
nsys stats -r nvtx_gpu_proj_trace nsys_sweep.nsys-rep --format csv -o nsys_sweep
python3 $SKILL/src/parse_nsys.py nsys_sweep_nvtx_gpu_proj_trace.csv
```

GOTCHAS: profile with `env -u GITHUB_TOKEN -u HF_TOKEN` (nsys sqlite embeds
process env); never commit `*.sqlite`/`*.nsys-rep`; on hosts with a broken-
cooling GPU0 (umbriel-b200-019/035) pin `CUDA_VISIBLE_DEVICES=1`.

## Calibration assets (committed; regeneration optional)

`assets/calib_{v32,v4flash,v4pro}.npz` (0.16–0.52 MB each) hold, per real
layer: quantile table on a tail-densified p-grid, GPD (p_u, u_thr, ξ, β),
retention-by-rank curve (20 buckets), miss-depth samples ((thr−logit)/σ),
per-step hit-rate samples, valid-count fractions, lag-1 copula ρ. Layer
buckets = mean-terciles. `assets/valpool_{model}.npz` are validation-only
real-sample pools; `assets/validation_gates.json` is the acceptance record.

Regenerate after a capture refresh (needs NFS access to the raw captures;
~1 min total):

```bash
python3 $SKILL/src/calibrate_from_real.py            # rebuild assets/
python3 $SKILL/src/validate_against_real.py          # must PASS 5 gates x 3 models
```

Real sources (64K production captures — same as synth_vs_real_validation):
- V3.2: `…/tllm_toolbox/indexer_topK_perf/data_distri/deepseek-v3.2-logging/notebooks/SWE_Bench_64K_decode_logits/Layer_{L}_pd.npy`, 9 layers; temporal stats from consecutive rows (production preIdx == prev-step topK by GVR closed loop).
- V4-Flash: Q9j `capture_20260520T082958_alllayers_swe64k`, 21 layers; real `preidx.in` per cell.
- V4-Pro: Q9k `capture_20260520T164146Z_v4pro_K1024_64k`, 30 layers; same.

Measured reality worth knowing (in assets): real hit-rate is a per-layer
DISTRIBUTION — V3.2 layer means 0.54–0.90, Flash 0.55–0.84, Pro 0.58–0.86
(the legacy fixed 0.50 / 0.36–0.46 / 0.69–0.77 scalars are replaced by
sampling); lag-1 copula ρ = 0.79–0.98.

## Paths & data dependencies (portability)

| execution path | needs | resolution |
|---|---|---|
| **synthesis** (`synth_temporal_data.py`) | only `assets/calib_*.npz` (~1 MB total, committed) + numpy/torch | assets resolved **relative to the skill dir** — no absolute paths; works from any checkout/machine |
| validation (`validate_against_real.py`) | committed `assets/valpool_*.npz` (42 MB, validation-only) + the falsification study's real cache `synth_vs_real_validation/cache/real_*.npz` (~66 MB, NOT part of the skill; regenerate via that study's `extract_real.py`) | cache found **repo-relative** in the same checkout; override with `$REAL_CACHE` |
| bench (`bench_nsys.py`) | built `libth_common.so` + `nvtx` + nsys | default = `<this repo>/cpp/build/tensorrt_llm/thop/libth_common.so` derived from the skill's own location; override with `$LIBTH_COMMON` |
| re-calibration (`calibrate_from_real.py`) | the RAW production captures on NFS (V3.2 `Layer_*_pd.npy`, Q9j Flash/Pro capture dirs) + `q9j_load.py` loader | canonical NFS paths are the **defaults**, all overridable via `--v32_dir/--flash_cap/--pro_cap/--q9j_src`; not needed unless refreshing assets |

If pushing this skill to a code-only branch, `assets/valpool_*.npz` (42 MB,
raw real-logit samples used only by the G1 gate) and
`assets/validation_gates.json` can be dropped — synthesis needs neither.

## Kernel contracts preserved (verbatim from the legacy skills)

| | v32 | v4flash / v4pro |
|---|---|---|
| K / compress_ratio | 2048 / 1 | 512 / 4, 1024 / 4 |
| preIdx caller offset | −1 (kernel adds +1, `heuristicTopKDecode.cu`) | 0 (kernel uses `preIdx[i]` directly) |
| logits pad align | 4 (float4) | 8 |
| seq_lens | N | N·cr + next_n − 1 |
| radix_aux_{indices,logits} | not required | required when blocksPerRow>1 (post-#14297) |
| argmax invariant | rank-0 always in preIdx (enforced) | same |

## Failure modes

| Failure | Behaviour |
|---|---|
| `N <= 2*K` | aborts with explicit message |
| assets missing | aborts naming `calibrate_from_real.py` |
| unknown `--cfg` | aborts listing valid choices |
| `--sentinel_mode real` on v32 | silently coerced to `full` (V3.2 capture has no preidx stream) |
| captures moved/renamed | only calibration is affected; committed assets keep synthesis working |

## See also

- Motivating falsification study: `synth_vs_real_validation/SYNTH_VS_REAL_VALIDATION.html` + `metrics.json`
- Real-data paths for tail-sensitive CONCLUSIONS: `indexer_topk_op_bench/harness/real_data.py` (V3.2 wired), Q9j/Q9k captures for V4 — synthetic data is for controllable sweeps over arbitrary (N, BS, seed), not a substitute for final real-data verdicts
- Legacy skills (kept for provenance, superseded): `swebench-temporal-synth{,-v4flash,-v4pro}`
