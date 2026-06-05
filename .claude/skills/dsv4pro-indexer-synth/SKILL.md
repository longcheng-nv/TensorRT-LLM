---
name: dsv4pro-indexer-synth
description: >
  Synthesize realistic DSV4 Pro (K=1024) or Flash (K=512) indexer input tensors
  — Q (FP4), K-cache (FP4), weights (fp32), logits (fp32), topK (int32),
  preIdx (int32) — for any BS / ISL / OSL combination. Logits-First +
  Rank-Transform + Temporal-Bias algorithm. Both models calibrated from real
  SWE-bench 64K B300 captures (2026-06-05). Key finding: Flash K-cache uses
  same FP4 e2m1 format as Pro (NOT FP8 as commonly assumed).
  Trigger: "合成 DSV4 indexer 测试数据", "synth Pro/Flash indexer inputs",
  "generate DSV4 random Q/K for kernel bench", "indexer unit test data".
license: LicenseRef-NvidiaProprietary
metadata:
  author: loncheng@nvidia.com
  calibrated_from: indexer_qkw_capture/capture_pro_swe64k_osl500 (2026-06-05)
  report: indexer_qkw_capture/REPORT_INDEXER_SYNTH.md
---

# DSV4 Pro Indexer Input Synthesis Skill

## What this skill does

Generate statistically realistic `(Q_fp4, K_cache_fp4, weights, logits, topK, preIdx)` 
tuples for DSV4 **Pro** indexer decode steps, given:

- **Model**: `pro` (K=1024, 30 GVR layers) or `flash` (K=512, 21 GVR layers)
- **BS**: batch size
- **ISL**: input sequence length (determines KV cache size via compress_ratio=4)
- **OSL**: number of decode steps to synthesize
- **target_hr**: GVR hit-rate (fraction of prev_topK re-selected each step)

All tensors use the same dtype and shape as real captures from `dsv4-indexer-capture`.

### Algorithm summary

```
For each (layer, decode_step):
  1. Sample logits_target ~ Gumbel_r(μ_t, σ_t)  [shallow L02-L10]
                          ~ Normal(μ_t, σ_t)     [deep    L12-L60]
     where (μ_t, σ_t) are drawn from the per-bucket distribution
  2. Generate random Q ~ N(0, 1.93) → FP4 pack
             random K ~ N(0.08, 2.32) → FP4 pack  
             weights ~ N(μ_w[layer], σ_w[layer])
  3. Compute raw_logits = Σ_h(w_h · Q_h @ K.T) / √128
  4. Rank-transform raw_logits → logits_target
     (preserves distribution shape; mean/std match target exactly)
  5. Temporal bias (AFTER rank-transform):
     c = binary_search(logits, prev_topK, target_hr)
     logits[prev_topK] += c
     → hit-rate(topK_t, topK_{t-1}) = target_hr ± 1e-9
  6. topK = argtopK(logits)
```

**⚠️ Critical ordering**: temporal bias MUST be applied AFTER rank-transform.
Applying it before is silently destroyed (rank-transform reassigns values by rank, 
not by position → hit-rate stays at random baseline K/N ≈ 6.9%).

---

## Distribution parameters (baked in)

Measured from SWE-bench 64K captures on B300, 30 GVR layers, 377 decode steps:

| Tensor | Distribution | Parameters | Layer variation |
|---|---|---|---|
| **Q** (FP4) | N(0.00, 1.93) clip(-6,6) | All layers identical | None |
| **K** (FP4) | N(0.08, 2.32) clip(-6,6) | All layers identical | None |
| **Weights** | N(0.020, 0.048) fp32 | Per-layer params available | std: 0.032–0.078 |
| **Logits mean** | μ ≈ -1.4 (shallow) / -1.1 (deep) | Per-layer params | Layer varies ±0.5 |
| **Logits dist** | Gumbel_r (L02-L10), Normal (L12-L60) | — | Distributional shift |
| **Hit-rate** | Default 0.69 (Pro mean) | Configurable 0–0.90+ | L02=0.757, L12=0.597 |

---

## CLI

```bash
python3 ${SKILL_DIR}/src/synth_indexer_inputs.py \
    --model     pro                  # required; only "pro" supported
    --bs        1                    # batch size (default 1)
    --isl       65536                # input seq len in tokens (default 65536)
    --osl       500                  # decode steps (default 500)
    --target-hr 0.69                 # GVR hit-rate (-1 = model default; 0 = none)
    --layers    even|all|2,12,20     # default: even (GVR-active layers)
    --format    pt|npz               # default pt
    --out-dir   /path/to/output      # required
    --seed      42
    --params    /path/to/params.json # override built-in params
    --no-k-cache                     # skip K-cache (saves memory/time)
    --no-logits                      # skip logits/topK/preIdx
```

### Common examples

**Quick Pro unit test (1 step, 3 representative layers):**
```bash
python3 ${SKILL_DIR}/src/synth_indexer_inputs.py \
    --model pro --bs 1 --isl 65536 --osl 1 \
    --target-hr 0.0 --layers 2,30,60 \
    --out-dir /tmp/synth_pro_quick
```

**Pro realistic decode sequence (GVR hit-rate 0.69):**
```bash
python3 ${SKILL_DIR}/src/synth_indexer_inputs.py \
    --model pro --bs 1 --isl 65536 --osl 500 \
    --target-hr 0.69 --layers even \
    --out-dir /tmp/synth_pro_hr069
```

**Kernel testing only (no logits, faster):**
```bash
python3 ${SKILL_DIR}/src/synth_indexer_inputs.py \
    --model pro --bs 1 --isl 65536 --osl 100 \
    --no-logits --out-dir /tmp/synth_qkw_only
```

---

## Output layout

```
<out-dir>/
  manifest.json          model, BS, ISL, OSL, n_kv, K, target_hr, ...
  layer_02/
    q_fp4.pt             {step: Tensor [BS,1,64,64] int8}   FP4 packed
    k_cache.pt           {step: Tensor [n_blocks,32,1,68]}  FP4+scale packed
    weights.pt           {step: Tensor [BS,64] float32}
    logits.pt            {step: Tensor [BS,n_kv] float32}
    topk.pt              {step: Tensor [BS,K] int32}         topK indices
    preidx.pt            {step: Tensor [BS,K] int32}         prev step topK (GVR input)
  layer_04/ ...
  layer_60/ ...
```

### K-cache layout detail (Pro FP4 path)

```
k_cache shape: [n_blocks, tokens_per_block=32, 1, 68 bytes]
  bytes [0:64]  = FP4 e2m1 packed K data  (128 values × 0.5 byte = 64 bytes)
  bytes [64:68] = quantization scale       (1 × float32 = 4 bytes)
```

### Loading synthesized data

```python
import torch
# Load one layer
q   = torch.load("layer_02/q_fp4.pt",    weights_only=True)  # {step: [BS,1,64,64] int8}
w   = torch.load("layer_02/weights.pt",  weights_only=True)  # {step: [BS,64] float32}
k   = torch.load("layer_02/k_cache.pt",  weights_only=True)  # {step: [n_blocks,32,1,68]}
lgt = torch.load("layer_02/logits.pt",   weights_only=True)  # {step: [BS,n_kv] float32}
tk  = torch.load("layer_02/topk.pt",     weights_only=True)  # {step: [BS,K] int32}
pre = torch.load("layer_02/preidx.pt",   weights_only=True)  # {step: [BS,K] int32}

# For kernel test at step 0:
q_step0 = q[0]    # [BS, 1, 64, 64] int8 — FP4 packed query
k_step0 = k[0]    # [n_blocks, 32, 1, 68] — FP4+scale paged K cache
w_step0 = w[0]    # [BS, 64] float32 — attention weight scalars
# Expected output:
expected_topk = tk[0]  # [BS, 1024] int32
```

---

## n_kv calculation

```
n_kv = ceil(ISL / compress_ratio) rounded up to nearest 64

Pro:   compress_ratio = 4,  ISL=65536 → n_kv = 16384
Flash: compress_ratio = 4,  ISL=32768 → n_kv = 8192
Flash: compress_ratio = 4,  ISL=65536 → n_kv = 16384

n_blocks = ceil(n_kv / tokens_per_block)
  Pro:   tokens_per_block = 32  → n_blocks ≈ 512 for 64K ISL
  Flash: tokens_per_block = 128 → n_blocks ≈ 128 for 64K ISL
```

---

## Validation quality

**Measured on DSV4 Pro SWE-bench 64K captures (10 layers, 100 steps)**:

| Configuration | KL_sym | W1 (logit units) | hit-rate achieved |
|---|---|---|---|
| target_hr=0.00 (no bias) | **0.036** | 0.055 | 0.069 (random) |
| target_hr=0.69 (Pro real) | 0.067 | 0.167 | **0.690 ± 0.000** |
| target_hr=0.75 | 0.074 | 0.174 | **0.750 ± 0.000** |
| target_hr=0.90 | 0.122 | 0.232 | **0.900 ± 0.000** |

KS test at N=14848 always rejects parametric distributions by construction
(too much statistical power). Use KL/W1 for synthesis quality assessment.

---

## Gotchas

- **G1 — temporal bias ordering**: Apply `logits[prev_topK] += c` AFTER rank-transform,
  never before. Rank-transform destroys positional structure.
- **G2 — n_kv must be even**: FP4 packing requires even head_dim and even token count
  for block alignment. Use `((n_kv + 63) // 64) * 64`.
- **G3 — Flash K-cache is FP4 (NOT FP8)**: Despite Flash lacking `indexer_k_dtype` in
  config.json, real captures on B300 show Flash K-cache is `[n_blocks,32,1,68]` FP4 e2m1
  packed — identical format to Pro. The initial assumption of FP8 was wrong. Both models
  use FP4 for indexer K. Flash hit-rate ≈ 0.61 (lower than Pro 0.69; use `--target-hr 0.61`).
- **G4 — preIdx at step 0**: step 0 has no prev_topK → `preidx[0]` is zeros tensor.
  This is correct — real captures also show zeros for the first decode step.
- **G5 — hit-rate saturation**: For target_hr > 0.95 at small n_kv, binary search
  may saturate at `max_c=20`. Bump to `max_c=50` if `actual_hr < target_hr - 0.01`.

---

## Customizing distribution parameters

Override built-in parameters with `--params /path/to/params.json`.
See `src/params/pro_params.json` for the full schema.

Key fields to override for other ISL/model configs:
```json
{
  "logits": {
    "shallow": {"dist": "gumbel_r", "mu_mean": -1.4, "mu_std": 0.31, "sig_mean": 0.59, "sig_std": 0.07},
    "deep":    {"dist": "norm",     "mu_mean": -1.1, "mu_std": 0.56, "sig_mean": 0.95, "sig_std": 0.18}
  },
  "temporal": {"default_target_hr": 0.69}
}
```

Measure your own parameters with `dsv4-indexer-capture` +
`analyze_qkw_distrib.py` + `per_step_indep_fit_results.json`.

---

## See also

- `dsv4-indexer-capture` — capture real indexer inputs from a running model
- `swebench-temporal-synth-v4pro` — synthesize logits directly (no Q/K, faster)
- `indexer_qkw_capture/REPORT_INDEXER_SYNTH.md` — full methodology report
- `indexer_qkw_capture/analysis/unit_test_inputs_pro_64k.pt` — real-capture unit tests
