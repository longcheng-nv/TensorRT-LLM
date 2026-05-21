---
name: swebench-temporal-synth-v4pro
description: >
  Generate V4-Pro-aligned temporally-coherent synthetic decode logits +
  preIdx for GVR Top-K (K=1024) benchmarks. Fits 3 beta distributions to
  real V4 Pro swe-bench captures (30 GVR-active layers, even 2..60;
  shallow/moderate/deep buckets by layer mean). preIdx built via Gaussian
  noise + binary-searched coefficient to hit per-cfg target_hr ≈ 0.69 /
  0.75 / 0.77 (measured average preIdx∩topK across 32K/64K ISL
  captures — substantially higher than V4 Flash's 0.36-0.46). Supports
  variable N (post-compress), BS replication, dtype ∈ {fp32, bf16, fp16}.
  Caller-side preIdx offset = 0 (V4 cr=4); kernel uses preIdx[i]
  directly. radix_aux_{indices,logits} pre-allocated per post-#14297
  contract. Mirrors swebench-temporal-synth (V3.2) methodology but
  parameterized for V4 Pro. Trigger keywords: "V4 Pro synthetic
  temporal-correlated logits", "V4 Pro GVR Top-K synth K=1024",
  "V4 Pro preIdx hit-rate calibration", "swebench-temporal V4 Pro",
  "synthetic V4 Pro kernel bench data".
license: LicenseRef-NvidiaProprietary
metadata:
  author: loncheng@nvidia.com
  parent_skill: swebench-temporal-synth (V3.2 K=2048)
  derived_from: real V4 Pro swe-bench captures (Q9k)
  hardware: NVIDIA B200 (sm_100) / B300 (sm_100a)
  K: 1024
  compress_ratio: 4
---

# V4-Pro-aligned temporal-coherence synthetic data — skill

## What this skill does

Same shape as the V4 Flash sibling, but tuned to V4 Pro production:

| Knob | V4 Flash sibling | **V4 Pro (this skill)** |
|---|---|---|
| K | 512 | **1024** |
| beta cfgs | 21-layer Flash fits | **30-layer Pro fits** |
| target_hr (default per cfg) | 0.36 / 0.46 / 0.44 | **0.69 / 0.75 / 0.77** |
| Production hint | weaker prev-step correlation | **much stronger** (Pro decode steps re-use ≈75 % of prev top-K) |

Outputs:
1. **`logits`** — `[BS, N_padded]` target dtype (fp32 / bf16 / fp16),
   sampled from one of 3 Pro beta cfgs. 8-element alignment.
2. **`preIdx`** — `[BS, K=1024]` int32, no caller offset (V4 cr=4
   contract). Hit-rate calibrated to per-cfg V4 Pro target.
3. **`seq_lens`** — `[BS] int32 = N * 4 + next_n - 1`.

## When to invoke

- "generate V4 Pro random synthetic data for GVR top-K bench"
- "synthesize V4 Pro temporal-correlated logits at N=14K"
- "V4 Pro preIdx with realistic prev-step hit-rate"
- "synthetic V4 Pro data K=1024 fp32/bf16/fp16"

Do **not** invoke for:
- V3.2 K=2048 synth → use `swebench-temporal-synth`
- V4 Flash K=512 synth → use `swebench-temporal-synth-v4flash`

## Parameters

| Flag | Default | Description |
|---|---|---|
| `--N` | `14478` | Post-compress seq len (32K ISL → 7557, 64K → 14460, 100K → ~25100). Must satisfy `N > 2·K = 2048`. |
| `--cfg` | `beta_moderate` | `beta_shallow` / `beta_moderate` / `beta_deep` / `all`. |
| `--bs` | `1` | Batch size (rows replicated). |
| `--K` | `1024` | Top-K. V4 Pro native. |
| `--compress_ratio` | `4` | V4 default. |
| `--target_hr` | (per-cfg) | Override per-cfg default (shallow=0.69, moderate=0.75, deep=0.77). |
| `--dtype` | `fp32` | `fp32` / `bf16` / `fp16`. |
| `--seed` | `42` | |
| `--max_c` | `5.0` | Upper bound for noise-coefficient binary search. |
| `--outdir` | (required) | |
| `--bench` | (off) | In-process GVR vs Radix wall measurement. |

## Per-cfg beta parameters (fitted from real captures)

| cfg | mean | std | full_range | target_hr | Source layers |
|---|---:|---:|---:|---:|---|
| `beta_shallow` | −1.184 | 0.864 | 8.76 | **0.69** | L2, L6, L12, L14, L16, L18, L24, L30, L32, L60 |
| `beta_moderate` | −1.885 | 1.025 | 10.04 | **0.75** | L8, L10, L20, L22, L34, L36, L42, L44, L48, L58 |
| `beta_deep` | −2.590 | 0.870 | 9.74 | **0.77** | L4, L26, L28, L38, L40, L46, L50, L52, L54, L56 |

Source: `/tmp/dsv4/v4_dist_pro_K1024.json`.

> **Note on the high hit_rate**: V4 Pro decode steps show much stronger
> prev-step / current-step preIdx overlap (~75 %) than V4 Flash (~40 %).
> This is consistent with Pro's larger model + more stable indexer
> attention patterns. The synth's binary search may saturate at small
> `--max_c` for the deep cfg — bump `--max_c` to 10 if `meta.json`
> reports `calibration_saturated: true`.

## How to run

### Synth only

```bash
python3 ${SKILL_DIR}/src/synth_temporal_data.py \
    --N 14460 --cfg beta_moderate --bs 1 --dtype bf16 \
    --outdir /tmp/v4pro_synth_64k
```

### Multi-cfg sweep + bench

```bash
bash ${SKILL_DIR}/src/run_all_n.sh /tmp/v4pro_synth
```

### Production-grade nsys multi-dtype sweep

```bash
cd /tmp/v4pro_synth && nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o nsys_sweep \
    python3 ${SKILL_DIR}/src/bench_nsys.py \
      --indir . --warmup 3 --reps 10 --dtypes fp32,bf16,fp16

nsys stats -r nvtx_gpu_proj_trace nsys_sweep.nsys-rep \
    --format csv -o nsys_sweep

python3 ${SKILL_DIR}/src/parse_nsys.py nsys_sweep_nvtx_gpu_proj_trace.csv
```

## Loading from PyTorch

```python
import torch, json

base = "/tmp/v4pro_synth/beta_moderate_N14460_bs1"
logits = torch.load(f"{base}/logits.pt").cuda()
preIdx = torch.load(f"{base}/preIdx.pt").cuda()
seq_lens = torch.load(f"{base}/seq_lens.pt").cuda()
meta = json.load(open(f"{base}/meta.json"))

K, BS = meta["K"], meta["BS"]  # K = 1024
indices = torch.empty((BS, K), dtype=torch.int32, device="cuda")
scratch = torch.empty((BS * K,), dtype=logits.dtype, device="cuda")
radix_aux_indices = torch.empty(
    (BS * 32 * K,), dtype=torch.int32, device="cuda")
radix_aux_logits = torch.empty(
    (BS * 32 * K,), dtype=torch.float32, device="cuda")

torch.ops.trtllm.indexer_topk_decode(
    logits, seq_lens, indices, 1, K,
    pre_idx=preIdx,
    heuristic_scratch=scratch,
    compress_ratio=meta["compress_ratio"],
    radix_aux_indices=radix_aux_indices,
    radix_aux_logits=radix_aux_logits,
)
```

## V4 kernel invariants enforced

Same as V4 Flash sibling — see that SKILL.md for the full table.
Key Pro-specific point: **K=1024 changes the dispatcher `isSupportedTopK`
check** (`K ∈ {512, 1024, 2048}`); the synth verifies `K == 1024` on
load.

## Failure modes

| Failure | Behaviour |
|---|---|
| `N ≤ 2·K = 2048` | Aborts |
| Calibration saturates at default `max_c=5.0` for deep cfg | Re-run with `--max_c 10` |
| Pro 100K real-capture is missing (eval failed) | Synth still works at N=25100, but no real-data anchor for validation |

## Validated context

Expected R/H bracket when fed to the production kernel on B200:

| N | bf16 R/H expected | fp32 R/H expected |
|---:|---:|---:|
|  7557 (32K ISL) | 2.7–3.0× | 2.3–2.6× |
| 14460 (64K ISL) | 2.2–2.5× | 1.9–2.2× |
| ~25100 (100K ISL, no real anchor) | TBD | TBD |

Anchor: real-capture nsys numbers from
`/tmp/dsv4/v4_real_bench_20260521T061240/REPORT.md`.

## See also

- V4 Flash sibling: `swebench-temporal-synth-v4flash`
- V3.2 sibling: `swebench-temporal-synth`
- Real V4 Pro captures (32K + 64K only):
  `auto_optimization_v1/ablation_study/gvr_phase_timing/09_precision_ablation/12_dsv4_pro_indexer_data_capture/data/capture_20260520T16{2818,4146}Z_v4pro_K1024_*/`
