---
name: swebench-temporal-synth-v4flash
description: >
  Generate V4-Flash-aligned temporally-coherent synthetic decode logits +
  preIdx for GVR Top-K (K=512) benchmarks. Fits 3 beta distributions to
  real V4 Flash swe-bench captures (21 GVR-active layers, even 2..42;
  shallow/moderate/deep buckets by layer mean). preIdx built via Gaussian
  noise + binary-searched coefficient to hit per-cfg target_hr ≈ 0.36 /
  0.46 / 0.44 (measured average preIdx∩topK across 32K/64K/100K ISL
  captures). Supports variable N (post-compress), BS replication, dtype
  ∈ {fp32, bf16, fp16}. Caller-side preIdx offset = 0 (V4 cr=4); kernel
  uses preIdx[i] directly. radix_aux_{indices,logits} pre-allocated per
  post-#14297 contract. Mirrors swebench-temporal-synth (V3.2) methodology
  but parameterized for V4 Flash. Trigger keywords: "V4 Flash synthetic
  temporal-correlated logits", "V4 Flash GVR Top-K synth K=512", "V4 Flash
  preIdx hit-rate calibration", "swebench-temporal V4 Flash", "synthetic
  V4 Flash kernel bench data".
license: LicenseRef-NvidiaProprietary
metadata:
  author: loncheng@nvidia.com
  parent_skill: swebench-temporal-synth (V3.2 K=2048)
  derived_from: real V4 Flash swe-bench captures (Q9j/Q9k)
  hardware: NVIDIA B200 (sm_100) / B300 (sm_100a)
  K: 512
  compress_ratio: 4
---

# V4-Flash-aligned temporal-coherence synthetic data — skill

## What this skill does

Given a target **post-compress** seq length `N` (e.g. 7530, 14474, 25110
matching V4 Flash 32K/64K/100K captures), a beta cfg, and `K=512`, this
skill emits tensors ready for `torch.ops.trtllm.indexer_topk_decode`:

1. **`logits`** — `[BS, N_padded]` with target dtype (fp32 / bf16 / fp16),
   sampled from one of 3 beta cfgs fitted from real V4 Flash captures
   (21 layers × {32K, 64K, 100K} ISL). 8-element alignment (V4 kernel
   `numColumns % 8 == 0` requirement).
2. **`preIdx`** — `[BS, K=512]` int32, built via Gaussian-noise + binary
   search to realise per-cfg hit_rate ≈ V4 Flash production (0.36–0.46).
   **NO caller-side −1 offset** (V4 cr=4 kernel uses `preIdx[i]`
   directly; see `cpp/tensorrt_llm/kernels/heuristicTopKDecode.cu`:
   `preIdxOffset = (cr==1) ? ((rowIdx % next_n)+1) : 0`).
3. **`seq_lens`** — `[BS] int32`, set to `N * 4 + next_n - 1` so the
   kernel internally computes `row_end[base] = N`.

The skill also bundles `radix_aux_indices` (int32) and `radix_aux_logits`
(fp32, **always** fp32 regardless of input dtype) when `--bench` is used,
covering the post-#14297 kernel contract.

## When to invoke

- "generate V4 Flash random synthetic data for GVR top-K bench"
- "synthesize V4 Flash temporal-correlated logits at N=14K"
- "V4 Flash preIdx with realistic prev-step hit-rate"
- "synthetic V4 Flash data K=512 fp32/bf16/fp16"

Do **not** invoke for:
- V3.2 K=2048 synth → use sibling [`swebench-temporal-synth`](../swebench-temporal-synth/SKILL.md)
- V4 Pro K=1024 synth → use sibling [`swebench-temporal-synth-v4pro`](../swebench-temporal-synth-v4pro/SKILL.md)
- Real V4 captures → use the existing Q9j/Q9k captures under
  `auto_optimization_v1/ablation_study/gvr_phase_timing/09_precision_ablation/11_dsv4_trtllm_indexer_data_capture/data/`

## Parameters

| Flag | Default | Description |
|---|---|---|
| `--N` | `14478` | **Post-compress** seq len (V4 cr=4: 32K ISL → 7530, 64K → 14474, 100K → 25110). Accepts `'14K'`, `'14478'`, etc. Must satisfy `N > 2·K = 1024`. |
| `--cfg` | `beta_moderate` | One of `beta_shallow`, `beta_moderate`, `beta_deep`, or `all`. |
| `--bs` | `1` | Batch size. Rows replicated, not independently sampled (single-cell amortization tests). |
| `--K` | `512` | Top-K. V4 Flash native; exposed for ablation only. |
| `--compress_ratio` | `4` | Indexer compress ratio. V4 default. |
| `--target_hr` | (per-cfg) | Override per-cfg default (shallow=0.36, moderate=0.46, deep=0.44). |
| `--dtype` | `fp32` | Logits dtype: `fp32` / `bf16` / `fp16`. |
| `--seed` | `42` | Random seed. Noise seed is `seed + 1000`. |
| `--max_c` | `5.0` | Upper bound for noise-coefficient binary search. |
| `--outdir` | (required) | Output dir. |
| `--bench` | (off) | Run in-process GVR vs Radix wall measurement with the same compress_ratio/preIdx-offset/radix_aux contract as production. Emits `speedup.txt`. |

## Per-cfg beta parameters (fitted from real captures)

| cfg | mean | std | full_range | target_hr | Source layers |
|---|---:|---:|---:|---:|---|
| `beta_shallow` | −1.315 | 0.605 | 7.31 | 0.36 | L2, L4, L10, L14, L16, L18, L22 (across 32K/64K/100K captures) |
| `beta_moderate` | −2.088 | 0.728 | 9.95 | 0.46 | L6, L8, L12, L30, L34, L40, L42 |
| `beta_deep` | −2.595 | 0.785 | 11.46 | 0.44 | L20, L24, L26, L28, L32, L36, L38 |

Source: `/tmp/dsv4/v4_dist_flash_K512.json` (run-2026-05-21T07:12Z analyzer).

## How to run

### Synth only

```bash
python3 ${SKILL_DIR}/src/synth_temporal_data.py \
    --N 14474 --cfg beta_moderate --bs 1 --dtype bf16 \
    --outdir /tmp/v4flash_synth_64k
```

### Synth + in-process GVR/Radix wall (cuda.Event)

```bash
TRTLLM_HEURISTIC_NMIN=4096 TRTLLM_HEURISTIC_BSMAX=2048 \
  python3 ${SKILL_DIR}/src/synth_temporal_data.py \
    --N 14474 --cfg all --bs 1 --dtype bf16 --bench \
    --outdir /tmp/v4flash_synth_64k_bench
```

### Production-grade nsys + L2-flush + multi-dtype sweep

```bash
# 1. Synthesize bundles (multi-N, multi-cfg)
bash ${SKILL_DIR}/src/run_all_n.sh /tmp/v4flash_synth

# 2. nsys capture with NVTX-bracketed GVR_<dtype> + RADIX_fp32 per cell
cd /tmp/v4flash_synth && nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o nsys_sweep \
    python3 ${SKILL_DIR}/src/bench_nsys.py \
      --indir . --warmup 3 --reps 10 --dtypes fp32,bf16,fp16

# 3. Export NVTX→GPU projection CSV
nsys stats -r nvtx_gpu_proj_trace nsys_sweep.nsys-rep \
    --format csv -o nsys_sweep

# 4. Parse: median µs + R/H per (cfg, N, BS, dtype)
python3 ${SKILL_DIR}/src/parse_nsys.py nsys_sweep_nvtx_gpu_proj_trace.csv
```

## Loading from PyTorch

```python
import torch, json

base = "/tmp/v4flash_synth/beta_moderate_N14474_bs1"
logits = torch.load(f"{base}/logits.pt").cuda()
preIdx = torch.load(f"{base}/preIdx.pt").cuda()
seq_lens = torch.load(f"{base}/seq_lens.pt").cuda()
meta = json.load(open(f"{base}/meta.json"))

K, BS = meta["K"], meta["BS"]
indices = torch.empty((BS, K), dtype=torch.int32, device="cuda")
scratch = torch.empty((BS * K,), dtype=logits.dtype, device="cuda")
# Post-#14297 radix_aux contract (required when blocksPerRow > 1)
radix_aux_indices = torch.empty(
    (BS * 32 * K,), dtype=torch.int32, device="cuda")
radix_aux_logits = torch.empty(
    (BS * 32 * K,), dtype=torch.float32, device="cuda")

torch.ops.trtllm.indexer_topk_decode(
    logits, seq_lens, indices, 1, K,
    pre_idx=preIdx,                       # V4: as-is, NO -1
    heuristic_scratch=scratch,
    compress_ratio=meta["compress_ratio"], # = 4
    radix_aux_indices=radix_aux_indices,
    radix_aux_logits=radix_aux_logits,
)
```

## V4 kernel invariants enforced

| Invariant | Where |
|---|---|
| `K ∈ {512, 1024, 2048}` | Hard-coded in dispatcher; this skill ships K=512 |
| `compress_ratio ∈ {1, 4}` | V4: cr=4 |
| `numColumns % 8 == 0` | `_pad_align_inf(align=8)` (V3.2 sibling used align=4 for fp32 only) |
| `preIdx[i]` directly indexes logits (no caller offset) | cr=4 path bypasses temporal +1 shift |
| `radix_aux_{indices,logits}` pre-allocated for split-work path | `bench_nsys.py` + `synthesize(..., --bench)` allocate `BS · 32 · K` int32 + fp32 (32 = safe blocksPerRow upper bound for ISL ≤ 100K) |
| Dispatcher reaches Heuristic at N ≥ 4096 | Set `TRTLLM_HEURISTIC_NMIN=4096` (default 12288 in source) |
| Dispatcher reaches Heuristic at BS > kBsWave on B200 | Set `TRTLLM_HEURISTIC_BSMAX=2048` (default 426 on B200) |

## Failure modes

| Failure | Behaviour |
|---|---|
| `N ≤ 2·K = 1024` | Aborts; error names K/N ratio |
| `cfg` not in `{beta_shallow, beta_moderate, beta_deep}` | Aborts; lists valid cfgs |
| `--bench` set but `$LIBTH_COMMON` missing | `--bench` exits with ImportError; synth still saves |
| Calibration cannot reach `target_hr` at K/N close to 0.5 | `meta.json.calibration_saturated = true`; realised hr clamped |

## Validated context

This skill's synth approximates real V4 Flash production preIdx behavior.
Expected R/H bracket when fed to the production kernel on B200 should be:

| N | bf16 R/H expected | fp32 R/H expected |
|---:|---:|---:|
|  7530 (32K ISL) | 3.5–4.0× | 3.0–3.5× |
| 14474 (64K ISL) | 2.0–2.5× | 1.9–2.3× |
| 25110 (100K ISL) | 1.7–2.0× | 1.3–1.5× |

Anchor: real-capture nsys numbers from
`/tmp/dsv4/v4_real_bench_20260521T061240/REPORT.md` (B200, May-21).
Synth target ranges align after correcting per-call kernel-time (see
`.perfbot/learnings/20260521T064441-agent.yaml`).

## See also

- **V3.2 sibling**: `swebench-temporal-synth` (K=2048, cr=1, V3.2 layer fits)
- **V4 Pro sibling**: `swebench-temporal-synth-v4pro` (K=1024, V4 Pro fits)
- **Real V4 capture toolbox**:
  `auto_optimization_v1/ablation_study/gvr_phase_timing/09_precision_ablation/{11,12}_dsv4_*_indexer_data_capture/`
- **Kernel invariants**: `cpp/tensorrt_llm/kernels/heuristicTopKDecode.cu`
  + `cpp/tensorrt_llm/kernels/heuristic_topk.cuh`
  + `cpp/tensorrt_llm/kernels/indexerTopK.cu` (dispatcher gates)
- **Dispatcher gate envs**: `TRTLLM_HEURISTIC_NMIN`, `TRTLLM_HEURISTIC_BSMAX`
- **PR #14297 radix_aux contract**: `cpp/tensorrt_llm/thop/IndexerTopKOp.cpp:158-161`
