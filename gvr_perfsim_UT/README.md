# gvr_perfsim_UT — DSv4 Pro **GVR-only** Top-K perfsim UT (4 cases)

Branch: `tmp/gvr_perfsim_UT` on `longcheng-nv/TensorRT-LLM`
(orphan / empty of upstream TRT-LLM tree).

## Purpose

A pure driver for the GVR (Heuristic) Top-K kernel — no Radix
comparison, no correctness check. The output is a clean, deterministic
GVR kernel execution stream intended for downstream hardware cycle-level
**perfsim**.

Two interchangeable kernel backends are shipped:

| `--backend` | Description |
|---|---|
| **`local`** (default) | Self-contained `gvr_kernel/` torch cpp_extension JIT-built on first import. **No `tensorrt_llm` install required.** Source is `heuristicTopKDecode.cu` + `heuristic_topk.cuh` extracted verbatim from TRT-LLM, with the three common-header references (`TRTLLM_NAMESPACE_*`, `TLLM_CHECK_WITH_INFO`, `getEnvEnablePDL`) stubbed in `gvr_kernel/csrc/trtllm_stubs.h`. |
| `trtllm`    | Calls `torch.ops.trtllm.indexer_topk_decode` through `import tensorrt_llm` (or `LIBTH_COMMON=<path>` fallback). Same kernel, dispatched through the V4 unified op. |

Both backends invoke the **same** `heuristicTopKMultiRowKernel(Dtype)<K>`
templated kernel — they differ only in how the launcher is reached and in
the buffer-allocation contract.

**Target hardware:** NVIDIA **Blackwell** (B200 SXM6 / B300 GB300 / GB200,
sm_100 / sm_103)
**Indexer Top-K:** **K = 1024** (V4 Pro native)

## 4-case matrix

| Case | BS | Seq Len (N, post-compress) | K | dtype | preIdx target hit-rate | Data source |
|---|---:|---:|---:|---|---:|---|
| **case-1a** | **1**   | **65 536** (= 64K) | **1024** | **fp32** | **0.60** | `data/beta_moderate_v4pro_N65536_bs1_hr0.6_fp32/` |
| **case-1b** | **1**   | **65 536** (= 64K) | **1024** | **bf16** | **0.60** | `data/beta_moderate_v4pro_N65536_bs1_hr0.6_bf16/` |
| **case-2a** | **256** | **65 536** (= 64K) | **1024** | **fp32** | **0.60** | same bundle as case-1a (BS=256 is replicated at load time via `--bs 256`) |
| **case-2b** | **256** | **65 536** (= 64K) | **1024** | **bf16** | **0.60** | same bundle as case-1b (BS=256 is replicated at load time via `--bs 256`) |

> **About "Seq Len" semantics:** `N` is the post-compress
> (`compress_ratio = 4`) length the indexer kernel actually consumes —
> 64K is the V4 Pro typical swe-bench prompt scale after cr=4 compression.
> Change `--N` in `synth_pro_data.py` for a different N; the synth refuses
> `N ≤ 2·K` (GVR requires `N > 2K`).

> **About half-precision (bf16) generation:** following the V4 Pro
> "Option A" recipe — sample + calibrate preIdx in fp32, then cast the
> saved logits via `.to(torch.bfloat16)` RNE truncation. preIdx (int32)
> is unchanged. This mirrors V4 production semantics: previous-step fp32
> preIdx fed into a current-step half-precision kernel.

> **About BS=256 data replication:** only one BS=1 bundle is committed
> per dtype. At bench time, `bench_gvr_topk.py` does
> `logits.expand(256, -1).contiguous()` on-device — exactly matching the
> user spec ("the BS dimension literally replicates the same data").

## File layout

```
gvr_perfsim_UT/
├── README.md                       ← this file
├── synth_pro_data.py               ← Data generator (V4 Pro typical dist +
│                                    temporal-coherence preIdx + dtype switch).
│                                    Reference implementation only — the 4
│                                    cases' input data is already committed.
├── bench_gvr_topk.py               ← Pure GVR driver (load .pt → warmup →
│                                    reps → median µs). No Radix, no
│                                    correctness check.  --backend local|trtllm
├── run_case1a_BS1_fp32.sh          ← one-shot runner for case-1a
├── run_case1b_BS1_bf16.sh          ← one-shot runner for case-1b
├── run_case2a_BS256_fp32.sh        ← one-shot runner for case-2a
├── run_case2b_BS256_bf16.sh        ← one-shot runner for case-2b
├── gvr_kernel/                     ← Standalone GVR kernel torch cpp_extension
│   ├── __init__.py                  (JIT-builds on first import; no `tensorrt_llm` dep)
│   ├── setup.py                     (optional: `pip install -e gvr_kernel/`)
│   └── csrc/
│       ├── heuristic_topk.cuh         (verbatim from TRT-LLM, 1809 lines)
│       ├── heuristicTopKDecode.cu     (adapted launcher; 3 common-header refs stubbed)
│       ├── heuristicTopKDecode.h
│       ├── trtllm_stubs.h             (stubs for TRTLLM_NAMESPACE_* / TLLM_CHECK_WITH_INFO / getEnvEnablePDL)
│       └── binding.cpp                (pybind11 / torch op wrapper)
├── data/                           ← (committed) two BS=1 physical bundles
│   ├── beta_moderate_v4pro_N65536_bs1_hr0.6_fp32/
│   │   ├── logits.pt       [1, 65 536]   fp32   (~258 KB)
│   │   ├── preIdx.pt       [1, 1024]     int32  (~6 KB)
│   │   ├── seq_lens.pt     [1]           int32  = N*4 + 0 = 262144
│   │   └── meta.json       cfg, calibrated noise c, realised hit-rate, V4 invariants
│   └── beta_moderate_v4pro_N65536_bs1_hr0.6_bf16/
│       ├── logits.pt       [1, 65 536]   bf16   (~130 KB)
│       ├── preIdx.pt       [1, 1024]     int32  (~6 KB, byte-identical to fp32 sibling)
│       ├── seq_lens.pt     [1]           int32
│       └── meta.json
└── results/                        ← (written by runs, .gitignore'd)
    ├── case1a_BS1_N65536_fp32.json
    ├── case1b_BS1_N65536_bf16.json
    ├── case2a_BS256_N65536_fp32.json
    └── case2b_BS256_N65536_bf16.json
```

## Prerequisites

1. **GPU (Blackwell only)** — single NVIDIA **B200 SXM6** (sm_100, 126 MiB L2,
   148 SM) or **B300 GB300 / GB200** (sm_103, 190 MiB L2, 148 SM).
   - The V4 GVR Heuristic Top-K kernel `heuristicTopKMultiRowKernel<1024>`
     in `gvr_kernel/csrc/heuristic_topk.cuh` is explicitly tabulated for
     Blackwell with `kFTarget = K = 1024`, `kC = 5120`, `kNumBins = …`
     etc. Running on non-Blackwell may execute but is **not** the tuned
     target and is unsuitable for perfsim.
2. **PyTorch with CUDA** — version ≥ 2.4 (validated against `torch 2.11`).
3. **CUDA toolkit** — `nvcc` available for the local backend's JIT build.
   The local backend hard-codes `sm_100,sm_103` arch flags; override with
   `GVR_KERNEL_ARCH_FLAGS="..."` if you target a different arch.
4. **TensorRT-LLM (optional, `--backend trtllm` only)** — must be able to
   `import tensorrt_llm` to register the custom op
   `torch.ops.trtllm.indexer_topk_decode`. If `import tensorrt_llm` fails
   in a bare environment (e.g. transformers symbol conflicts), set
   `LIBTH_COMMON` to let `bench_gvr_topk.py` fall back to a direct shared
   object load:
   ```bash
   export LIBTH_COMMON=/path/to/TensorRT-LLM/cpp/build/tensorrt_llm/thop/libth_common.so
   # or
   export LIBTH_COMMON=/path/to/TensorRT-LLM/tensorrt_llm/libs/libth_common.so
   ```
5. **Kernel env knobs** (the runner sets these with `setdefault`):
   ```bash
   TRTLLM_HEURISTIC_NMIN=1024   # let GVR Heuristic fire at N ≥ 1024  [trtllm backend]
   TRTLLM_HEURISTIC_BSMAX=2048  # let GVR Heuristic fire at BS ≤ 2048 [trtllm backend]
   TRTLLM_ENABLE_PDL=1          # CUDA programmatic stream serialization (both backends)
   ```

## How to run

### Run all 4 cases

```bash
cd gvr_perfsim_UT
# Default --backend local. JIT-builds gvr_kernel/ on first invocation
# (~30-60 s); subsequent runs reuse the cached .so.
bash run_case1a_BS1_fp32.sh
bash run_case1b_BS1_bf16.sh
bash run_case2a_BS256_fp32.sh
bash run_case2b_BS256_bf16.sh

# To compare against the in-tree TRT-LLM unified op:
BACKEND=trtllm bash run_case1a_BS1_fp32.sh
```

### Standalone kernel use (no runner script)

```python
import torch
from gvr_perfsim_UT.gvr_kernel import gvr_topk_decode, gvr_topk_decode_into

logits   = torch.randn(1, 65536, dtype=torch.float32, device='cuda')
preIdx   = torch.arange(1024, dtype=torch.int32, device='cuda').unsqueeze(0)
seq_lens = torch.tensor([65536*4], dtype=torch.int32, device='cuda')

# Convenience entry point — allocates output every call.
indices, values = gvr_topk_decode(logits, preIdx, seq_lens,
                                  K=1024, compress_ratio=4, next_n=1)

# perfsim-grade entry point — caller pre-allocates outputs once and
# reuses them across reps. Matches the TRT-LLM op contract exactly.
indices_out = torch.empty((1, 1024), dtype=torch.int32, device='cuda')
values_out  = torch.empty((1, 1024), dtype=torch.float32, device='cuda')
for _ in range(100):
    gvr_topk_decode_into(logits, preIdx, seq_lens,
                         indices_out, values_out,
                         K=1024, compress_ratio=4, next_n=1)
```

### Expected output (Blackwell, verified in this repo)

| Case | `--backend local` median | `--backend trtllm` median |
|---|---:|---:|
| case-1a BS=1   fp32 | ~39 µs | ~36 µs |
| case-1b BS=1   bf16 | ~31 µs | ~30 µs |
| case-2a BS=256 fp32 | ~58 µs | ~54 µs |
| case-2b BS=256 bf16 | ~41 µs | ~42 µs |

The local backend lands within ≤ 3 µs of the in-tree TRT-LLM op — same
kernel SASS, same launch contract (both use `gvr_topk_decode_into`-style
caller-provided output buffers, no per-call `torch::empty` / `copy_()`).
The remaining residual is pure pybind11 vs torch op dispatch overhead
(~1-3 µs at this BS / dtype). For cycle-accurate perfsim either backend
is sound — pick `local` to drop the TRT-LLM dependency, `trtllm` when
you want to A/B against the production op path directly.

These match the order of magnitude reported by the upstream sweep
`13_v4_synth_sweep_AB` (Pro K=1024):

- BS=1 N=64K bf16: 18.07 µs (B200, same-kernel build); this repo on a
  B-class card lands at ~28 µs (kernel-build deltas).
- BS=256 N=64K bf16: 29.10 µs (B200 reference).
- bf16 is consistently 15–24% faster than fp32 (half the L2 footprint for
  the heuristic kernel, matching the Q19c cross-dtype findings).

### Tunable knobs

```bash
WARMUP=50 REPS=200 L2_FLUSH_MIB=192 BACKEND=local  bash run_case1a_BS1_fp32.sh
WARMUP=50 REPS=200 L2_FLUSH_MIB=192 BACKEND=trtllm bash run_case1a_BS1_fp32.sh
```

### Driving a cycle-level perfsim

Each launch is an independent, deterministic GVR Heuristic kernel
execution (cold L2, fixed input, fixed preIdx) — exactly the shape a
cycle simulator wants. Examples:

```bash
# One launch only (perfsim wants a single kernel to model)
python3 bench_gvr_topk.py \
    --case-dir ./data/beta_moderate_v4pro_N65536_bs1_hr0.6_fp32 \
    --bs 1 --warmup 0 --reps 1

# nsys, gated to a single steady-state launch window
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o gvr_case2a python3 bench_gvr_topk.py \
    --case-dir ./data/beta_moderate_v4pro_N65536_bs1_hr0.6_fp32 \
    --bs 256 --warmup 5 --reps 20

# Nsight Compute, full-set HW counters for a single kernel
ncu --set full --target-processes all \
    -o gvr_case1b_ncu --replay-mode kernel \
    --launch-skip 30 --launch-count 1 \
    python3 bench_gvr_topk.py \
    --case-dir ./data/beta_moderate_v4pro_N65536_bs1_hr0.6_bf16 \
    --bs 1 --warmup 30 --reps 1
```

## Regenerating the data (`synth_pro_data.py`, supplemental)

The 4 cases' input data is already committed under `data/`; the generator
script is kept only as a reference implementation. Re-generate with
different knobs (N / hit-rate / dtype / cfg bucket) when needed:

```bash
# fp32
python3 synth_pro_data.py --N 65536 --bs 1 --K 1024 --target_hr 0.6 --dtype fp32 \
    --outdir ./data/beta_moderate_v4pro_N65536_bs1_hr0.6_fp32
# bf16
python3 synth_pro_data.py --N 65536 --bs 1 --K 1024 --target_hr 0.6 --dtype bf16 \
    --outdir ./data/beta_moderate_v4pro_N65536_bs1_hr0.6_bf16
```

Switching the distribution bucket: edit `BETA_MODERATE` in
`synth_pro_data.py`; the three buckets used by the in-tree
`swebench-temporal-synth-v4pro` skill are:

```python
"beta_shallow":  dict(mean=-1.184, std=0.864, clip_low=-4.54, clip_high=7.33)   # shallow layers
"beta_moderate": dict(mean=-1.885, std=1.025, clip_low=-6.15, clip_high=8.45)   # mid layers (default)
"beta_deep":     dict(mean=-2.590, std=0.870, clip_low=-5.42, clip_high=6.47)   # deep layers
```

## Method notes

- **Logits sampling:** N samples drawn from the V4 Pro typical logit
  distribution (`beta_moderate`, fitted from real swe-bench captures
  pooled across the 30 GVR-active layers).
- **preIdx construction:** add i.i.d. Gaussian noise scaled by a
  calibrated `c · σ` to the same row, then take the noised row's top-K as
  `prev_topk`, then use that directly as `preIdx[i]` (V4 caller offset
  = 0). Binary-search `c` until
  `|preIdx ∩ topk(row)| / K ≈ 0.60`.
- **Argmax invariant:** if the row's argmax is not in `prev_topk`, force
  it into `prev_topk[-1]` so the kernel can read `logits[argmax]` via a
  preIdx slot.
- **Half-precision path:** sample + calibrate in fp32, then
  `.to(torch.bfloat16)` RNE-truncate the saved logits. preIdx unchanged.
- **L2 hygiene:** every launch is preceded by `zero_()` on a 128 MiB fp32
  buffer (cold-L2 baseline).
- **Timing:** `torch.cuda.Event(enable_timing=True)`, 100 reps, report
  median µs.
- **No correctness check** (intentional — perfsim does not need it). The
  output `indices` are written into a pre-allocated device buffer and
  never read back to host.

## Upstream sweeps using the same methodology

`https://sc.talos.nvidia.com/view/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1/ablation_study/gvr_phase_timing/`

- `13_v4_synth_sweep_AB/` — Pro K=1024 + Flash K=512, paired Seq × BS × dtype sweep.
- `09_precision_ablation/09_synth_temporal_KxNxDtype_bs1/` (Q19c) — K = 512 / 1024 / 2048, three dtypes, five N values, full grid.
- `_synth_full_sweep/K1024/` — raw synthetic bundle (the methodology origin of this UT).
