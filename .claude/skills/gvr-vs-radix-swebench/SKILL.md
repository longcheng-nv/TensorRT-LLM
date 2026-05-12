---
name: gvr-vs-radix-swebench
description: >
  Run an end-to-end performance comparison of the GVR (Heuristic-Guided)
  Top-K kernel against the TRT-LLM Radix Top-K kernel on the 9-layer
  SWE-Bench-64K decode logits dataset (DeepSeek V3.2). Produces nsys-measured
  per-row kernel times and emits an English report covering BS=1 (real
  variable-length rows, per-row valid-N masked) and BS-scaling (BS=1..512,
  last row N=70690 replicated). Reports per-BS min / max / mean speedup
  across the 9 layers. Trigger keywords: "compare GVR vs radix on SWE-bench",
  "GVR vs radix-based top-K speedup", "indexer topk decode 9-layer benchmark",
  "BS scaling GVR radix", "real-data top-K speedup", "heuristic top-K
  production perf", "SWE-Bench 9-layer top-K perf report".
license: LicenseRef-NvidiaProprietary
metadata:
  author: loncheng@nvidia.com
  hardware: NVIDIA Blackwell (B200 sm_100 / B300 sm_100a)
  cluster: computelab-sc-01 (SWE-Bench-64K raw logits only live here)
  dataset: SWE_Bench_64K_decode_logits (9 layers x 2025 rows x ~70690 fp32)
---

# GVR Top-K vs Radix Top-K on SWE-Bench-64K -- Benchmark Skill

## What this skill does

Given a working TRT-LLM build that exposes
`torch.ops.trtllm.indexer_topk_decode`, this skill:

1. **Verifies** the kernel internally applies `preIdxOffset = +1` (V3.2
   decode semantics). If not, it stops and reports the issue.
2. **Profiles BS=1** with the 9-layer real SWE-Bench-64K data. Each row uses
   its true valid-length (`N_valid = 70690 - (2024 - rowId)`); rows
   beyond `N_valid` are masked to `-inf`. `preIdx` is the **exact** Top-K of
   the previous row in the same layer (sorted descending so `preIdx[0]` is
   the argmax, as required by the indexer invariant). nsys captures every
   per-row kernel launch; per-row durations are reduced to a per-(layer,
   variant) median, and per-layer Radix/GVR speedups are aggregated across
   the 9 layers (min / max / mean).
3. **Profiles BS-scaling** at BS in `{1,2,4,8,16,32,64,128,256,512}` (user
   configurable). For each layer, the **last** row (`rowId = 2024`,
   `N_valid = 70690`) is replicated BS times. Per-BS speedups are aggregated
   across the 9 layers (min / max / mean).
4. **Renders** an English `REPORT.md` with both BS=1 and BS-scaling tables
   plus a short interpretation.

The default GVR variant is **`gvr_K2048_fp32`** -- the only apples-to-apples
match against the Radix baseline (`radix_K2048_fp32`). Other K x dtype
variants can be added via `--variants`.

## When to invoke

Triggered automatically when the user asks for a performance comparison
between GVR and Radix-based Top-K on SWE-Bench / DeepSeek V3.2 real data.
Sample phrasings:

- "compare GVR top-K against radix top-K on SWE-bench"
- "what is the GVR-vs-radix speedup on the 9 layers?"
- "run BS-scaling top-K perf experiment"
- "real-data indexer top-K BS=1 vs BS=512 perf report"

Do **not** invoke for purely synthetic / random-data sweeps -- use the
`gpu-benchmarking` skill instead.

## Prerequisites

### Environment

- **GPU**: NVIDIA **Blackwell** family -- sm_100 (B200) or sm_100a (B300).
  The TRT-LLM `libth_common.so` and the local `cuda_ext/topk_ext.{cpp,cu}`
  JIT extension are both compiled targeting `sm_100`. Earlier
  architectures (Hopper sm_90, Ampere sm_80) lack the cluster / dynamic
  shared-memory size required by the multi-row GVR kernel and will fail to
  launch.
- **Cluster**: **`computelab-sc-01`** -- the SWE-Bench-64K raw decode
  logits (`Layer_{0,1,20,21,22,40,41,42,60}_pd.npy`) only live on this
  cluster, under `/home/scratch.loncheng_gpu/workspace/tllm_toolbox/...`.
  Other hosts have only the per-row distribution-fit CSVs, not the raw
  fp32 logits the skill needs. Use the `computelab-sc-01-launch` skill /
  Slurm to allocate a Blackwell node before running.

### Software

- `nsys` CLI on `PATH` (Nsight Systems 2024.x+).
- TRT-LLM repo built with the GVR kernel and Scheme X dispatcher:
  `$TRTLLM_REPO/tensorrt_llm/libs/libth_common.so` must exist. Required
  for the Radix half of the comparison **regardless** of `--gvr_backend`
  (no standalone Radix exists in this codebase).
- Standalone GVR backend only: `auto_optimization_v1/` checkout with
  `topk_cuda.py` + `cuda_ext/topk_ext.{cpp,cu}` (`$GVR_STANDALONE_REPO`,
  default `/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1`).

### Data

- SWE-Bench-64K decode logits directory containing
  `Layer_{0,1,20,21,22,40,41,42,60}_pd.npy`. The skill defaults to
  `$SWE_BENCH_PATH` (override per host) and falls back to
  `/home/scratch.loncheng_gpu/workspace/tllm_toolbox/indexer_topK_perf/data_distri/deepseek-v3.2-logging/notebooks/SWE_Bench_64K_decode_logits`
  on `computelab-sc-01`.

## GVR backends

| `--gvr_backend` | GVR source | Kernel preIdxOffset | preIdx shift in Python | When to use |
|---|---|---|---|---|
| `production` (default) | `torch.ops.trtllm.indexer_topk_decode` -> `heuristicTopKMultiRowKernel{,Dtype}` in `libth_common.so` | `+1` baked in | **none** -- bench feeds raw prev-row Top-K | The numbers you'd put in a PR / release note. F006 production semantics. |
| `standalone` | `heuristic_topk_cuda` in `auto_optimization_v1/topk_cuda.py` (local JIT) | `0` (kernel default) | **`+1`** -- bench adds 1 to every preIdx index, then clamps to N_valid-1 | Bench the standalone GVR kernel while keeping the **same effective offset semantics** as the production kernel. Useful for V2e / V2d sanity checks before pushing into TRT-LLM. |

Radix always lives in TRT-LLM (no standalone Radix exists in this repo);
both backends therefore require `$TRTLLM_REPO`. Standalone GVR also
requires `$GVR_STANDALONE_REPO` (defaults to the `auto_optimization_v1/`
checkout).

`standalone` mode is **single-row**: BS>1 cells are served by a Python
`for` loop inside the timed region, so the BS-scaling numbers in standalone
mode are not apples-to-apples vs the production multi-row kernel at BS>1.
BS=1 *is* apples-to-apples on the algorithm itself (same kernel family,
same +1 effective offset, only the launch surface differs).

## How to run

One-shot wrapper (recommended):

```bash
# Production GVR vs TRT-LLM Radix (default)
bash ${SKILL_DIR}/src/run_full.sh \
    --outdir  /tmp/gvr_vs_radix_swebench \
    --bs_list 1,2,4,8,16,32,64,128,256,512 \
    --row_stride 10 \
    --warmup 3 --repeats 5

# Standalone GVR (with Python +1 shift) vs TRT-LLM Radix
bash ${SKILL_DIR}/src/run_full.sh \
    --outdir       /tmp/gvr_vs_radix_swebench_standalone \
    --gvr_backend  standalone \
    --bs_list      1,2,4,8,16,32,64,128,256,512 \
    --row_stride   10 --warmup 3 --repeats 5
```

The wrapper performs, in order:

1. **`verify_preidx_offset.py`** -- greps `$TRTLLM_REPO/cpp/tensorrt_llm/kernels/heuristicTopKDecode.cu` for the literal `int const preIdxOffset = (rowIdx % next_n) + 1;` on both the fp32 path (`heuristicTopKMultiRowKernel`) and the half-precision path (`heuristicTopKMultiRowKernelDtype`). Exit non-zero with an explicit error if either is missing -- this is the user-required preIdxOffset audit (kernel must internally apply +1 column shift on top of the prev-row preIdx).
2. **`bench_bs1.py profile`** under `nsys profile --trace=cuda,nvtx` -- 9 layers x stride-sampled rows x [radix, gvr] x repeats.
3. `nsys stats --report nvtx_gpu_proj_trace --format csv` -> `bench_bs1.py parse` -> per-row CSV + per-(layer, variant) median.
4. **`bench_bs_scaling.py profile`** under nsys -- last row of each layer replicated for each BS, 9 layers x BS_list x [radix, gvr] x repeats.
5. `nsys stats` -> `bench_bs_scaling.py parse` -> per-(BS, layer, variant) CSV.
6. **`render_report.py`** -- builds `${OUTDIR}/REPORT.md` summarising:
   - **BS=1 table**: per-layer median us for Radix and GVR + Radix/GVR speedup. Footer row gives **min**, **max**, **mean** speedup across the 9 layers.
   - **BS-scaling table**: rows = BS values, columns = min / max / mean GVR-vs-Radix speedup across the 9 layers.
   - Plus the absolute baseline numbers, the test environment, and a one-line interpretation.

Sub-commands can also be run individually -- see each script's `--help`.

## What the skill outputs

After `run_full.sh` finishes, `${OUTDIR}/` contains:

```
preidx_offset_check.txt      Result of the +1 offset audit
nsys_bs1.nsys-rep            BS=1 profile
nsys_bs1_nvtx_gpu_proj_trace.csv
bs1_index.json
bs1_results_raw.csv          Per-(layer, row, variant) median us
bs1_results_summary.csv      Per-(layer, variant) pooled median us
nsys_bs_scaling.nsys-rep     BS-scaling profile
nsys_bs_scaling_nvtx_gpu_proj_trace.csv
bs_scaling_index.json
bs_scaling_results_raw.csv   Per-(layer, BS, variant) median us
bs_scaling_results_summary.csv
REPORT.md                    Final English report (entry point)
```

`REPORT.md` is the user-facing artifact. It is written in English and
contains all aggregated min / max / mean speedups, ready to paste into a
PR description or follow-up issue.

## Key invariants enforced

| Invariant | Where enforced |
|---|---|
| Effective preIdxOffset = +1 (V3.2 decode) | `verify_preidx_offset.py`. Production: greps for `(rowIdx % next_n) + 1` in TRT-LLM. Standalone: greps for `/*preIdxOffset=*/0` in `auto_optimization_v1/heuristic_topk.cuh` (the bench compensates by adding +1 in Python before each call). |
| `preIdx[0]` = argmax of prev row | `prep_inputs` in `bench_bs1.py` / `bench_bs_scaling.py` (`torch.topk(..., sorted=True)`) |
| Per-row valid-N masking on BS=1 path | `prep_inputs` zero-pads `[N_valid:] = -inf` |
| Last-row N=70690 used for BS-scaling | `bench_bs_scaling.py` always picks `row_id = lscore.shape[0]-1` |
| Same effective semantics across backends | Production: kernel adds +1 internally. Standalone: bench adds +1 in Python, clamps to `N_valid - 1`, then calls the standalone kernel (which uses `preIdxOffset=0`). |
| Per-iter L2 flush (128 MiB > B200 126.5 MiB) | `flush_l2()` zeros a > L2-cap buffer before every timed launch |

## Failure modes and how the skill handles them

| Failure | Behaviour |
|---|---|
| `preIdxOffset = +1` missing from kernel source | `verify_preidx_offset.py` exits 2 with a banner: "GVR kernel does NOT bake preIdxOffset=+1 -- production semantics broken, aborting." |
| Missing `libth_common.so` / `torch.ops.trtllm` unavailable | `bench_*.py` raises a clear ImportError with the expected `$TRTLLM_REPO` value |
| Missing layer .npy files | `load_layer()` raises FileNotFoundError naming the missing layer |
| nsys not on PATH | `run_full.sh` aborts with an explicit "install Nsight Systems" message |
| Empty NVTX CSV (e.g. nsys produced no projection trace) | `parse_nvtx_csv()` raises a RuntimeError listing the offending CSV path |

## Notes on methodology

- **Timing source**: nsys `nvtx_gpu_proj_trace` -- per-kernel projected GPU
  duration. CUDA events would also work, but nsys gives independent
  cross-verification and is the standard for production benchmarks in this
  repo (see `09_precision_ablation/05_*` and `04b_*`).
- **L2 flush** before every iteration ensures the timing reflects the
  cold-cache reality of decode-time top-K (each new token has a different
  logit row).
- **NVTX tagging** uses `L{layer}/row{r}/{variant}/rep{i}` for BS=1 and
  `L{layer}/bs{BS}/{variant}/rep{i}` for BS-scaling. The parser is strict:
  malformed ranges are skipped silently, so stray nsys CUDA-runtime ranges
  do not contaminate the medians.
- **Per-layer aggregation**: median across all sampled rows / reps within a
  layer; per-BS aggregation is min / max / mean of the **per-layer** speedups
  (9 values). This matches the convention in
  `09_precision_ablation/04b_*/REPORT.md`.
