# op28 — LATEST external top-K arms (SGLang v2 + FlashInfer) on the op22 dataset

Benchmarks the **latest upstream** DeepSeek-V4-style top-K kernels against the
in-tree ops, on **byte-identical inputs and protocol** to the op22 REPORT §1-2
re-test dataset (`op22_temporal_fixed_hr_bench`, bundles_rr + original real).

## Arms
| arm | what | kernels/call |
|---|---|---|
| `gvr_cutedsl` | in-tree baseline #14602 — **anchor** for cross-node transfer | 1 |
| `radix_cutedsl` | in-tree hint-blind rival | 1 |
| `sglang_streaming` | OLD SGLang `top512::StreamingTopK` (op#11 vendor; deleted upstream) | 1 |
| `sglang_v2` | **sglang@main 2026-07-13** v2 architecture (`ops/sglang_v2`, kernels verbatim: register/streaming/persistent-cluster paths, PDL) | 1-2 |
| `flashinfer_topk` | **flashinfer 0.6.11 public `top_k`** (topk.py + B200 clusters kernel byte-identical to main) | 1 |
| `flashinfer_topk_i32` | flashinfer `topk_clusters_exact` minimal contract (int32 indices only) | 1 |

fp32 only (both external kernels are fp32-scores). K 512/1024/2048
(`sglang_streaming` only K<=1024). `sglang_v2`'s `topk_plan` runs UNTIMED at
build (production amortizes it across the ~61 indexer layers) — see ops_ext.py.

## Protocol (identical to op22)
nsys pure-kernel NVTX-range projection; 10 warmup, 50 warm-L2 reps, 20 cold-L2
reps with 512MB evict outside the range; eager+sync in range; cudaProfilerApi
window; 1-row bundle replicated to BS; seq_lens = N.
Extra: `us_span` (nvtx_gpu_proj_sum) recorded because the sglang_v2
persistent-cluster path is 2 PDL-overlapped kernels — kernel-time SUM
double-counts overlap there; span = honest wall-clock.

## Files
- `ops_ext.py` — build_call extension (new arms; delegates old to harness)
- `sweep_op28.py` / `drive_nsys_op28.sh` — sweep under nsys (per-batch rep)
- `gate_op28.py` — exactness pre-gate (459/459 green 2026-07-13)
- `parse_op28.py` — nvtx_kern_sum (+ span) -> results.jsonl
- `gen_results_op28.py` — CSVs + RESULTS_SUMMARY.md (anchor-transferred
  op22rr baselines)
- vendored sources: `../ops/sglang_v2/` (sglang), `../ops/flashinfer_topk/`
  (snapshot of installed==main sources; kernels run from installed package JIT)

## Run
```bash
python3 gate_op28.py
OUT=results_b200_op28 GPU=2 SCENARIOS=real SWEEPS=bs KS="512 1024" ./drive_nsys_op28.sh
python3 parse_op28.py && python3 gen_results_op28.py
```
