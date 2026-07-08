# HLS Top-K decode operator (op21 production-GVR campaign, iter16 ship state)

Standalone source snapshot of the **HLS (h-tracked ladder-secant) indexer top-K
decode operator** for DeepSeek-V3.2 / V4 (Flash / Pro) on NVIDIA Blackwell
(sm_100), extracted from the op-bench workspace. This is the baseline for the
**op25 optimization campaign** (expanding the HLS win region vs
radix-cuteDSL / SGLang StreamingTopK); subsequent commits on this branch are
the campaign's optimization iterations.

Baseline provenance: workspace branch `omni/op21-gvr-prod` @ `8bb22d1daf`
(op21 iter16 `51be558e77`: HLS all three steps on silicon — log-falsi
fallback + distributed msc fallback + `_fb_dist` code-mass diet, default
gate `n >= 65536`).

## Layout

| File | Role |
|---|---|
| `indexer_topk_op_bench/op21_gvr_prod/src/gvr_ms_op.py` | Single-CTA mode-5 rank-quantile sandwich kernel (`gvr_ms`), M=4 ladder (qfracs 0.75/0.5/0.25 of K_valid), log-falsi fallback, P4 rank-scatter |
| `indexer_topk_op_bench/op21_gvr_prod/src/gvr_msc_op.py` | Row-chunked multi-CTA cluster variant (`gvr_msc`, C=4/8, DSMEM count merge, iter7 P3 remote-store push, iter14/16 distributed fallback) + `gvr_ms_auto` dispatch |
| `indexer_topk_op_bench/op18_gvr_1cta_multithresh/src/gvr_mt_op.py` | Multi-threshold base kernel class (`GvrMultiThreshKernel`) the sandwich derives from |
| `indexer_topk_op_bench/ops/cute_vendored/blackwell/top_k/gvr_topk_decode.py` | Vendored GVR base kernel (phases P1-P4, cuteDSL) |
| `indexer_topk_op_bench/ops/cute_vendored/blackwell/top_k/gvr_topk_decode_cluster.py` | Vendored cluster primitives (DSMEM ld/st, mapa) |
| `indexer_topk_op_bench/ops/cute_vendored/blackwell/top_k/block_scan.py` | Warp/block scan helpers |
| `indexer_topk_op_bench/ops/cute_vendored/blackwell/utils.py` | PDL/griddepcontrol helpers |

## Entry point

```python
from gvr_msc_op import gvr_ms_auto
out = gvr_ms_auto(logits, pre_idx, seq_lens, index_topk=K, compress_ratio=cr)
```

Exact top-K (sorted index-set equivalent to `torch.topk`; intra-row output
order is runtime-nondeterministic by design — compare as sets). Requires
CUDA 12.x + cutlass CuTe DSL + torch on sm_100.

Code only: benchmark harnesses, nsys results, and reports intentionally stay
in the local workspace (see workspace `indexer_topk_op_bench/op21_gvr_prod/`).
