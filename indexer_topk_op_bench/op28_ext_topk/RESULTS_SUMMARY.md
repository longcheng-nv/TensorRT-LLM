# op28 — LATEST SGLang v2 & FlashInfer top_k vs in-tree ops

cells loaded: 906  (node umbriel-b200-027, fp32, op22rr byte-identical bundles, nsys cold-L2 20 reps / warm 50)

anchor drift gvr_cutedsl orig/op28: med 0.9984  p10 0.9795  p90 1.0083  n=798

## SGLang v2 vs OLD StreamingTopK  (cold-L2 kernel-sum, t(sglang_streaming)/t(sglang_v2), >1 => sglang_v2 faster; geomean over all cells)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.993 | 105 |
| best | 1024 | 1.984 | 105 |
| best | 2048 | — | 0 |
| worst | 512 | 1.962 | 105 |
| worst | 1024 | 1.958 | 105 |
| worst | 2048 | — | 0 |
| real | 512 | 1.984 | 105 |
| real | 1024 | 1.970 | 105 |
| real | 2048 | — | 0 |

## SGLang v2 vs GVR(cuteDSL) baseline  (cold-L2 kernel-sum, t(gvr_cutedsl)/t(sglang_v2), >1 => sglang_v2 faster; geomean over all cells)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.879 | 105 |
| best | 1024 | 2.113 | 105 |
| best | 2048 | 1.876 | 92 |
| worst | 512 | 1.396 | 105 |
| worst | 1024 | 1.547 | 105 |
| worst | 2048 | 1.507 | 92 |
| real | 512 | 2.030 | 105 |
| real | 1024 | 2.021 | 105 |
| real | 2048 | 1.932 | 92 |

## SGLang v2 vs Radix(cuteDSL)  (cold-L2 kernel-sum, t(radix_cutedsl)/t(sglang_v2), >1 => sglang_v2 faster; geomean over all cells)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.808 | 105 |
| best | 1024 | 1.753 | 105 |
| best | 2048 | 1.682 | 92 |
| worst | 512 | 1.764 | 105 |
| worst | 1024 | 1.755 | 105 |
| worst | 2048 | 1.678 | 92 |
| real | 512 | 1.794 | 105 |
| real | 1024 | 1.742 | 105 |
| real | 2048 | 1.705 | 92 |

## FlashInfer top_k vs GVR(cuteDSL)  (cold-L2 kernel-sum, t(gvr_cutedsl)/t(flashinfer_topk), >1 => flashinfer_topk faster; geomean over all cells)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.369 | 105 |
| best | 1024 | 1.509 | 105 |
| best | 2048 | 1.350 | 92 |
| worst | 512 | 1.043 | 105 |
| worst | 1024 | 1.108 | 105 |
| worst | 2048 | 1.093 | 92 |
| real | 512 | 1.485 | 105 |
| real | 1024 | 1.463 | 105 |
| real | 2048 | 1.393 | 92 |

## FlashInfer top_k vs Radix(cuteDSL)  (cold-L2 kernel-sum, t(radix_cutedsl)/t(flashinfer_topk), >1 => flashinfer_topk faster; geomean over all cells)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.317 | 105 |
| best | 1024 | 1.252 | 105 |
| best | 2048 | 1.211 | 92 |
| worst | 512 | 1.318 | 105 |
| worst | 1024 | 1.257 | 105 |
| worst | 2048 | 1.217 | 92 |
| real | 512 | 1.312 | 105 |
| real | 1024 | 1.261 | 105 |
| real | 2048 | 1.229 | 92 |

## FlashInfer top_k vs SGLang v2  (cold-L2 kernel-sum, t(sglang_v2)/t(flashinfer_topk), >1 => flashinfer_topk faster; geomean over all cells)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 0.729 | 105 |
| best | 1024 | 0.714 | 105 |
| best | 2048 | 0.720 | 92 |
| worst | 512 | 0.747 | 105 |
| worst | 1024 | 0.716 | 105 |
| worst | 2048 | 0.725 | 92 |
| real | 512 | 0.732 | 105 |
| real | 1024 | 0.724 | 105 |
| real | 2048 | 0.721 | 92 |

## FlashInfer i32-minimal vs public API  (cold-L2 kernel-sum, t(flashinfer_topk)/t(flashinfer_topk_i32), >1 => flashinfer_topk_i32 faster; geomean over all cells)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.023 | 105 |
| best | 1024 | 1.034 | 105 |
| best | 2048 | 1.048 | 92 |
| worst | 512 | 1.022 | 105 |
| worst | 1024 | 1.032 | 105 |
| worst | 2048 | 1.047 | 92 |
| real | 512 | 1.023 | 105 |
| real | 1024 | 1.034 | 105 |
| real | 2048 | 1.050 | 92 |

## sglang_v2 vs op21_hls (op22rr arm, anchor-transferred onto node-027 scale; >1 => sglang_v2 faster)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.375 | 93 |
| best | 1024 | 1.390 | 93 |
| best | 2048 | 1.389 | 80 |
| worst | 512 | 1.594 | 93 |
| worst | 1024 | 1.766 | 93 |
| worst | 2048 | 1.625 | 80 |
| real | 512 | 1.488 | 93 |
| real | 1024 | 1.610 | 93 |
| real | 2048 | 1.440 | 80 |

## sglang_v2 vs op26_r0auto (op22rr arm, anchor-transferred onto node-027 scale; >1 => sglang_v2 faster)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.407 | 93 |
| best | 1024 | 1.642 | 93 |
| best | 2048 | 1.494 | 80 |
| worst | 512 | 1.336 | 93 |
| worst | 1024 | 1.524 | 93 |
| worst | 2048 | 1.457 | 80 |
| real | 512 | 1.386 | 93 |
| real | 1024 | 1.516 | 93 |
| real | 2048 | 1.592 | 80 |

## sglang_v2 vs op25_hls (op22rr arm, anchor-transferred onto node-027 scale; >1 => sglang_v2 faster)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.485 | 93 |
| best | 1024 | 1.445 | 93 |
| best | 2048 | 1.394 | 80 |
| worst | 512 | 1.451 | 93 |
| worst | 1024 | 1.404 | 93 |
| worst | 2048 | 1.611 | 80 |
| real | 512 | 1.395 | 93 |
| real | 1024 | 1.392 | 93 |
| real | 2048 | 1.445 | 80 |

## sglang_v2 vs gvr_multicta_cutedsl (op22rr arm, anchor-transferred onto node-027 scale; >1 => sglang_v2 faster)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.629 | 93 |
| best | 1024 | 1.861 | 93 |
| best | 2048 | 1.568 | 80 |
| worst | 512 | 1.260 | 93 |
| worst | 1024 | 1.424 | 93 |
| worst | 2048 | 1.354 | 80 |
| real | 512 | 1.767 | 93 |
| real | 1024 | 1.756 | 93 |
| real | 2048 | 1.640 | 80 |

## flashinfer_topk vs op21_hls (op22rr arm, anchor-transferred onto node-027 scale; >1 => flashinfer_topk faster)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 0.959 | 93 |
| best | 1024 | 0.947 | 93 |
| best | 2048 | 0.980 | 80 |
| worst | 512 | 1.139 | 93 |
| worst | 1024 | 1.211 | 93 |
| worst | 2048 | 1.157 | 80 |
| real | 512 | 1.043 | 93 |
| real | 1024 | 1.120 | 93 |
| real | 2048 | 0.992 | 80 |

## flashinfer_topk vs op26_r0auto (op22rr arm, anchor-transferred onto node-027 scale; >1 => flashinfer_topk faster)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 0.981 | 93 |
| best | 1024 | 1.119 | 93 |
| best | 2048 | 1.054 | 80 |
| worst | 512 | 0.954 | 93 |
| worst | 1024 | 1.045 | 93 |
| worst | 2048 | 1.037 | 80 |
| real | 512 | 0.971 | 93 |
| real | 1024 | 1.054 | 93 |
| real | 2048 | 1.096 | 80 |

## flashinfer_topk vs op25_hls (op22rr arm, anchor-transferred onto node-027 scale; >1 => flashinfer_topk faster)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.035 | 93 |
| best | 1024 | 0.984 | 93 |
| best | 2048 | 0.983 | 80 |
| worst | 512 | 1.037 | 93 |
| worst | 1024 | 0.962 | 93 |
| worst | 2048 | 1.146 | 80 |
| real | 512 | 0.978 | 93 |
| real | 1024 | 0.968 | 93 |
| real | 2048 | 0.995 | 80 |

## flashinfer_topk vs gvr_multicta_cutedsl (op22rr arm, anchor-transferred onto node-027 scale; >1 => flashinfer_topk faster)

| scenario | K | geomean | n |
|---|---|---|---|
| best | 512 | 1.135 | 93 |
| best | 1024 | 1.268 | 93 |
| best | 2048 | 1.106 | 80 |
| worst | 512 | 0.900 | 93 |
| worst | 1024 | 0.976 | 93 |
| worst | 2048 | 0.963 | 80 |
| real | 512 | 1.238 | 93 |
| real | 1024 | 1.221 | 93 |
| real | 2048 | 1.129 | 80 |

## Caveats
- canonical `us` = per-range kernel-time SUM (comparable to all
  prior report numbers). sglang_v2's persistent-cluster path
  (N>=131072, 30<BS<=512) launches 2 PDL kernels: sum can
  double-count overlap (observed up to 1.8x at N=262144 BS=64 where
  span=0.56x sum) or miss the inter-kernel gap (span up to 1.2x sum
  at N=131072); `*_span_us` columns carry the honest wall-clock.
- sglang_v2 `topk_plan` runs untimed (production: once per step,
  reused across ~61 layers; measured ~7us wall => ~0.11us/layer).
- flashinfer public top_k returns (values fp32, indices int64) --
  slightly larger output traffic than the int32-only in-tree
  contract; flashinfer_topk_i32 is the contract-matched variant
  (~2-5% faster).