# op18 — single-CTA multi-threshold GVR top-K (B200)

This branch contains ONLY the op18 operator deliverable:
`indexer_topk_op_bench/op18_gvr_1cta_multithresh/` — kernel (`src/gvr_mt_op.py`, entry `gvr_mt_auto`),
tuning/validation scripts, results (text; binary nsys artifacts excluded), and the bilingual REPORT.html.

It depends on the parent repo's `indexer_topk_op_bench/ops/cute_vendored` + `harness/` (not included here);
see REPORT.html §8 for reproduction.
