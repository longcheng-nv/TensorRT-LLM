# op17 — GVR Threshold-Portfolio Top-K Operator (source only)

Operator source code for the DSv4 indexer Top-K threshold-portfolio kernels
(GVR, cuteDSL, Blackwell sm_100). This branch carries **code only** — no
reports, benchmark data, or measurement scripts.

## Layout

| path | what |
|---|---|
| `indexer_topk_op_bench/op17_gvr_portfolio/v2/gvr_portfolio_cluster_v2_op.py` | **portfolio operator (current)** — multi-round G-way sweep + warp-parallel peer read at t=512 shapes; v1 body at t=1024 shapes (register-budget gate); auto-G dispatch |
| `indexer_topk_op_bench/op17_gvr_portfolio/v2/gvr_portfolio_fusion_op.py` | **P×T fusion operator (current)** — P row-slices × T threshold-slots cluster; large-N regime |
| `indexer_topk_op_bench/op17_gvr_portfolio/src/gvr_portfolio_cluster_op.py` | v1 portfolio operator (superseded by v2) |
| `indexer_topk_op_bench/op17_gvr_portfolio/src/gvr_portfolio{,_mcta}_op.py` | falsified iter4/iter5 variants (kept for the record) |
| `indexer_topk_op_bench/harness/gvr_cutedsl_op.py` | single-CTA baseline wrapper (imported as the G<2 dispatch fallback) |
| `indexer_topk_op_bench/ops/cute_vendored/` | vendored cuteDSL kernels the operators build on |

## API

```python
from gvr_portfolio_cluster_v2_op import gvr_portfolio_cluster_v2
out = gvr_portfolio_cluster_v2(logits, pre_idx, seq_lens, K, compress_ratio, G="auto")

from gvr_portfolio_fusion_op import gvr_portfolio_fusion
out = gvr_portfolio_fusion(logits, pre_idx, seq_lens, K, compress_ratio, P=4, T=4)  # T>=4 required
```

Requires torch + cutlass cuteDSL on sm_100. The `__main__` self-tests expect
the local bench environment (synthetic-data generators) and are not runnable
from this branch alone.
