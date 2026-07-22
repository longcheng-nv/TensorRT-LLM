# compB BS>1 extension — kernel + harness (code-only branch)

Batched extension of the KF R3 ship kernel (compB, single-row BS=1 exact
top-K for the DSv4 indexer) to BS>1, developed against PR#16457 head as the
rival. Measurement data, nsys artifacts and reports stay local
(TensorRT-LLM working tree: indexer_topk_op_bench/.../kf_bs_scaling/ext/,
ledger: kf_campaign/R3_LEDGER.md).

Arms in kernel_ext.cu (all exact, tie-robust):
- grid.y batched single-CTA tiers (n <= 16896, any BS)
- ext_v4: chunked single-wave row teams, register diet launch_bounds(512,4)
- tp:  D1 barrier-free 3-kernel throughput arm (hist / collect / finish)
- tp2: D2 sampled-estimate single-pass arm (budget-driven b_safe,
       candidate superset invariant, count-check fallback)
- pq:  B' persistent queue (kept for the record; falsified vs chunked)

Validated state at branch creation (nsys cold-L2, umbriel-b200-039,
real-capture rows, N=131075): best-arm gm over BS 8-1024 = 1.597x vs
PR#16457 head, 48/48 exact. Optimization campaign continues on this branch.
