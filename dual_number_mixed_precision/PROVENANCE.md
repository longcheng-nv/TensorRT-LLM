# dual_number_mixed_precision

Dual-number-based automated mixed-precision error-tracing & performance-optimization tool.

- `mixed_precision_tracer/` — the productized tool (`mptracer`): operator-agnostic harness core,
  pluggable twins/backends, 5 Skills, the `mixed-precision-specialist` agent, passing regression.
- `dsv4_moe_harness/` — the 8-iteration validation harness on the DSV4 MoE grouped GEMM (B200),
  with the Phase-0 design docs and the bilingual final report.

Generated artifacts (`results/`, `.matplotlib/`, `__pycache__/`) are excluded; scripts regenerate them.
Research provenance: chenglong92/Mixed-precision-Computing (branch ErrorAnalysis).
