# op36 RESUME PROMPT — paste into a fresh Claude Code session

> Keep CURRENT at every iteration commit. Disaster-recovery handoff.

---- PASTE BELOW THIS LINE ----

Continue the op36 GVR-vs-sglang real-§7b campaign in
`indexer_topk_op_bench/op36_gvr_rival_7b/` (TensorRT-LLM checkout). Read in
order: PLAN.md (goal, baseline arithmetic 0.745, tracks A0-A3/B, feasibility
pivot gate, red lines, measurement discipline), then this file's State.

Key facts:
- Base = PR#16457 shipped HEAD eae374554c, worktree ../TensorRT-LLM-gvr-r0.
- Judgment axis = op26 report §8 real fp32 BS×ISL grid (275 sglang-comparable
  cells, us_span, nsys cold-L2). Baseline gvr_pr/sglang gm 0.745; hole =
  ISL 4-16k (gm 0.599, 99 cells); target 1.10 gated by feasibility pivot.
- Harness = clone of op26_r0_upstream_port_report/rival_harness/ (cell-
  resumable). 8-GPU shards for screening ONLY; ship verdicts ≤2 concurrent
  nsys. Node umbriel-b200-047 (8×B200, all idle at campaign start).
- bundle-v2 kernel diffs ready in op35_gvr_round2/variant/gvrpkg35/
  (default-off flags skip_h1 + K2048 kNumBins 512).

## State after setup (2026-07-18)

- Campaign skeleton committed: PLAN.md, analysis/baseline_7b.py (extracts
  RIVAL grid from op26 REPORT → results/baseline_real_bs.csv 825 rows +
  bound table A=0.745/B=0.931/C=0.827/D=1.030/E=1.069).
- Tasks #1-#7 in session task list mirror tracks A0-A3, B, GATE.
- NEXT: Track A0 — register gvr_a0 arm (gvrpkg35 flags) in a cloned
  rival_harness, smoke 1 cell/model on GPU0, then screening sweep of the
  real fp32 BS×ISL grid sharded across 8 GPUs; verdict cells re-run ≤2-way.

## Iteration log

- iter0 (2026-07-18): setup + baseline arithmetic. No silicon yet.
