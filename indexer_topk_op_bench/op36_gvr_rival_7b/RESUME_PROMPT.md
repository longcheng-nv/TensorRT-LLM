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

## State after iter4-in-flight (2026-07-18) — Track B built, screening running
- NODE: umbriel-b200-093 now (047 gone). Same-node composites only.
- Track B port DONE + battery 93/93 (src/trackb/, battery_bx.py, arm
  sgl_bx in ops_op36.py). Screening 25 batches 8-way IN FLIGHT →
  results/b_screen (resume: relaunch drive_op36_shard.sh W W 8 with
  OPS="gvr_pr,sglang_v2,sgl_bx" OUT=results/b_screen; .done markers skip).
- Analysis: analysis/trackb_verdict.py results/b_screen (anchor drift +
  guard ε + dispatch threshold + hole close-up). Anchor checkpoint GREEN.
- NEXT after screening: verdict re-run ≤2-way on decisive batches (incl.
  ε outlier pro/16k/BS1 + threshold-boundary batches), fix per-model
  dispatch table, update ITERATIONS, then A2 distP4 → feasibility gate.

## State after iter3 (2026-07-18) — A0 CLOSED
- A0 ship table: skip_h1 ON {K512@N>=262144, K2048(+kb512)}, OFF K1024;
  composite vs sglang 0.726->0.738 same-node, zero regression; verdict
  re-run confirmed screening exactly (ITERATIONS iter3).
- GOTCHA fixed iter2: GVR arms must use launch/pick_config contract
  (refresh_harness pattern); iter1 frozen-shape data INVALID (archived).
- NEXT (priority): Track B small-N 8-CTA exact path (the 99-cell 0.60
  hole; port+fix-tie-overflow beats from-scratch per apex); then A2
  distP4; A1 escape residual AFTER A0 gating (overlapping wins); A3 C>8.
- Feasibility gate pending after B+A2 first verdicts (PLAN.md).

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
