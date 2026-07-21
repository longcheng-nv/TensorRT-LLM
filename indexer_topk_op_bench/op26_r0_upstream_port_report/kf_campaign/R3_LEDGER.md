# R3 campaign ledger — gvr-topk-r3 (beyond-champion)

Campaign id: `e5q1zgrfhs0z57dj6850kc444r` (KF managed B200, effort high,
6 agents/round: 2×fable-5(high) + 2×gpt-5.6-sol(high) + 2×n3-opus-4.8;
max_rounds 20, max_duration 8h, max_cost $800, stagnation 4).
Node: umbriel-b200-027 (8×B200). Started 2026-07-21 13:40Z.

## Decisions

- **D1 (baseline packaging workaround).** `kf campaign prepare` with
  `--baseline-solution` failed: the platform baseline evaluator does not stage
  campaign assets (0/28 workloads — safetensors missing). Workaround: champion
  per-workload platform timings extracted from campaign-1 trace of kernel
  `c74fb3c0` (28/28 PASSED, geomean 10.76 µs) → `gvr-topk-r3/baselines.jsonl`;
  champion full source inlined into prompt.md v2 instead. c74fb3c0 timings ==
  c74f_sbx timings on this subset (no cell in the sbx graft rung 8448<n≤16896).
  Residual risk: the sbx graft raised `topk_small` `__launch_bounds__` 768→1024,
  which could shift register allocation on small-n rungs; local ab_sbx showed
  no give-back, accepted.
- **D2 (PR head moved).** PR#16457 head advanced e6fdbfac3d → `b14ec40e1b`
  (182 commits; GVR deltas: P4 bracket-window histogram + multi-level
  refinement, lane-parallel bin-search, redundant-warp sync reduction, parity-
  buffered DSMEM count exchange, float-domain bin clamp; only
  `gvr_topk_decode.py` changed among gvrpkg files, 210833→247384 B).
  `gvrpkg_head/` refreshed in place; old arm kept at `gvrpkg_e6fd/`.
- **D3 (foreign GPU load).** Intermittent short-lived foreign job bursts
  observed (67-77 GB, up to 83% util), hopping GPUs (0-3 then 7). Full grids
  run on 7 shards (GPUs 0-6) when GPU7 busy; contamination watched via
  per-rung pr_cold anchors + per-batch p95.

## Timeline

| ts (UTC) | event |
|---|---|
| 13:33 | prepare attempt #1 with baseline_solution → platform asset-staging gap (D1) |
| 13:40 | campaign started (baselines.jsonl path), monitor armed |
| 13:47 | gvrpkg_head refreshed to b14ec40e1b, import OK |
| 13:52 | 28-cell probe champh2 (GPU6): champion vs NEW head cold gm **1.7193**, 0/28 reg, 28/28 exact |
| 13:56 | full 865-cell grid champh2 launched, 7 shards GPUs 0-6 |

## Verdicts

| tag | arms | cells | cold gm | regs | exact | notes |
|---|---|---|---|---|---|---|
| champh2_probe | c74f_sbx vs PR@b14ec40e1b | 28 | 1.7193 | 0 | 28/28 | GPU6 probe |
