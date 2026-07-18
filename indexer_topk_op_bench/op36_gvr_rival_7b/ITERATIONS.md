# op36 iteration log (canonical numbers live here)

## iter0 (2026-07-18) — setup
Baseline arithmetic from op26 §8 backfilled grid (analysis/baseline_7b.py):
gvr_pr vs sglang_v2 real fp32 275 cells gm 0.745 (4-16k 0.599 / 32-128k
0.793 / 256k-1M 0.910). Bounds: levers ×1.25 → 0.931; best-dispatch 0.827;
+parity 1.030; +levers 1.069. 1.10 target gated (PLAN.md).

## iter1 (2026-07-18) — INVALID build, discarded
gvr arms at frozen shapes (ops_rival T1024/mbpm1): anchor drift vs report
med 1.143 / p95 1.929 while sglang med 1.000 → build mismatch, not node.
Report §8 GVR rows use the production launch contract (launch/pick_config,
refresh_harness). Data archived results/a0_screen_iter1_INVALID_frozen_shapes.
LESSON: validate anchors at ~1/3 of any screening sweep, saved ~2h here.

## iter2 (2026-07-18) — A0 screening (launch contract), 25 batches, 8-way
Anchors: gvr_pr med 1.022 / p95 1.073 vs report; sglang med 1.000 / p95
1.036. gvr_a0 exact 275/275 (all cells all BS).
- a0 vs pr overall gm 0.996 — WASH on the campaign axis (vs sglang
  0.726 → 0.727; unchanged).
- Wins: flash 1024k gm 1.206 (BS≥128 1.30-1.41 — the disclosed low-hit
  cold region, hit≈0.42); v32 256k 1.095 (kb512); v32 small-N 1.02-1.05.
- LOSSES: pro 512k gm 0.813 (0.72-0.86 ALL BS, one-signed) — CONTRADICTS
  op35 BS=1 "0 lost / worst 0.975". flash 256k 0.952, pro 64k 0.946.
- Verdict re-run (≤2 concurrent, 6 pole batches: pro512k flash1024k
  flash256k v32-256k pro64k pro8k) IN FLIGHT → decides per-(K,N)
  flag table (skip_h1 shape-gated; kb512 K2048).

## Notes
- 8-concurrent screening is a SCREEN; every ship claim re-measured ≤2-way.
- Track A1 scoping: R0-miss path in gvrpkg35 (@~L3600) is already seeded
  log-falsi (fb_fix); "escape" = cold-seed bail to base-secant-equivalent
  (base beats pr 1.3-1.5x in the disclosed §7b low-hit BS≥128 region).
  skip_h1's flash-1024k win overlaps this — measure residual after A0
  verdict before building.

## iter3 (2026-07-18) — A0 VERDICT + shape-gated close
Verdict re-run (≤2-way, 6 pole batches) CONFIRMS screening bit-for-bit
(a0/pr gm identical to 3 decimals; drift ~1.0): pro-512k 0.813 is REAL.
Mechanism: skip_h1 loses where hit is warm at large N (K1024/N=131072,
0.72@BS1..0.91@BS1024, one-signed), wins where hit is cold (flash-1024k
hit≈0.42, 1.21). op35 "0 lost" verdict was BS=1-only + different launch
shapes — recorded as a LEARNING, not a contradiction.
SHIP TABLE (shape-keyed, no hit dispatch): skip_h1 ON for K512@N>=262144
and K2048 (+kNumBins 512); OFF for K1024. Composite vs sglang same-node:
0.726 -> 0.738 (+1.7%), gated arm vs pr: flash 1.021/pro 1.000/v32 1.031
(v32 min 0.956 single rung — watch at ship gate).
A0 CLOSED. Campaign axis confirms baseline arithmetic: skeleton levers
move single points, not the sglang gap. Weight shifts to Track B (4-16k,
99 cells, gm 0.60) + A2 distP4.
