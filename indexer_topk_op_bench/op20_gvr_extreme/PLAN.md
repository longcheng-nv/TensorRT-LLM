# op20 — GVR extreme (omni-kernel campaign)

**Goal**: single GVR operator (op19 sandwich lineage, secant-threshold → refine
structure preserved, exact only) that is (a) faster on average than every rival
and (b) the fastest op on ≥95% of report-grid cells.

**Priority (user, 2026-07-03)**: tier1 fp32 K512/1024 > tier2 fp32 K2048 > tier3 16-bit.

**Rival to beat**: `radix_cutedsl` (auto) — measured IN-RUN by `scripts/tier_bench.py`
(3-way cold-L2 CUDA-graph A/B; no cross-run scale transfer needed).

## Resolved plan (omni-kernel Phase 0)
- Operator: DSv4 indexer top-K decode, GVR sandwich (start = op19 `gvr_sw_auto` copy).
- Shapes: report grid K{512,1024,2048} × N{4K..262K} × BS{1..2048}; iteration set
  BS{1,4,16,64,256,1024}, gate = full 12-BS grid.
- Reference: torch.topk value-equivalence (vdiff=0) + uniq=K per row (`exact_all`).
- GPU: B200 sm_100 (umbriel-b200-044, 2 GPUs); B300 cross-check at final gate.
- Language: CuTe DSL (existing kernel); CUDA C++ fallback only if DSL blocks a lever.
- Branch: `omni/op20-gvr-extreme`.

## Targets from report loss-map (fp32, vs best rival, gm needed on losing cells)
| bucket | needed | n_lose |
|---|---|---|
| N≤32K BS≤16 | 1.29× | 51 |
| N≤32K BS≤256 | 1.22× | 34 |
| N≤32K BS≥512 | 1.09× | 14 |
| N=131K BS≤16 | 1.19× | 22 |
| N=262K BS≤16 | 1.69× | 18 |

## Phase budget (clock64, K512 fp32, BS=1) → attack order
- Small N (≤16K): total ~16.4µs = P1 2.7 + P2 3.0 + P3 1.4 + **P4 9.2 (56%)**. Radix ≈5.5–7.
- Large N (262K): total ~41 = P1 2.1 + **P2 16.2 + P3 16.6 (80%)** + P4 6.1.
- P2 secant iters: K512 = 1.0 (optimal); K1024 1.3–2.1 @N≥32K; K2048 2.0–2.65 (never 1).
- P3 over-collect: K512 = 3.0–4.6×K candidates (kFTarget too loose).

## Direction queue (from 2026-07-03 deep analysis; falsification-safe)
- **D2 P4 rebuild** (small-N decisive): tighter thr1 → cand ~1.5×K; replace serial
  3-step bin-search + ≤15 snap iters with parallel 2-level digit histogram + warp scan.
  Target P4 9.2 → ~2.5µs.
- **D3 P1 fold** into P2 first streaming pass (−2.2–2.7µs at small N).
- **D1 P2 interpolation**: log-CCDF-domain secant/IQI (K1024/K2048 large-N evals → 1);
  ladder multi-point interpolation instead of straddle-pair (tighter band feeds D2).
- **D4 P3 collect**: shrink over-collect at source (thr1 from D1), lighter compaction.
- **D5 large-N**: apply D1's one-pass ladder via cluster path for N≥131K BS≤16 (1.69× hole).

## Red lines (falsified, do NOT retry)
smem-resident rows (op15) · HBM-pass compaction (op14) · warp-collectives in the
streaming loop (op14) · band-shrink inside secant / two-threshold peel taxes (op16) ·
grid.sync multi-CTA · union-buffer smem cut that drops residency (op19 iter14).

## Gates (every iteration)
exact_all on every cell (×3 seeds at milestones) · cold-L2 canonical · nsys
pure-kernel spot-check for any claimed win ≥1.1× · commit per iter (`[op20 iter N]`).
