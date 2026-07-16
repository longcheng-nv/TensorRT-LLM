# op35 gap analysis — PROPOSAL_GVR_NEXT_OPT.html vs PR-HEAD code reality

Date 2026-07-16. Proposal anchor @a4fb75dff6; PR worktree HEAD eae374554c
(3 commits past the 018251950f snapshot the proposal was written against:
vseed + per-K qfracs + P4 exact-tail).

## Proposal items — status against actual code

| Item | Proposal claim | Code reality @eae374554c | Verdict |
|------|----------------|--------------------------|---------|
| H1 log-falsi fallback | "PR fallback is plain secant" → GO | **STALE**: `fb_fix=True` default, α=0.2 `log2_mstar` (L474), R0-miss does seeded bounded log-falsi (L3426+); vseed donates a measured interior bracket | ALREADY IN — not a lever |
| H2 dist-msc fallback (N≥65536) | GO with N-gate | R0 count pass IS cluster-merged at cs>1 (`block_count_ge_multi` + `s_cluster_partial_m`, L3348-3417; header comment L3341 is stale). Fallback refine at cs>1: needs verification — if falsi recount is also cluster-sliced, H2 is largely in | VERIFY in iter0, then residual only |
| H3 K2048 tail ladder (0.75,0.45,0.048) | GO | Per-K qfracs exists but K2048 default = (0.85,0.35)+vseed; **no deep-tail 0.048 column** | LIVE — cheap ctor-arg experiment |
| H4 native 16-bit compares | GO | P2/P3 upconvert to fp32 (L2893 comment) | LIVE but **fp32-only §6 metric ⇒ zero target impact**; follow-up-PR harvest only |
| B1 fused block-max sideband → sparse P3 | proposed | P3 (`phase3_collect_candidates`) is a full-N re-stream; NO per-tile sideband exists | LIVE — main structural lever; iter0 host replay for ceiling |
| B2 fallback rescan skip | proposed | fallback recount re-streams full N | LIVE, rides on B1 |
| B3 cross-step speculative skip | proposed (2nd wave) | needs side tensor + API change | DEFERRED (2nd wave; API surface) |
| C op29-HBE port | proposed P1 | — | **EXCLUDED by USER** (too sglang-like) |
| D warp pipelining + launch | ≤10% opportunistic | cs4-nt512 (+3.8% cs4 unconfirmed) from OPT_CAMPAIGN iter1 | LIVE minor |

## Points the proposal MISSED (added by this campaign)

1. **P2-sideband per-tile COUNTS (not just rung-class max)** — B1 upgrade: if P2
   records per-tile per-rung counts (or even count-nonzero bitmap at the admitted
   rung), P3 can (a) skip empty tiles AND (b) know exact per-tile write offsets
   (prefix over tile counts), turning the collect into coalesced sparse writes
   with no re-scan of skipped tiles. SMEM cost: nTiles u8/u16. Falsi-ledger
   check: NOT the ms slot-collect (which collected *values* during P2 — the
   1.47× loser); this only collects *statistics* (2 inst/tile-class), P3 still
   does the value pass but sparsely. Closest relative op29 (excluded) works on
   sampled columns; this is exact per-tile truth.
2. **§6 metric set is fp32-only** — the proposal's H4 (16-bit) and region-③
   framing do not move the USER's +40% metric at all. Reprioritized down.
3. **cs=8 vs cs=4 boundary + nt at cluster** — pick_config picks cs8 at
   BS≤4 & N≥131072 from synth tuning; per-cell regret on the §6 real cells
   never audited at PR HEAD (vseed changed P3/P4 cost balance).
4. **kC (kCC) tightening for K1024/K2048** — P3 over-collect 3.96×K@K512 was
   dieted only for K512-cs1 (kC=3072). K1024 (kC=6144?) / K2048 untested;
   16-bit tie contract kC≥5K only binds 16-bit, and §6 is fp32 ⇒ more room.
5. **Worst-axis fallback iteration budget** — synth-worst large-N cells pay the
   falsi refine loop; vseed already donates a bracket. Residual: bound refine
   iters + B2 skip on recounts. (Distinct from H1 which is already in.)
6. **Honest UB/LB bound for the +40% ask** — neither report nor proposal bounds
   the §6-metric ceiling of any GVR-skeleton change; campaign adds oracle
   experiments (e.g. P3-free / P2-free ablations) to double-lock feasibility.

## Prioritized experiment order

iter0: (a) cluster-path R0/fallback verification; (b) clock64 phase split on
  §6 representative cells (P2/P3/P4/fallback shares); (c) B1 host replay on
  real captures → tile-coverage ceiling. → GO/NO-GO for B1.
iter1: L1 H3-tail K2048 qfracs (config-only, immediate A/B).
iter2: B1 kernel (P2 sideband → sparse P3) behind flag, N≥65536 gate.
iter3: launch tweaks (cs4-nt512, cs8 boundary) nsys confirm.
iter4: kC tightening K1024/K2048 (fp32).
iter5+: B2, warp-pipeline probe, UB/LB bounding, stall protocol.
