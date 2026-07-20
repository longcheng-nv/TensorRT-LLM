# PR head (@e6fdbfac3d) GVR decode — ncu pipeline analysis (2026-07-20, umbriel-b200-081)

Driver `ncu_cell_prod2.py` (real §4 cells via launch() production contract,
cold-L2 evict between launches, `ncu --set full --launch-count 3`, launch #1
analyzed). Reps in /tmp/gvrlayers/ncu_prod2/ (node-local, not committed).
ncu durations are replay-inflated vs nsys (~1.3-1.8x) — use for STRUCTURE,
not absolutes.

## Measured launch contract (BS=1 fp32) — corrects a pick_config misprobe

| cell | N | actual launch |
|---|---|---|
| flash/32k  | 8195   | grid 1, **cs=1**, 512T |
| flash/128k | 32771  | grid 1, **cs=1**, 512T |
| pro/128k   | 32771  | grid 1, **cs=1**, 512T |
| pro/512k   | 131075 | grid 8, **cs=8**, 512T |
| v32/128k   | 131087 | grid 8, **cs=8**, 512T |
| flash/1024k| 262127 | grid 8, **cs=8**, 512T |

The cs=8 switch engages at N>=~128K; everything below runs ONE 512-thread
CTA. (A probe calling `pick_config(fp32, 1, N, N*cr)` returns cs=8 already at
N=32771 and T=1024 at N=131075 — the ncu-observed launch() contract differs;
launch shapes above are ground truth.) All cells: 80 reg/thread, occupancy
25% (Block Limit Registers=1 & SMEM=1 per SM), waves 0.05 — whole-GPU idle
by construction at BS=1; wall time == critical-path latency of 1 CTA/cluster.

## Stall-per-issued-instruction breakdown (top 3, % of stall cycles)

| cell | regime | ncu dur | #1 | #2 | #3 |
|---|---|---|---|---|---|
| flash/32k   | cs1 | 16.1us | no_instruction 35.4% | barrier 25.3% | long_scoreboard 13.0% |
| flash/128k  | cs1 | 22.0us | no_instruction 36.0% | barrier 19.7% | long_scoreboard 14.8% |
| pro/128k    | cs1 | 23.0us | no_instruction 34.4% | barrier 18.7% | long_scoreboard 14.8% |
| pro/512k    | cs8 | 26.8us | **barrier 61.1%** | long_scoreboard 10.7% | no_instruction 9.7% |
| v32/128k    | cs8 | 28.0us | **barrier 61.1%** | long_scoreboard 10.3% | no_instruction 9.6% |
| flash/1024k | cs8 | 26.1us | **barrier 55.2%** | long_scoreboard 14.0% | no_instruction 8.2% |

SM/DRAM throughput <=0.6%/0.4% everywhere; One-or-More-Eligible ~9%,
cycles-per-issued-inst ~43 (pro/512k) — pure latency-bound, never
bandwidth-bound.

## Pipeline verdict

Two distinct walls by regime:

1. **cs=1 (N<128K): instruction-fetch + serialization wall.** no_instruction
   ~35% = icache pressure of the 108KB unified kernel on a single CTA
   (matches op32's 31.4% finding); barrier ~20-25% and long_scoreboard ~14%
   secondary. Consistent with op32's conclusion: no lever inside the skeleton
   (register/thread/warp variants all wash-dead).

2. **cs=8 (N>=128K): BARRIER wall — 55-61% of stall cycles.** Static census
   (gvr_topk_decode.py): 83 `cute.arch.barrier()` sites, 7 cluster
   arrive/wait sites, 36 thread0-serial scalar sections (each fenced by 2
   block barriers). Structural sources on the critical path:
   - P1 preIdx gather runs FULL-ROW on all 8 CTAs (deliberate: keeps P1b
     rungs per-CTA identical) — 8x redundant scattered-load work.
   - `load_slice_to_smem` stage and the R0 M-ary count pass BOTH stream the
     slice from GMEM (count does NOT read the SMEM cache; cache serves only
     secant re-scans + P3) — 2x GMEM stream on the R0-hit path, no overlap
     between stage and count.
   - Cluster rendezvous chain per row: count-merge arrive/wait + thread0
     serial 8-peer x M DSMEM reduce -> handoff #1 (post-P2) -> handoff #2
     (post-P3 candidate gather) -> final barrier.
   - Phase 4 (+exact-tail/p4tt) runs on the LEADER CTA only: 7/8 CTAs (3584
     threads) sit at the final barrier for the entire P4 select.

## Lever candidates implied (to cross-check with clock64 phase breakdown)

- Fuse stage+count (count during load_slice_to_smem, or point count at the
  SMEM cache) — removes one full GMEM stream at cs=8 and cs=1.
- Slice-parallel or peer-parallel P4 (or collect-into-leader during P3
  handoff) — attacks the 7/8-idle tail; op34's oracle-UB showed multi-CTA
  collect-only ≈ sglang parity, i.e. the tail is where the cluster loses.
- Warp-parallel DSMEM merge (M columns x 8 peers is currently thread0
  serial).
- cs=1 wall: known-closed (op32); icache diet would need kernel splitting —
  out of scope for kernel-internal tweaks.

Next: merge with gvrpkgtimed clock64 per-phase breakdown (in flight) to put
us numbers on each phase before proposing the follow-up PR campaign.
