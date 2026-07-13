# op33 — iteration log

Incumbents: `op26_r0auto` (auto 1CTA/MC), `gvr_ms_auto` (op21 HLS/op27).
Node umbriel-b200-092, branch omni/op21-gvr-prod. op22-identical synth bundles.

## iter0 — 2026-07-13 — CRUX (NCU attribution, no kernel) — verdict: P5 axis mostly WALLED in-envelope; one live sliver (BS≥256 large-N)

Hypothesis (ledger check: op15 smem-resident DEAD, op32 short-row DEAD): do the
P5 execution levers (occupancy via launch_bounds/reg-reduction, 256-bit
vectorized loads) have NCU-visible headroom on the incumbents, per regime?

Probe (rung-0): NCU on 12 elapsed-metric cells + 6 `_active`-metric cells
(`scripts/crux_ncu.py` + `drive_crux.sh`, single profiled call, op22 bundles).
Metrics in `results/crux/` + `results/crux_summary.txt`.

Result — decisive structural map:
- **BS=1 (ALL N, ALL K) = LATENCY/BARRIER-bound, structural.** DRAM 0.06-0.5%
  (GPU-wide → memory system idle), occupancy 25% (1cta) / 50% (cluster) per-active
  (grid = 1..8 CTAs on 148 SMs → structurally grid-limited), per-active issue ~15%
  (op32) → ~85% stall on the serial secant/barrier chain. L2-trap + occupancy-
  structure vetoes BOTH fire. → **P5 vectorization VOID (DRAM idle), P5 occupancy
  VOID (grid≪SM).** Same wall as op15/op32, now generalized to all N at BS=1.
- **`waves_per_multiprocessor` < 1 up through BS=128** (BS16 grid=64→0.43 waves;
  BS128 grid=128→0.86 waves). ⇒ each SM holds ≤1 block; raising per-SM occupancy
  (reg 50%→75%) adds nothing when there is <1 block/SM. **P5-occupancy VOID up to
  BS~148.**
- **BS=128 K2048 N262144**: DRAM 37-41%, issue_act 47-49%, occ 50% (reg-limited,
  occlim_reg=1), sm% 38-40% — the *only* cells with balanced non-trivial load.
  Extrapolating to BS≥256 (waves≥1.7) is the ONE regime where occupancy lift
  (launch_bounds(1024,2+) + reg reduction) and 256-bit collect-pass vectorization
  could pay — exactly sglang_v2's `__launch_bounds__(1024,2)`+smem-spilling recipe,
  and op28's op26_mc-weak band (N≥262K BS31-512, 0.86× vs sglang). UNFALSIFIED.
- sec/req 14-30 (scattered gather) everywhere, but memory is never the bottleneck
  in-envelope ⇒ coalescing/vectorization can't pay except possibly BS≥256.

Diagnosis (one sentence): the GVR threshold skeleton is latency/barrier/grid-
limited across the BS=1 deployment core (idle DRAM, sub-wave grids), so execution-
throughput levers are structurally void there; the only unfalsified P5 sliver is
the high-BS (≥256) large-N cluster path where the grid finally fills the machine.

Ledger write-back: WALLS.md W1 (BS=1 exec-throughput void) + W2 (sub-wave grid
up to BS~148). Live target recorded for iter1.

Next: iter1 = occupancy lift on the cluster/mc path at BS≥256 large-N (reg
reduction + `__launch_bounds__` minBlocks 2→3 + 256-bit collect vectorization),
A/B vs op26_r0auto/gvr_ms_auto on that slice only; nsys ×3 verdict. Parallel
consideration: P3 warp-register tie-select at BS=1 is barrier-reduction (the RIGHT
class for a latency wall) but op32 caps a full barrier-fusion rewrite at ~6-12% —
low priority vs the high-BS throughput slice.
