# op35 iteration log
(entries appended per omni-kernel protocol; verdicts ∈ SHIP/FALSIFIED/WASH/PIVOT)

## iter0 — 2026-07-16 — PROBE (rung 1+2)
Hypothesis: B1 block-skip viable if (a) tile coverage sparse on real data, (b) P3 scan
is a material kernel share. Ledger check: none (new granularity; ≠op16 value-peel, ≠ms fusion).
Probe A (host replay, 77 cells, results/replay_b1.csv):
  - (warp,window)=256-elem quantum is the right skip granularity:
    P3-scan-work saved ceiling: synth N≥64K med 79%, real N≥64K med 56%, small-N 16-19%.
  - whole-window (8K) granularity (proposal's literal B1): near-zero at real cells → dead.
  - warp early-exit (zero-sideband): 51%/24% — weaker than sideband but ~free.
Probe B (p3_oracle_frac=0.001 ablation, 39 N≥65536 cells, 8-GPU shard, CUDA-event):
  - event axis carries ~40µs host overhead → use DIFFERENCE (base-var) vs nsys pr time.
  - early shards: P3 scan share ≈ 0-25% at cluster cells (slice/CTA only 16-32K elems
    → P2/P3 scans are ~2-4 iters; kernel dominated by fixed phase-chain + gather + handoff).
Design settled (if GO): fp32 wwmax[warp][win] sideband in P2 (1 FMAX/elem) → P3 per-warp
ballot bitmap → bit-loop; EXACT for any thr (fallback recounts B2 ride free); gate cs≥4.
