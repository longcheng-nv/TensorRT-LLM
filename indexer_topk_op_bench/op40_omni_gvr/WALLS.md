# WALLS.md — op40_omni_gvr

Structural walls (config-insensitive, mechanism understood). Inherited candidates
to re-verify on the PR-head kernel (they were measured on earlier variants):

- Launch/latency floor REVISED 2026-07-23 by floor probe: true floor is
  ~1.7 µs (GVR prologue+identity, nsys). The inherited "~10 µs" figure was
  algorithm latency, not a wall. Small-N cells carry ~6-8 µs of attackable
  phase-chain latency.
- Grid ≪ SM count at BS=1 ⇒ occupancy is structural; only more-CTAs-per-row
  forms move it (multi-CTA/cluster P4 is the allowed lever family).

## Campaign entries (append below)
- Instruction-fetch / icache wall (NCU full, pro_64k v7): 47.6% of stall
  cycles = no-instruction-fetched/icache-miss on the single-CTA mega-kernel
  (all phases + R0 + fallback + P4 coarse/fine/tail inlined). Unroll
  reduction falsified (iter9). Structural fix = kernel splitting /
  out-of-line cold paths — not expressible in single-kernel cuteDSL GVR.
- Occupancy structural at BS=1 (NCU): grid = 1-8 CTAs on 148 SMs; DRAM 0.1%,
  SM 0.09%, IPC 0.68 — latency chains, not bandwidth. Confirms inherited
  wall on e612.
- Phase-minima UB (measured minima, same kernel family): small-N ~6.1-6.4us
  vs v7 ~7.3us; 1M-cs8 ~14.6us vs ~17us => residual headroom ~1.15-1.20x on
  top of v7 => campaign UB ~1.30-1.35x vs base. The 1.60 stretch goal is
  ~20% beyond the all-levers-perfect bound of the GVR phase-serial skeleton
  under zero-regression + static-config constraints.
