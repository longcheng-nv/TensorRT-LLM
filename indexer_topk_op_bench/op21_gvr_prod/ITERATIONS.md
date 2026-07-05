# op21 iterations

Protocol: nsys pure-kernel cold-L2 canonical from day 1; exact_all synth ×3
seeds + real-capture exactness gate; commit per iter `[op21 iter N]`.
Priority: fp32·K1024 > fp32·K512/2048 > 16-bit; smallBS·largeN >
largeBS·smallN > rest. Rival = per-cell best of report.html ops (nsys CSVs).

## Iter 0.5 — 2026-07-05 — HOST PROTOTYPE: P1 order-stat M-threshold seeding
**Question**: does one fused M-threshold round, seeded purely from preIdx
order statistics (no offline tables), straddle K with a small band on REAL
data (Pro h≈0.7, Flash h≈0.4, v32) and synth (h=0.6)?
**Method**: scripts/proto_p1_orderstat.py — c(f) = count_ge(row, g[f·K]) curves
per bundle; evaluate M=2/M=4 placements incl. the guaranteed lower anchor
g_min (c(g_min) ≥ K always); metrics = straddle rate, band size (p50/p90/max),
miss-mode rate (all counts ≥ K ⇒ need round 2 above).
**Result (55 rows: pro 30L + flash 7L + v32 9L + synth 9)**: **GO.**
- M=4 order-stat placement (0.25,0.5,0.75,1.0)·K: straddle **94.5% overall,
  100% on Pro (P0 priority) and synth**; misses are ALL the benign `all_ge`
  mode (flash/v32, h≈0.35 ⇒ true Kth above g[0.25K]) → one round-2 above
  thr(0.25) fixes; **never** `all_lt` (g_min anchor guarantees c ≥ K).
- Band (straddling pair, /K): pro p50 1.20 max 3.48; synth p50 0.88 max 1.03;
  M=6 tightens to p50 ~0.5. Not ≪K with static placement, but bounded ≤3.5K.
- Speculative slot-collect at the f=0.75 threshold: pro cnt/K p50 1.03 max
  4.68 → fits kC=5120 (5×K) on 30/30 Pro layers; flash needs a lower collect
  column (cnt(0.75)/K p50 8.8 at K512) → per-K collect fraction constant
  (GvrParams-style, production-legal).
- Placement-vs-h law confirmed: gathered rank i ↔ global rank ≈ i/h ⇒ f_hi
  must sit below h. Static (0.25,0.5,0.75,1.0) spans h∈[0.35,0.77] observed.
**Decision**: iter1 kernel = P1 order-stats (in-smem select on K gathered) +
one fused M=4 count round + speculative slot-collect at per-K column + band
refine; round-2 secant fallback (all_ge) + classic-P3 fallback (overflow).
No offline straddle-fracs dependency.
