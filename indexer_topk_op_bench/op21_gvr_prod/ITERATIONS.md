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

## Iter 1 — 2026-07-05 — single-CTA kernel v1 (place_mode=5 rank-quantile)
**What shipped in src/gvr_ms_op.py** (op19 sandwich base + mode-5):
histogram-quantile P1b placement (QBINS=256 over [g_min, g_max], columns at
qfracs 0.75/0.5/0.25 of K_valid, column0 = g_min anchor), M=4 R=1 fused
ladder + spec slot-collect, band refine; entry `gvr_ms` = ZERO dispatch
tables, 2-rule fuse gate: `bs <= NUM_SMS AND 4*K <= kC(5120)`.
**Exactness gate**: synth 54/54 (3 seeds x {512,1024,2048} x {8K,65K,262K} x
BS{1,16}) AND real 60/60 (pro 30L + flash 21L + v32 9L) — held through every
edit below.

**Bug found & fixed (the iter's main lesson)**: v1's P1b ran the 256-bin
cum-scan serially on tid0 (~1800 dependent smem ops) = +30µs FIXED,
N-independent (43µs @ N4K vs op19-p3 15.7µs). Isolated by same-file p3-vs-p5
A/B before touching code. Fix = in-place Hillis-Steele parallel suffix-scan
(8 double-barrier steps) + stateless parallel crossing check (largest bin
with suffix >= tgt). 45.5 -> 16.3µs @ N4K; beats p3R2 21-33% at large N.

**Second lever**: phase1_stats_stash — gather prev-K ONCE, stash values into
smem_keys (free until P3), P1b histograms from smem. Kills the K-load L2
re-gather (+25% L2 traffic at small N). P1 gm vs x20 0.865 -> 0.882.

**Fuse-gate data (event screen)**: fuse wins at BS<=NUM_SMS for K512/K1024
everywhere; fuse LOSES 13% at K2048 largeN (kC/K=2.5, slot overflow ->
collect + classic-P3 both paid). kC=4K/6K no help (6K = smem overflow at
232KB cap). Gate deviates from per-cell best by <=1-2% (noise).

**Event screen standing (vs op20 x20 240-key dispatch / radix_cutedsl)**:
P1 gm 0.882 / 0.965; beats radix at BS>=256 K512/K1024 (1.04-1.35);
P0 gm 0.775 / 0.837 — single-CTA aggregate-BW-bound at largeN smallBS as
expected; x20 wins there via fusP4T4/cluster4/mc (multi-CTA lineage) =
EXACTLY iter2's scope. BS64 smallN vs radix 0.72-0.84 = the known
deprioritized structural wall.

**nsys pure-kernel cold-L2 VERDICT (canonical; scripts/drive_nsys.sh +
scripts/nsys_verdict.py; rival = per-cell best of report CSVs B200 fp32
cold: radix single/multi CUDA+cuteDSL, SGLang; gvrbest = best existing
GVR-family op, no-regression reference)**:
- P0 gm rival/ms **0.830** (win 4/12): WINS the whole K1024 N65536 column
  vs every rival (1.03-1.06x, all BS 1-16); N131K 0.80-0.97; N262K
  0.55-0.84 (worst). gvr_op8 beats gvr_ms on ALL P0 cells (g/ms 0.56-0.89).
- P1 gm rival/ms **0.816** (win 1/24): SGLang owns midN-highBS (we sit
  0.77-0.94 of it); radix owns N4-8K BS64 (0.60-0.79 — the known wall);
  K2048 N16K BS256/1024 ties radix (1.00).
- **Attribution (N-scaling of the nsys medians)**: gvr_ms N-slope at P0
  ~87ns/Kelt (= ~2 effective single-CTA passes incl. gather+scan) vs
  gvr_op8 ~17ns/Kelt -> op8's P0 edge is pure multi-CTA aggregate L2 BW,
  NOT a smarter algorithm. Host-check at N65K-262K synth: NO spec-collect
  overflow (cnt@0.75col ~1.6-1.7K << kC 5120), straddle holds in round 1,
  band ~850-950. The fused path is healthy; the row is just too big for
  one SM. => iter2 multi-CTA C-chunk is THE P0 lever, as planned.
- Event-screen ratios overstated us by ~5-10% vs the nsys axis (op20
  red-card lesson re-confirmed; nsys stays canonical).

**Carry-forward levers**: (a) iter2 multi-CTA for P0 (op8 N-slope proves
5x aggregate-BW headroom), (b) P1b per-row cost at BS1024 (hist zero + 16
scan barriers; QBINS-vs-BS rule?), (c) K2048 fused-collect needs a kC
redesign or per-K collect column (iter0.5 finding), (d) 16-bit ports
untouched, (e) band ~900 >> band_accept 64 at K1024 -> P4 always runs the
256-bin rank-scatter; a tighter qfrac pair around 1.0*K could shrink band
(M=6 tightens p50 to ~0.5K per iter0.5) at the cost of M=6 scan tax.
