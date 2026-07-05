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

## Iter 2 — 2026-07-05 — row-chunked C-CTA cluster (gvr_msc / gvr_ms_auto)
**What shipped in src/gvr_msc_op.py**: C CTAs per row (cluster launch),
REPLICATED P1 stash + P1b seeding (identical thresholds per CTA, zero
comm), slice-aware fused M=4 ladder (64-elt-aligned chunks, global slot
indices), DSMEM merge of M counts (one cluster barrier), distributed P3
direct-write straight to the output row at rank-prefix offsets, leader
DSMEM band gather (op8 Shift-D pattern) + unchanged exact P4 band snap.
Fallback (pair=(0,1)/no-pair/band>kC/overflow) = leader-only classic
collect+snap over the full row. `gvr_ms_auto` = gvr_ms + ONE extra rule:
C=4 iff N >= 65536 AND 4*BS <= NUM_SMS.

**Bugs found & fixed**:
1. Leader fallback fed phase3_collect_candidates SLICE-local smem_ptcnt —
   the vendored P3 prefixes over per-thread counts at s_thr[0] over the
   SAME [0,N) striding, it does NOT recount (646/1024 output holes on pro
   L4, pair=(0,1) case h<0.5 at midsize N). Fix: leader block_count_ge
   full-row recount before collect.
2. My d_off/b_off/m1g scratch initially sat in s_iscalars[1..3] — those are
   the vendored done/cnt_lo/cnt_hi slots; moved to dedicated s_cluster
   slots and re-seeded s_iscalars before the fallback phases.

**Exactness**: built-in smoke 27/27 (C in {2,4,8} x 3K x 3N); real captures
180/180 (pro30+flash21+v32-9, C in {2,4,8}); adversarial preIdx: random +
half-invalid exact 16/16; all-invalid emits identity = inherited vendored
degenerate contract (single-CTA gvr_ms bit-identical, never occurs on real
rows post-warmup-drop).

**Event screen (P0 17 cells, vs per-cell best C)**: gm rvl/best **1.041**
(single-CTA was 0.84), x20/best 0.968, ms1/best 1.320 (cluster speedup).
C4 best-or-tied 15/17; C8 gains <=5% at N262K BS1 only and collapses at
BS16 (43.8 vs 28.5us) -> dropped, ONE dispatch rule.
Remaining event-screen losses: K1024 131K BS1/4 vs x20 (0.88-0.90, radix
~tie), K2048 262K BS1 vs radix 0.85.

**nsys pure-kernel cold-L2 P0 VERDICT (canonical; drive_nsys_iter2.sh +
nsys_verdict.py msa; rival = per-cell best of report CSVs B200 fp32)**:
gm rival/ms **1.051**, win 12/17 — P0 flips to a WIN (iter1: 0.830, 4/12).
vs best existing GVR-family op gm 0.958 (op8 per-cell), incl. wins at
K2048 262K BS1/16 (1.05/1.07) and 512 262K (1.00).

| K | N | BS | ms_us | rival | r/ms | | K | N | BS | ms_us | rival | r/ms |
|---|---|----|-------|-------|------|-|---|---|----|-------|-------|------|
|1024|65536|1|17.66|19.96|1.130| |1024|262144|8|23.01|24.16|1.050|
|1024|65536|4|18.34|20.10|1.096| |1024|262144|16|23.55|31.36|**1.331**|
|1024|65536|8|18.69|20.54|1.099| |512|131072|1|16.90|19.12|1.132|
|1024|65536|16|19.26|21.16|1.098| |512|262144|1|19.87|19.13|0.963|
|1024|131072|1|18.85|19.91|1.056| |2048|131072|1|21.18|20.09|0.948|
|1024|131072|4|19.61|20.11|1.025| |2048|262144|1|24.83|19.81|**0.798**|
|1024|131072|8|20.06|20.71|1.032| |2048|262144|16|26.50|31.97|1.207|
|1024|131072|16|20.67|24.68|1.194| | | | | | | |
|1024|262144|1|21.76|20.06|0.922| | | | | | | |
|1024|262144|4|22.43|20.43|0.911| | | | | | | |

**Remaining holes (iter3 targets)**: K1024 262K BS1/4 (0.91-0.92 vs
radix_cutedsl ~20.1-20.4us — PLAN's named target needs another ~10%);
K2048 262K BS1 (0.80 vs radix_cutedsl_multi — worst; cr=1 K-proportional
costs + C4-only); P1 highBS gap unchanged from iter1 (SGLang 0.77-0.94).
Levers: C=8 gains ~5% at exactly the losing BS1 cells (needs a
BS1-only tier or launch-geometry fix for the BS16 collapse), P1b per-row
cost, K2048 collect column.

## Iter 3 — 2026-07-05 — dist-P1 falsified; C8 tier lands K2048-only
**Hypothesis tested**: the iter2 fixed-cost analysis blamed the REPLICATED
per-CTA K-gather (P1). Implemented distributed P1 (each CTA gathers K/C
preIdx + DSMEM stats & histogram merges rebuilding identical global
seeds; src flag dist_p1, kept as A/B reference).
**FALSIFIED (event A/B, 8 cells)**: C4-dist is +0.6-1.7us WORSE at every
K512/K1024 cell (28.3 vs 26.6 @ 1024/262K/BS1) and a wash at K2048.
Mechanism: at BS<=16 all C CTAs gather the SAME addresses — after the
first CTA misses, the rest hit L2, so replication is nearly free; the 3
extra cluster barriers + merges cost more than the saved loads. Exactness
held (180/180 real, C 2/4/8) but the lever is dead on B200.
**C8 tier**: consistent win ONLY at K2048 hugeN BS<=4 (28.7/29.2 vs C4
30.3/30.7us across two runs); K1024 noise-level; BS16 collapse (47us).
gvr_ms_auto rule now: C=8 iff K>=2048 AND N>=196608 AND BS<=4 (K is a
compile-time dispatch key — production-legal), else C=4 rule unchanged.
**nsys delta (canonical)**: K2048 262K BS1 24.83 -> 23.90us (rival ratio
0.798 -> 0.829); K2048 131K BS1 unchanged (stays C4). P0 gm rival/ms
1.051 -> **1.054**, still 12/17 wins.
**Where the remaining fixed cost actually lives (revised)**: NOT P1
replication. Candidates for iter4: leader-only P4 band snap + K output
writes (~3-5us serial-ish at BS1), band ~900-1800 >> band_accept, P1
cold-gather latency itself (~1024 DRAM lines, irreducible without
prefetch overlap), ladder tail. Also the P1-grid highBS SGLang gap
(P1b per-row cost) remains untouched this iter.

