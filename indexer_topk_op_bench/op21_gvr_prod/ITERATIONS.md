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

## Iter 4 — 2026-07-05 — ablation pins P4; distributed P4 falsified too
**Ablation (no-op subclass overrides, /tmp probe, event BS1)**:
| component | K1024 262K C4 | K2048 262K C8 |
|---|---|---|
| P4 band snap (leader-only) | **3.9us** | **7.0us** |
| P3 slots + leader band gather | 2.1us | 3.1us |
| rest (P1+P1b+ladder+merge) | 20.7us | 19.8us |
P4 == exactly the remaining rival gap at both named holes.

**Hypothesis tested**: distribute P4 — per-CTA 256-bin histogram of the
LOCAL band, DSMEM merge, replicated cut-bin pick, bulk (bins > c*)
emitted distributed at prefix offsets, ONLY the boundary bin gathered to
the leader for the existing exact snap (provably exact; src flag
dist_p4, ~200 lines, kept as reference).
**FALSIFIED (event A/B, 10 cells)**: +0.1..+1.7us everywhere that
matters (K1024 262K BS1: 28.2 vs 26.6; only 512/262K a -0.2 wash).
Exactness held everywhere (smoke 27/27, real 180/180).
**Mechanism**: P4's 3.9-7us is dominated by the snap's OWN fixed
machinery, which the leader still runs on the boundary bin — the
pre-filter only ADDED 4 cluster barriers + a replicated 256-bin suffix
scan on every CTA. Combined with iter3: on this kernel, EVERY
"distribute the fixed part" move has lost to cluster-barrier cost.
**Standing**: defaults unchanged (dist_p1=False, dist_p4=False), nsys
verdict stays iter3's (P0 gm 1.054, 12/17).
**iter5 leads (re-ranked)**: (a) make the SNAP itself cheaper — op8's P4
is rank-scatter-exact, not histogram-snap, and op8 posts 20.3us at the
very cells where our snap costs 3.9-7us: port rank-scatter-exact as the
band refine (single-CTA benefit too); (b) P1-grid highBS SGLang gap
(P1b per-row cost, QBINS-vs-BS rule); (c) 16-bit ports (roadmap).

## Iter 5 — 2026-07-06 — rank-scatter-exact P4 lands; P0 flips to 1.104

**Lead (a) SHIPPED — phase4_band_rank_scatter (src/gvr_ms_op.py)**: op8's
exact rank-scatter P4 (`op8b_gvr_b300/.../gvr_topk_decode_cluster_rs.py`,
enable_p4_rank_scatter_exact — the PR #15709 primitive) ported as the band
refine, default ON (p4_rank_scatter=True; OP21_P4_RS=0 = legacy snap A/B).
Port deltas vs op8: band range [thr1, thr0) already known (min/max pass
drops out), rank target = runtime k_rem (not const K), all output
positions offset by m0. Chain = coarse kBins hist -> b* + rank_above ->
ONE fine 256-bin recursion on b* -> single scatter pass; replaces the
data-dependent snap-convergence loop entirely. One method in the base
class serves gvr_ms AND gvr_msc (all 3 call sites incl. the dist_p4
reference path).

**Lead (b) FALSIFIED — P1b QBINS=64 at bs > NUM_SMS**: paired event A/B
(OP21_QBINS, 14 P1 cells): gm q256/q64 = 1.004, everything within ±2.5%.
The 8-step suffix scan is NOT the highBS per-row bottleneck; rule
reverted (constant 256), env knob kept.

**Exactness gates (all green)**: synth 54/54 (smoke_exact) + real
single-CTA 60/60 + real x C {2,4,8} 180/180 + adversarial preIdx
(random/half-invalid x ms/C4/C8) 36/36 (new scripts/smoke_real_msc.py).

**Event screen (paired same-process via env-keyed compile cache, 22
cells)**: gm snap/rs 1.058, rs wins 18/22; biggest at 262K BS8/16
(1.24-1.25) and K2048 (1.05-1.12).

**GOTCHA — first nsys grid discarded (GPU0 thermal)**: umb-b200-035 GPU0
now idles at 79C (GPU1 31C) and throttled the later grid cells +2.3-2.7us
(~+13%), mimicking a regression at 131K/262K while 65K cells improved.
GPU1 probe reproduced iter3-era numbers within 1.5% (K512 262K BS1: snap
20.19 vs iter3 19.87us) => iter1-4 baselines clean, GPU0 cooling since
degraded. Poisoned run archived (results/nsys/iter5_gpu0_thermal_poisoned/);
canonical grid re-run with GPU=1. Memory note saved.

**nsys pure-kernel cold-L2 P0 VERDICT (canonical, GPU1; drive_nsys_iter2.sh
+ nsys_verdict.py msa; rival = per-cell best of report CSVs B200 fp32)**:
gm rival/ms **1.104**, win **13/17** (iter3: 1.054, 12/17); vs best
GVR-family op gm **1.007** (iter3 0.958) — first iter at/above op8
aggregate. K2048 131K BS1 flips to a win (19.20us, 1.046).

| K | N | BS | ms_us | rival | r/ms | | K | N | BS | ms_us | rival | r/ms |
|---|---|----|-------|-------|------|-|---|---|----|-------|-------|------|
|1024|65536|1|16.48|19.96|1.211| |1024|262144|8|22.34|24.16|1.082|
|1024|65536|4|17.31|20.10|1.161| |1024|262144|16|22.98|31.36|**1.365**|
|1024|65536|8|17.63|20.54|1.165| |512|131072|1|16.70|19.12|1.145|
|1024|65536|16|18.11|21.16|1.168| |512|262144|1|19.71|19.13|0.971|
|1024|131072|1|18.02|19.91|1.105| |2048|131072|1|19.20|20.09|1.046|
|1024|131072|4|18.85|20.11|1.067| |2048|262144|1|22.05|19.81|**0.899**|
|1024|131072|8|19.36|20.71|1.070| |2048|262144|16|24.26|31.97|1.318|
|1024|131072|16|20.03|24.68|1.232| | | | | | | |
|1024|262144|1|21.12|20.06|0.950| | | | | | | |
|1024|262144|4|21.82|20.43|0.936| | | | | | | |

**Remaining holes (all narrowed, none closed)**: K1024 262K BS1/4
0.950/0.936 (was 0.922/0.911; rival bar 20.1-20.4us, we're 1.1-1.4us
over); K512 262K BS1 0.971; K2048 262K BS1 0.899 (was 0.829). Pattern:
pure smallBS-largeN scan floor — P4 is no longer the margin (rank-scatter
took 0.6-2.2us out); what remains is the C4/C8 aggregate-BW N-term vs
radix_cutedsl's flatter N-scaling. P1 highBS grid untouched (SGLang
0.77-0.94, structural).
**iter6 leads**: (a) C-scaling at the 262K BS1/4 holes: C=8 tier for
K1024/K512 at N262K BS<=4 (iter3 saw K1024 noise-level with the OLD
expensive P4 — the calculus may flip now that the serial tail shrank);
(b) P3/P4 residual: leader band gather + K output writes; (c) 16-bit
ports (roadmap); then iter5-roadmap items (dispatch distillation,
no-regress full grid, B300 cross-check).

## Iter 6 opening probes — 2026-07-06 (same session as iter5, GPU1 event)
**(a) C8-at-holes FALSIFIED-AS-MARGINAL**: C4/C8 paired event at the hole
cells: K1024 262K BS1/4 1.007/1.006, K512 262K BS1 1.013, K2048 262K BS1
1.007 (already C8 in auto), K1024 131K BS1 0.955 (C8 loses), BS16
collapse unchanged (0.615). +0.6-1.3% = noise; the serial-tail shrink did
NOT flip the C-scaling calculus. No gate change.
**(b) fresh phase ablation (no-op subclass, event BS1, rank-scatter P4)**:
| cell | full | P4_us | P3_us | scan floor (noP34) |
|---|---|---|---|---|
| K1024 262K C4 | 26.4 | 2.94 (was 3.9) | 2.75 | 20.7 |
| K512 262K C4 | 24.8 | 2.30 | 1.95 | 20.5 |
| K2048 262K C8 | 27.3 | 4.03 (was 7.0) | 3.36 | 19.9 |
Scan floor is already AT the rival bar (19.8-20.4us) => the remaining
1.1-2.2us nsys gap is still ALL in the leader-only P3+P4 tail (4.3-7.4us
event). iter6 design space: small-bin P4 fast path (band ~900 in ~1024
coarse bins => cnt(b*) usually tiny; skip the fine 256-hist when
rank_above + cnt(b*) == k_rem, or select among <=64 b*-members in
registers instead of building a full fine hist), cheaper P3 leader gather,
or accept the wall and go 16-bit (lead c). Respect the iter3/4 red line:
no new cluster barriers unless they buy >0.5us.

## Iter 6 — 2026-07-06 — small-bin P4 fast paths; P0 gm 1.125, op8 +2.6%

**Host probe first (scripts/proto_p4_smallbin.py, 68 synth+real rows)**:
replayed the P1b->ladder->sandwich->coarse-bin chain on host: cnt(b*)
p50=2 p90=3 **max=4** (band ~1K spread over >=1024 coarse bins); eq-hit
(rank_above+cnt(b*)==k_rem) 69%; cnt(b*)<=32 **100%**. The fine 256-bin
recursion is fallback-only in practice => GO.

**Shipped (src/gvr_ms_op.py phase4_band_rank_scatter)**: three-way branch
after the coarse search — (B) cnt(b*)<=32: stash the b* members' band
indices (smem_hist[8..39] — coarse hist is dead there, cnt already in a
register), ONE band pass emits above-members and stashes b*; warp0 exact
register ranking (32 constant-src shuffle_sync compares, tie by stash
order) emits at exact positions, NO fine hist, NO atomics on the b*
group, 2 barriers total. (A) big-bin equality: whole-bin emit, skip fine.
(C) fine recursion extracted to _p4_band_fine_scatter as the
distribution-shift fallback (never fires on probe data). Default ON;
OP21_P4_FAST=0 forces fine (A/B knob, env-keyed compile cache).

**Exactness (both modes)**: FAST=1: synth 54/54 + real 60/60 + C-smoke
27/27 + real x C 180/180 + adversarial 36/36. FAST=0 (forces the
extracted fine path): 54/54 + 60/60 — extraction is lossless.

**Event screen**: gm fine/fast 1.017, win 13/14. Sole apparent loss
K512 131K BS1 (0.957, reproducible across 5 paired reps) was REFUTED by
nsys (16.74 vs iter5 16.70us — flat): event-axis codegen jitter, not an
algorithmic regression. Lesson: a paired-event single-cell verdict can
still lie when the binary changes — nsys arbitrates.

**nsys pure-kernel cold-L2 P0 VERDICT (canonical, GPU=1)**: gm rival/ms
**1.125** (iter5 1.104), win 13/17; vs best GVR-family **1.026** (iter5
1.007). K1024/K2048 -0.3..-0.6us across the board.

| K | N | BS | ms_us | rival | r/ms | | K | N | BS | ms_us | rival | r/ms |
|---|---|----|-------|-------|------|-|---|---|----|-------|-------|------|
|1024|65536|1|16.10|19.96|1.240| |1024|262144|8|21.95|24.16|1.101|
|1024|65536|4|16.80|20.10|1.197| |1024|262144|16|22.62|31.36|**1.386**|
|1024|65536|8|17.09|20.54|1.202| |512|131072|1|16.74|19.12|1.143|
|1024|65536|16|17.57|21.16|1.204| |512|262144|1|19.94|19.13|0.960|
|1024|131072|1|17.73|19.91|1.123| |2048|131072|1|18.66|20.09|1.077|
|1024|131072|4|18.53|20.11|1.085| |2048|262144|1|21.47|19.81|**0.923**|
|1024|131072|8|19.01|20.71|1.090| |2048|262144|16|23.71|31.97|1.348|
|1024|131072|16|19.55|24.68|1.263| | | | | | | |
|1024|262144|1|20.74|20.06|0.967| | | | | | | |
|1024|262144|4|21.50|20.43|0.950| | | | | | | |

**Remaining holes**: the four 262K smallBS cells only — K1024 BS1/4
0.967/0.950 (0.7-1.1us over the bar), K512 BS1 0.960, K2048 BS1 0.923.
P4 is now essentially drained (fast path = 1 band pass + 2 barriers +
warp ranking; fine never fires). **iter7 leads**: (a) P3 leader tail
(slot walk vs DSMEM band gather ablation split first), (b) 16-bit ports
(roadmap), (c) then dispatch distillation / no-regress full grid / B300.

## Iter 7 — 2026-07-06 — P3 band remote-store push; P0 gm 1.249, 17/17

**Node move + re-anchor (protocol worked)**: new host umbriel-b200-047
(both GPUs 31C idle). Anchor cell K512 fp32 262K BS1 on iter6 code:
20.13us vs iter6-axis 19.94us (+0.95%, inside the 3% band) => same
measurement axis, iter5/6 tables transfer, no re-baseline. (Note: axis is
~1% SLOWER than 035-GPU1, so this iter's gains are slightly understated.)

**Split ablation first (scripts/ablate_p3_split.py)**: iter6's "P3" number
was the slot walk ONLY — the leader DSMEM band gather was INLINE in the
kernel body, invisible to the no-op-subclass harness, hiding inside the
"scan floor". Extracted it to `_p3_leader_band_gather` (behavior-identical
refactor, smoke-gated), then split (event BS1; gat excludes the 2 cluster
barrier pairs, which the push ALSO removes one of):
| cell | gather | slot walk | P4 |
|---|---|---|---|
| K1024 262K C4 | **1.66** | 0.51 | 3.39 |
| K512 262K C4 | 0.54 | 1.44 | 2.11 |
| K2048 262K C8 | **2.40** | 2.21 | 3.01 |
Gather = the bigger half at exactly the K1024/K2048 hole cells.
(NoWalk ablation gotcha: the walk publishes s_iscalars[0]=band; a bare
no-op feeds garbage p_cnt to the gather — the no-op MUST zero the
published counts. iter6's NoP34 had this hazard; decomposition used
noGat/noWG increments instead.)

**Shipped — P3 band remote-store push (src/gvr_msc_op.py, default ON,
OP21_P3_PUSH=0 = gather A/B)**: each CTA's global band prefix b_off is
already known BEFORE the walk (ladder-count publish), so the slot walk
writes band entries straight into the LEADER's smem at [b_off + wcb] via
new `st.shared::cluster` primitives (leader stores locally; peers fire
remote stores, visibility via the existing release/acquire cluster
barrier). Deletes the whole gather pass AND one cluster barrier pair +
the band-count publish. This is the sanctioned "make the serial phase
cheaper" direction — no new barriers, one fewer (iter3/4 red line
respected). dist_p4 (falsified reference) needs the local band copy =>
push forced off there.

**Exactness (all green, push ON)**: synth 54/54 + real single-CTA 60/60 +
built-in C {2,4,8} smoke + real x C 180/180 + adversarial 36/36; push-OFF
legacy path re-smoked (extraction lossless) + per-cell exact checks in
the paired A/B both modes.

**Event screen (paired same-process, 14 cells)**: gm gather/push 1.077,
win 14/14; biggest K2048 262K BS1 (1.195); P1 single-CTA canaries flat
(1.002) as expected.

**nsys pure-kernel cold-L2 P0 VERDICT (canonical, 047 GPU=0;
drive_nsys_iter2.sh + nsys_verdict.py msa; iter6 grid archived to
results/nsys/iter6_msa/)**: gm rival/ms **1.249** (iter6 1.125), win
**17/17** — first clean sweep of the P0 grid; vs best GVR-family
**1.139** (iter6 1.026). ALL four 262K smallBS holes closed:
K1024 BS1/4 0.967/0.950 -> **1.064/1.038**, K512 BS1 0.960 -> **1.064**,
K2048 BS1 0.923 -> **1.115**. Per-cell deltas match the event A/B
(−1.6..−3.7us); no run-order thermal signature.

| K | N | BS | ms_us | rival | r/ms | | K | N | BS | ms_us | rival | r/ms |
|---|---|----|-------|-------|------|-|---|---|----|-------|-------|------|
|1024|65536|1|14.11|19.96|1.414| |1024|262144|8|20.19|24.16|1.197|
|1024|65536|4|14.94|20.10|1.345| |1024|262144|16|20.77|31.36|**1.510**|
|1024|65536|8|15.20|20.54|1.352| |512|131072|1|15.10|19.12|1.266|
|1024|65536|16|15.68|21.16|1.349| |512|262144|1|17.98|19.13|1.064|
|1024|131072|1|15.87|19.91|1.255| |2048|131072|1|17.34|20.09|1.158|
|1024|131072|4|16.61|20.11|1.211| |2048|262144|1|17.76|19.81|1.115|
|1024|131072|8|17.02|20.71|1.217| |2048|262144|16|22.43|31.97|1.425|
|1024|131072|16|17.79|24.68|1.387| | | | | | | |
|1024|262144|1|18.85|20.06|1.064| | | | | | | |
|1024|262144|4|19.68|20.43|1.038| | | | | | | |

**Standing**: P0 goal MET (beats all report.html rivals on every P0 cell,
nsys axis). P1 highBS grid unchanged (single-CTA, SGLang 0.77-0.94
structural; deprioritized). **iter8 leads**: (a) 16-bit ports (roadmap),
(b) dispatch distillation (rules already <=3) + no-regress full grid
(largeN midBS/highBS + P1 canaries), (c) B300 cross-check.

## Iter 8 — 2026-07-06 — 16-bit lands (C8 flips at 16-bit); P1 refreshed

**P1 nsys refresh (24 cells, iter7 HEAD, 047 GPU0; iter1 reps archived
results/nsys/iter1_ms_p1/)**: gm rival/ms **0.901** (iter1 0.816), win
5/24 — the iter5-7 band-refine work carried ~10% into the P1 grid (push
itself is a single-CTA no-op there). Structure unchanged: SGLang owns
midN highBS (we sit 0.86-0.96), radix owns N4-8K BS64 (0.68-0.86, the
deprioritized structural wall). New outright wins: K1024/K512 16384 BS64,
K512 16384 BS1024, K2048 16384 BS256/1024.

**16-bit port status**: the kernels (gvr_ms + gvr_msc incl. push + P4
fast paths) already compile & run 16-bit from the op18/19 lineage —
"port" reduced to validation + measurement + dispatch.
- Exactness: NEW scripts/smoke_real_16bit.py — real captures
  dtype-truncated (real_data_v2 per-dtype refs, tie-robust): 60 layers x
  {ms, C4, C8} x {bf16, fp16} = **360/360 exact**; synth C4 6-cell + C8
  12-cell spot checks exact.
- First 16-bit nsys grid (C4-era rule, archived
  results/nsys/iter8_16bit_c4rule/): bf16 gm 0.973 (7/17), fp16 0.985
  (7/17). Diagnostic: our largeN smallBS time ~flat fp32->bf16 (18.85 ->
  18.66us at K1024 262K BS1 C4) while radix drops 20.1 -> 14.9 — at
  16-bit the C4 scan is NOT L2-BW-bound; radix's flatter multi-CTA
  N-scaling wins.
- **C8-at-16bit probe: the fp32 falsification does NOT transfer.** Event
  C4/C8 at bf16: 262K BS1/4/8 1.10-1.14, 131K BS1/4 1.08-1.10, 65K BS1
  1.007, 131K BS8 1.019 (marginal), 262K BS16 0.713 (same collapse);
  fp16 matches. Halved scan cost re-weights the serial tail => 8-way
  chunking pays at 16-bit where it was noise at fp32.
- **Shipped dispatch rule** (gvr_ms_auto, ONE new production-legal
  comparison on compile-time dtype + BS + max-N):
  `C=8 iff 16-bit AND N >= 65536 AND N >= 32768*BS` — covers exactly the
  measured win region, excludes the BS16 collapse and the 131K BS8
  marginal. Exactness for 16-bit C8 already gated (real 360/360 incl.
  C8; synth C8 12/12).

**nsys pure-kernel cold-L2 16-bit P0 VERDICT (canonical, 047 GPU0, C8
rule)**: bf16 gm rival/ms **1.028**, win **11/17**, vs best GVR-family
1.211; fp16 gm **1.043**, win **11/17**, gvrbest 1.253. C8 cells
improved 0.9-2.5us over the C4-rule grid (e.g. bf16 K1024 262K BS1
18.66 -> 16.19, BS8 19.33 -> 16.77 = flips to 1.062 win).

**Remaining 16-bit holes (all largeN smallBS)**: 262K BS1/4 K1024
0.918/0.917 (bf16), K512 262K BS1 0.965, K2048 131K/262K BS1
0.964/0.864; 131K BS8 0.960-0.984 (C8 marginal there, excluded).
Mechanism: with C8 the SM count is no longer the binding constraint —
the per-element scan cost (16->32 cvt + fp32 compare ladder) is. Lever =
16-bit native compares in the streaming ladder (PLAN fine-grain #6,
untested). **iter9 leads**: (a) 16-bit native-compare ladder,
(b) dispatch distillation writeup + no-regress full grid, (c) B300
cross-check (needs a B300 host; write launch prompt).

## Iter 9 — 2026-07-06 — native 16-bit ladder; bf16 gm 1.091 (15/17)

**Microbench first (probe/count16_native.cu, B200, cold-L2 events)**:
standalone M=4 count ladder, 16-bit: cvt->fp32 (current) vs p1
(set.ge.u32 mask + packed u32 add) vs p2 (set.ge.16x2 1.0/0.0 +
add.rn.16x2 accumulate). p2 = **1.73x** at N262K single-CTA, **1.21x**
at the C8 slice (N32K BS8), ~1.00 at full occupancy (BS148,
L2-BW-bound). Counts bit-match the fp32 path on every config once
thresholds are pre-quantized to the dtype grid => GO, and the
quantization-equivalence story is validated.

**Shipped (src/gvr_ms_op.py + gvr_msc_op.py, default ON,
OP21_P2_NATIVE=0 = cvt A/B)**:
- P1b quantizes all M threshold columns to the dtype grid at emit
  (`quant_f32_16`; cvt.rn is monotonic so non-descending survives;
  column 0 = g_min is a data value, already on-grid). This makes the
  16-bit-domain ladder compares bit-equivalent to the fp32 compares in
  P3/P4/fallback — one quantize point, every consumer consistent.
- Both fused ladders (single-CTA + msc slice) get a paired path:
  u32-typed 256-bit loads (pairs land packed, no repack),
  `set.ge.{bf16x2,f16x2} + add.rn` packed accumulate for the count
  columns (flushed to int32 every 16 vec iters; per-half growth <=
  8/iter => <=128 << the 256 bf16 integer grid), `set.ge.u32` packed
  mask for the collect column (exact per-element slot cursor + rare
  half-extract cvt on candidates only). Tails stay fp32 (equivalent
  under quantized thresholds). fp32 binaries unchanged (const_expr
  pruned). dist_p1 forces the flag off (its P1b does not quantize).

**Exactness (all green)**: synth 18-cell 16-bit C4/C8/ms probe + fp32
built-in smoke (no-regress) + real 16-bit gate **360/360** with the
native ladder ON.

**Event screen (28 cells paired)**: gm cvt/native 1.024, win 20/28 —
bf16 all holes improve (+3-11%); fp16 showed two ~5% single-cell
apparent regressions (131K BS1 0.943) that nsys REFUTED (13.54 vs
13.38us = ~1%): the iter6 codegen-jitter lesson again.

**nsys pure-kernel cold-L2 16-bit P0 VERDICT (canonical, 047 GPU0;
iter8 grid archived results/nsys/iter8_16bit_c8rule/)**:
- **bf16 gm rival/ms 1.091 (iter8 1.028), win 15/17** (was 11/17); vs
  best GVR-family 1.285. The whole K1024 column now wins incl. 262K
  BS1 0.918->1.035 and BS4 0.917->1.002; K512 262K 0.965->1.078;
  131K BS8 flips 0.960->1.025.
- **fp16 gm 1.055 (iter8 1.043), win 12/17**; gvrbest 1.267. 262K
  BS1/4 narrow to 0.977/0.968; 131K BS8 0.996 (par).
- Remaining 16-bit holes: K2048 131K/262K BS1 (0.95-0.96 / 0.88 both
  dtypes) — the K-proportional P3/P4 tail at cr=1, not the ladder —
  plus the three fp16 near-par cells above.

**Standing after iter9**: fp32 P0 17/17 gm 1.249; bf16 15/17 gm 1.091;
fp16 12/17 gm 1.055; every dtype's gm > 1 vs per-cell best rival, and
1.27-1.29 vs the best existing GVR-family op. **iter10 leads**:
(a) K2048 16-bit BS1 tail (the last structural hole family; C8 already
applied — needs a K2048-specific P3/P4 look or acceptance),
(b) fp16 262K BS1/4 residual (~3%), (c) no-regress ship review + B300
(B300_PROMPT.md ready).

## Iter 10 — 2026-07-06 — ship review + upstream assessment; B300 fp32 verdict

**Session context**: B200 host umbriel-b200-019 (GPU0 cooling broken 75C
idle — never used; docs-only iteration, no kernel edits). The parallel B300
session (umb-b300-dp-185) died mid-16-bit-sweep but its background driver
had already completed the fp32 grid on NFS.

**B300 fp32 cross-check verdict (nsys cold-L2, dp-185 GPU0, rival = B300
CSV rows)**: gm rival/ms **1.268, win 17/17** (B200 axis: 1.249, 17/17) —
fp32 is HW-invariant; same pattern shape (weakest cell K1024 262K BS1:
1.044 B300 vs 1.064 B200; strongest 262K BS16 1.57/1.51). bf16 partial
(11/17, K1024 column only) 11/11 wins gm 1.097 — matches the B200 pattern;
the 6 missing cells are exactly where B200's losses live (K2048/K512
tails). 16-bit completion recipe = B300_RELAUNCH_PROMPT.md (any B300 host;
archive dp-185 partials first, re-run all 51 cells on one axis).

**Deliverable a — SHIP_REVIEW.md**: consolidated no-regress table (17 P0
cells × 3 dtypes + 24 P1 fp32 canaries), dispatch distillation (3 C-rules +
fuse gate, all compile-time keys, CUDA-graph compatible), A/B env-knob
table with per-knob measured gains, exactness standing, ship risks.
nsys_verdict.py gained OP21_NSYS_DIR (regenerate tables from archives
without touching a live sweep dir); iter9 16-bit tables + iter8 P1 table
reproduced exactly from archives.

**Deliverable b — UPSTREAM_ASSESSMENT.md**: production surface = origin/
main #14602/#15198/#15304 (kernel + CuteDSLGvrTopKDecodeRunner +
enable_heuristic_topk in dsa.py); rank-scatter P4 opt-in on
fork/feat/gvr-rank-scatter-p4 after the ec04147502 exactness revert.
Lever-by-lever port table; **P0 blocker found: op21 P4 path C carries the
same fixed-depth inexactness upstream already reverted — must become an
exact fallback (snap-on-residual-bin) before any default-ON port** (see
LEARNINGS). Recommended route = Strategy B (kernel-variant PR chain,
opt-in → tests → dispatch flip), with the e2e plan staged as unit →
kernel nsys (B200+B300) → dsv4-pareto-bench 3-arm A/B (OFF/ON-old/ON-new)
→ gsm8k accuracy canary → soak + default flip.

**Iter 10 addendum — K2048 16-bit BS1 tail ablation (ACCEPTED as
structural)**: scripts/ablate_16bit_tail.py (event cold-L2 paired split on
umbriel-b200-019 GPU1 — relative same-process split, throttle-immune, no
re-anchor needed; absolute us NOT on the 047 axis). full/noP4/noWG at HEAD
defaults (push+RS-P4+native ladder ON), C8:
| cell | full | noP4 | noWG | P4_us |
|---|---|---|---|---|
| K2048 131K bf16 | 20.70 | 16.99 | 20.26 | 3.71 |
| K2048 262K bf16 | 22.24 | 18.56 | 20.70 | 3.68 |
| K2048 262K fp16 | 22.08 | 18.59 | 21.09 | 3.49 |
| K1024 262K bf16 (green ref) | 20.19 | 17.06 | 18.43 | 3.14 |
P4 at K2048 is only +0.5us over the green K1024 reference (K-proportional
output writes) — NOT the anomaly. The K2048 penalty lives in the FLOOR
(noWG +2.3us vs K1024 at 262K bf16): the K-proportional P1 preIdx gather +
P1b histogram at cr=1, against a radix 16-bit rival bar that is K-flat
(14.84 vs 14.87us). No P3/P4 lever exists; bf16==fp16 within noise
confirms it is not a ladder/dtype effect. ACCEPTED: the 0.88-0.96 K2048
16-bit BS1 family is a documented structural wall (v3.2 geometry only —
DSv4 Flash/Pro are K512/K1024, unaffected). noWG runs P4's degenerate
band=0 machinery, so noP4-noWG is negative by construction (the two
ablations are not nested).

**Iter 10 addendum 2 — B300 cross-check COMPLETE (B300_RESULTS.md)**:
full 51-cell single-axis re-run on umb-b300-dp-192 (Option A relaunch;
dp-185 partials archived iter10_b300_dp185_partial/). **fp32 gm 1.268
(17/17) / bf16 1.089 (15/17) / fp16 1.053 (13/17) — HW-INVARIANT** (B200:
1.249 17/17 / 1.091 15/17 / 1.055 12/17; all gm deltas <=0.02, zero
win->loss flips; sole loss->win flip = fp16 K1024 262K BS1 +4.2%, below
the 5% callout bar). K2048 16-bit BS1 tail reproduces on B300 (bf16
0.958/0.876) — independent confirmation of the ablation's structural-wall
verdict. NUM_SMS=148 on both parts => dispatch thresholds bit-identical,
no boundary pathology. Cross-B300-node bonus: dp-192 fp32 reproduces
dp-185 per-cell within ~±1% (same gm to 3 decimals). Campaign
measurement phase CLOSED; UPSTREAM_ASSESSMENT Stage-1 pre-port baseline
done on both architectures.

## Iter 11 — 2026-07-06 — P4 path-C exact fallback (upstream-port PR-1 step 1)

**Falsify first (NEW gate scripts/smoke_adversarial_band.py)**: planted
near-tie clusters (300 DISTINCT fp32 values spaced 2 ULP straddling the
K-th rank) inside a wide sparse band, production preIdx conventions
(K512/K1024 cr=4 offset-0; K2048 cr=1 caller prev-1/kernel +1).
**HEAD result: 0 ok / 72 FAIL** (ms/C4/C8 x all K/N/seed), vdiff
2.4e-7..2.1e-6 = the deepest-fine-bin stash-order truncation predicted by
the iter10 assessment (upstream ec04147502's exact failure mode).
Gate-authoring gotcha that cost one round: with cr=1 + raw preIdx the
kernel's +1 diagonal offset shifts every pointer into the bulk => no
straddle => the run silently exercises the fail-soft BASELINE path (which
is exact) — adversarial harnesses MUST follow the production preIdx
conventions or they test the wrong path. (Path attribution via a no-op
probe subclass dumping into output_indices_row; also proved off-pointer
preIdx fail-soft = exact baseline, a bonus robustness result.)

**Fix (src/gvr_ms_op.py)**: phase4_band_rank_scatter path C (and the
p4_smallbin=False branch) now falls back to phase4_band_snap_hist — the
value-EDGE snap (block_band_snap_iter steps thr onto actual data values,
so ==sel_thr is a true tie group; any k_rem-cut of it is exact).
_p4_band_fine_scatter (fixed-depth 1024x256 sub-histogram) DELETED: a
fine bin is a value INTERVAL, not a tie group — cutting it in stash order
is unfixably inexact at fixed depth. OP21_P4_FAST=0 semantics change:
"always fine" -> "fast paths off, always snap". Paths A/B and all entry
conditions byte-identical.

**Gates (all green, 019 GPU1)**: adversarial-band 72/72 (was 0/72) +
FAST=0 variant 72/72; synth 54/54; real 60/60; C {2,4,8} smokes; real x C
180/180 + adversarial preIdx 36/36; real 16-bit 360/360.

**Perf (nsys cold-L2 old-vs-new A/B, same GPU, 5 cells)**: gm new/old
0.996 — flat within the +-2.2% jitter band (K1024 262K/65K BS1 +0.7%,
K512 262K -0.5%, K2048 262K -2.2%, bf16 262K -0.4%); path C never fires
on real/synth data so the verdict tables in SHIP_REVIEW/B300_RESULTS
stand unchanged. (Method gotcha: `nsys profile -c cudaProfilerApi` exits
143 on success — a `set -e` A/B driver dies silently at the first cell.)

**Standing**: the UPSTREAM_ASSESSMENT P0 port blocker is RESOLVED in
op21; PR-1 can now port the kernel with exactness unconditional. The
adversarial-band gate joins the per-iteration gate suite.

## Iter 12 — 2026-07-07 — PR-1 step 2: kernel-variant assembled + full gate suite

**Deliverable — `port/gvr_topk_decode_ms.py` (3436 lines, the PR-1 kernel
artifact)**: assembled by `port/assemble_ms.py` — deterministic extraction
(exact line ranges + content-asserted edits, every slice/edit fails loudly
on source drift) from the frozen iter11 sources: vendored #14602 base
(copy-atom/reduces, block_count_ge, phase3 stream-write worker
[renamed from phase3_collect_candidates], snap-iter, phase4_histogram_snap)
+ op18 block_count_ge_multi + op21 ms (16-bit PTX helpers, fused ladder,
sandwich P3, band P4 incl. iter11 exact path-C, stats-stash P1,
rank-quantile P1b, kernel body) + op21 msc (st.shared::cluster helpers,
slice ladder, distributed P3, leader gather, cluster body). Dropped as
const_expr-dead: place_mode 0-4 tables, smem-row, dist_p1/p4, secant P2,
all OP21_* env knobs (constructor flags now). Two classes:
`GvrMsKernel` / `GvrMsClusterKernel`; module is torch-free; imports match
upstream main's unified gvr_topk_decode.py (cluster primitives are
module-level there; bench-local validation goes through `port/portshim/`
re-export shims — the artifact itself is never edited).

**Gate results (b200-027 GPU0, assembled kernel imported through portshim,
i.e. the exact bytes that ship)**:
| gate | grid | result |
|---|---|---|
| 1 synth fp32 | K{512,1024,2048} x N{8K,65K,262K} x BS{1,16} x 3 seeds, ms | 54/54 |
| 2 synth 16-bit | bf16/fp16 x 3(K,N) x BS{1,8} x {ms,C8} | 24/24 |
| 3 adversarial band (iter11 gate) | 24 cases x {ms,C4,C8} | 72/72 |
| 4 real x C | pro30+flash21+v32 9 layers x {ms,C4,C8} | 180/180 |
| 5 selection identity vs bench ops | 7 (K,N,path) spot cells | 7/7 sorted-set bitwise equal |
| 6 next_n varlen (NEW — bench never ran next_n>1) | nn{2,4} x {K512/cr4,K1024/cr4,K2048/cr1} x {ms,C4}, per-request varlen | 12/12 |
| upstream main-test grid (incl. the 30 adversarial cases) | 4 (dtype,K) pairs x N{4K,65K} x varlen x nn{1,2} x BS{1,32} x cr{1,4} x hit{0,.5} x cs{1,4}, minus sort-indirect | 384/384 (tie-aware upstream reference; 128 skipped per upstream rules) |

**KEY FINDING (gate-5 method)**: GVR output row ORDER is run-to-run
nondeterministic (P3/P4 smem atomicAdd emission cursors; the BENCH kernel
itself permutes across back-to-back identical calls — verified 4x). A
positional `torch.equal` old/new A/B false-fails 7/7 with 82-478 permuted
slots while the SORTED index sets are bit-identical. Equivalence criterion
for any GVR A/B = sorted-set equality (LEARNINGS iter12).

**Contract-gap status (UPSTREAM_ASSESSMENT §5 item 3)**: next_n/varlen was
a VALIDATION gap, not a code gap — the contract (row//next_n, cr=1
diagonal preIdxOffset, per-row actual_kv_len) is inherited verbatim in
both kernel bodies; gate 6 closes it. GvrParams kC map: vendored and
upstream tables are byte-identical (18 entries) — resolved by the kernel's
own GvrParams.get. return_output_values=False: const_expr'd, ctor-assert
indices-only. sort-indirect + LB: deliberately NOT in PR-1 step 1 (route
those batches to the classic runner; op#9 dispatcher lesson).

**Runner extension draft — `port/runner_ms_extension.py`**: paste-ready
`CuteDSLGvrTopKDecodeMsRunner` + opt-in custom op
`trtllm::cute_dsl_gvr_topk_decode_ms` + register_fake, for the
IS_CUTLASS_DSL_AVAILABLE block of cute_dsl_custom_ops.py. Tuning = op21
`_config` verbatim (T/256-bit/min_blocks — the SHIP_REVIEW tables were
measured with exactly these); cluster policy = `gvr_ms_auto` verbatim
(16-bit C8 rule, K2048-fp32-hugeN C8, C4 one-wave rule) + hw clamp; all
dispatch keys capture-time constants (CUDA-graph identity by
construction). All referenced upstream helpers verified present on
origin/main (a0c406ff88): _get_num_sms, _query_max_cluster_size,
_TORCH_TO_CUTLASS_DTYPE, is_sm_100f, logger. #15709 NOT merged at that
SHA — irrelevant to PR-1 (sibling-file route).

**Validation harness (committed)**: port/validate_port.py (gates 1-5),
port/run_gate6_nextn.py, port/run_upstream_cases.py +
port/_upstream_test_helpers.py (upstream _make_inputs/_tie_aware_check
extracted VERBATIM from origin/main), port/portshim/.
