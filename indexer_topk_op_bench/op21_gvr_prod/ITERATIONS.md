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

