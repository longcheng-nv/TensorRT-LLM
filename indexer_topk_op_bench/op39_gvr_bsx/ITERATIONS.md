# op39 iterations

## iter 0 — 2026-07-23 — GO (feasibility cruxes)
Hypothesis: post-op38 the 1.8-mean gap lives in (a) DRAM-bound big cells and (b) L2-resident BS>=16 cells.
Probe rung 0 x2 (NCU, pro_1024k BS512 + pro_64k BS256):
- (a) both arms read 2.05x floor (1.09-1.10GB vs 0.537GB); pr HW BW already 5.8TB/s (~80% roofline) -> pass-cut lever real but bounded ~1.9-2.0x; only 51/750 cases are DRAM-bound -> moves envelope mean 1.33->1.39 alone. NOT sufficient.
- (b) L2-resident mid cell: ALL SOL% <40 (SM 25-33, Mem 23-30, L2 8-11), 20us vs ~4us data floor -> latency/occupancy structural bound, room ~3-5x, 699 cases -> THE battleground. Need non-DRAM mean 1.79 (now 1.35).
Verdict: GO — arm design = tile-parallel single-pass collect (grid ~2xSM, (row,chunk) tiles, hint->conservative t_lo -> atomic append -> small exact top-K), unified for BS>=16; also cuts the 2nd pass for the DRAM-bound 51.
Ledger: none violated (new shape, not the falsified (TB,CS,MAXV,AR,HS) family).
Next: rung-2 microbench — oracle-threshold collect structure speed on pro_64k BS256 + pro_1024k BS512.

## iter 1 — 2026-07-23 — GO (rung-2 structure microbench)
Hypothesis: tile-parallel 1-pass collect (oracle thr) beats per-row-serial shape.
Probe: mb_collect v1 (ballot-aggregated append) FAILED structure (0.2-0.6x) — per-component ballot_sync tax + divergent-tail UB. v2 smem staging (CTA-local smem atomics, 1 global atomic/CTA, coalesced writeout):
- pro_1024k (DRAM-bound): 1.04/1.03/1.34/1.56x @BS16/64/256/1024 vs report pr — pass-cut lever CONFIRMED on silicon (event-axis, i.e. pessimistic vs nsys pr).
- flash_256k/v32_64k: 0.78-1.04 — parity band, handicapped by ~10-12us event-axis fixed tax (2 launches + memset).
- small cells (flash_16k/pro_64k): k1 flat ~12.5us at any BS<=256 -> fixed-overhead floor, NOT bandwidth.
Diagnosis: structure sound; k1 at 2.4TB/s (3x off 7TB/s roofline) needs ILP; K2 costs 5-40us (serial bucket scan + smem atomic contention); 2-launch+memset tax dominates small cells.
Next (iter2): fused single kernel — last-CTA-per-row reduce (drop 2nd launch+memset), 2xfloat4 ILP scan, parallel bucket suffix-scan; then nsys-axis screen.

## iter 2 — 2026-07-23 — GO (fused single kernel, event-axis)
Fused last-CTA-per-row reduce + self-clean (no memset launch) + 2xfloat4 ILP + warp-aggregated emit; chunks ladder swept.
Result (event-axis vs nsys report pr, i.e. pessimistic): pro_1024k 1.18/1.13/1.44/1.84 @BS16/64/256/1024; v32_64k 1.15/1.06/1.07/1.34; flash_256k 0.93-1.24; small cells (pro_64k, flash_16k) 0.66-0.70 at BS<=256 (event launch tax ~5-8us of a 13-22us reading), 1.0-1.2 at BS1024.
Diagnosis: structure GO everywhere at BS>=256 large-N; small-cell low-BS truth requires nsys axis.
Next: nsys-axis screen; then production arm = threshold-from-hint + undershoot fallback + tie-exact tail (reuse GVR P4 machinery on candidate set).

## iter 3 — 2026-07-23 — GO (nsys-axis screen, oracle thr)
8-cell battleground x BS16-1024 (56 cases): gm 1.3697, mean 1.3887, min 0.9667, <1.0: 2/56 (pro_64k_L30 BS128/256 = 0.967/0.990 parity band). Event-axis pessimism confirmed (~2x on small cells: flash_16k BS16 event 0.69 -> nsys 1.34).
Shape: BS16-32 strong (1.3-1.9), BS64-128 dip (1.07-1.5), BS512-1024 high (1.4-1.9); pro_1024k BS1024 1.90 ~= realistic pass-cut UB.
Caveats: oracle threshold (production needs hint->t_lo estimation + undershoot/overflow fallback + tie-exact tail); dip band + pro_64k parity are the residual battlegrounds; fused still at ~2.6TB/s vs 7 roofline at big cells.
Next (iter4): production arm v1 — pre-kernel hint-quantile t_lo (bucket hist on gathered hint values), reducer-CTA fallback rescan on undershoot/overflow, exact tie tail.
Data: results/f1_verdict.txt.

## iter 4 — 2026-07-23 — GO w/ falsifications (production arm v1->v2->v3)
v1 (t_lo = min hint): gate 0/225 EXACT (incl. adversarial const/near-tie) but 31/75 cells OVERFLOW (min-hint depth up to 189K cand) -> full-row fallback = perf collapse (gm 0.25 L1).
FALSIFIED: (t2 = K-th of stored subset; domain: real captures w/ positional structure; evidence: nsys+trace) — stored = first-flushed CHUNK slices (positionally concentrated), rank transfer wildly biased; rescue re-overflowed -> 285us final resorts. Root class: measurement... no — real-data structural bias (synthetic-uniform would have masked it).
FIX 1 (correctness): overflow predicate must be true_n > n_stored (single-chunk STAGE clip w/ true_n<=CAP was a silent exactness hole).
FIX 2 (v3): K0 = min-hint + position-unbiased STRIDED row sample (S<=8192), t = max(hint, sample quantile @ ~2K target) -> rescue now rare (K2 1.3us), all cells exact.
Status after v3b (T=2K, 2-level sample select, adaptive S): loss band event-axis 0.24-0.71 vs pr — threshold machinery tax ~3.5x over the oracle bound (gm 1.37 nsys). K0 ~12us + K1 ~29us at pro_64k BS16 vs oracle-K1 ~9us.
Next levers: (a) K0 -> <=3us: parallel suffix scan, fold hint+sample single pass, cut serial 256-loops; (b) K1 reduce single-level 2048-bucket for n~2K; (c) consider K0-into-K1 fusion or persistent cooperative launch; then nsys re-screen + full-envelope sweep.

## iter 5 — 2026-07-23 — GO (threshold-tax reduction, in progress)
Warp-parallel suffix search (replaces thread0 serial bucket scan in both reducer and K0 sampler; the serial scan was ~12us exposed at low-BS reducers) + fixed-point sample stride + T=2K target + 2-level sample select + adaptive S.
nsys split pro_64k BS16: K0 8.9->6.1us, K1 20.9->17.5us, K2 1.35us (rescue quiet). Event-axis loss band now 0.28-0.75 (from 0.18-0.71 pre-warp-scan; was 0.02-0.05 at iter4 v1).
Gap to oracle bound (fused 9.2us nsys at this cell): K1 +8us (2x candidates from T=2K + 4-level reduce exposure), K0 6.1us should be <=2.
Named next levers: (a) K1 reduce single-level 2048-bucket (11-bit) for n~2K; (b) K0: S=2048, fuse hint+sample phases, skip sample when npad<=CAP; (c) re-screen nsys 8-cell battleground w/ production thresholds vs oracle f1; (d) win-band check: pro_1024k BS256 event 148 vs oracle event 76 — verify rescue/threshold behavior on low-hit L46 big cells; (e) dispatch: arm only where it wins + op38 v3 elsewhere, then full-envelope sweep + [worst,real,best].

## iter 5b — 2026-07-23 — GO (clustered sampling)
FALSIFIED: (flat-stride sampling at stride == cache line; domain: any DRAM-resident row; evidence: nsys) — touches every DRAM line of the row, K0 = 46us at pro_1024k BS256 (a full extra pass).
Fix: 256 probes x 32 consecutive values (1 line per probe, 1/32 of lines): pro_1024k BS256 event 148 -> 112-116us (x0.96-0.99 vs pr, nsys likely >1.0); pro_1024k BS16 x0.58-0.59. Sub-L2 cells unchanged (0.27-0.55 event) — K1 reduce + K0 fixed cost still the battleground there.
All exact. Next: K1 single-level 2048-bucket reduce; K0 <=2us; arm_nsys.py 8-cell production screen; envelope sweep + dispatch-vs-op38v3.

## iter 5c/5d — 2026-07-23 — partial (production screens a1/a2 + undershoot rescue)
a1 (T=2K): nsys gm 0.621/min 0.347, wins only large-npad x BS>=256 (pro_1024k BS512-1024 1.34-1.46, flash_512k 1.10-1.22, v32_128k BS1024 1.16).
FALSIFIED: (T=1.25K tight target; domain: large npad >=262144, clustered sampling; evidence: nsys a2) — tail rank r=40 under cluster variance -> undershoot -> full-row resort storms (pro_1024k 1-7ms, gm 0.45). Tight-T is only safe with a cheap undershoot rescue.
v3f: T back to 2K + K0 dual quantile (primary 2K, fallback 6K in thr[BS+row]) + undershoot rescue at fallback (no more full-row for undershoot). Gate 0/225 + adversarial OK; pro_1024k healthy again (BS256 0.96 event).
Residual truth: sub-L2 cells (npad<=16K) 0.26-0.41 — 3-launch + reducer exposure vs pr's efficient single-kernel serial shape; large-npad BS16-128 0.55-0.97.
Strategy note: combined dispatch (op38 v3 for small-npad/low-BS + arm for large-npad) is the realistic ship shape; threshold tax (K0 6us, K2 1.3us, reduce depth) remains the named battleground to unlock the oracle bound (gm 1.37) beyond BS>=256.

## iter 5e — 2026-07-23 — checkpoint (a3 screen of v3f)
a3: gm 0.5902 / mean 0.6317 / min 0.352, <1.0 50/56 — robust now (no resort storms; all exact; rescue cheap) but dual-quantile K0 costs ~1-2us everywhere: win band trimmed slightly (pro_1024k BS1024 1.455->1.425, BS512 1.293; flash_512k BS512+ 1.03-1.14; v32_128k BS1024 1.08).
Band structure: large-npad BS>=256 = arm win band; everything else pr-dominated pending threshold-tax removal.

## iter 6 — 2026-07-23 — WASH (survivor-compaction reduce)
Level-0 boundary-bucket survivors compacted to smem ping-pong; levels 1-3 scan survivors only. Event-axis: pro_64k BS16 41.4 (was 39.7), v32_64k BS16 60.6 (worse) — reduce depth was NOT the dominant sub-L2 cost; combined-pass ballot overhead offsets the savings. Kept (correct, helps FROM_ROW final-resort), but not a tax lever.

## STRATEGIC NOTE (iter6) — oracle bound vs the 1.8 bar
The arm's own oracle bound (perfect thresholds) measured gm 1.37 / mean ~1.39
on the battleground sample — BELOW the campaign's mean>=1.8 bar. Even a
zero-tax production threshold cannot reach 1.8 with the CURRENT collect shape:
- big-N band: collect at ~2.6 TB/s vs ~6-7 achievable -> up to ~2x more there
  (pro_1024k BS1024 1.9 -> ~2.5+) IF the scan reaches roofline (occupancy/ILP:
  currently 3 CTA/SM x 512thr, 50KB smem);
- sub-L2 band: any 3-phase chain has a ~6-10us latency floor vs pr 9-15us ->
  per-cell cap ~1.3-1.7; the 1.8 MEAN therefore requires the big/mid bands to
  overshoot well beyond 1.8 to average out.
Next session MUST first: (1) push collect toward roofline (2xTB/s lever, wide
loads/occupancy variants) and re-run the ORACLE screen — if the oracle bound
itself cannot clear ~1.9-2.0 gm on the battleground, the 1.8 envelope mean is
structurally out of reach for this arm family and the campaign needs either a
different shape or a bounded verdict (double-lock rules apply).

## iter 7 — 2026-07-23 — verdict update (roofline screens f2/a4)
f2 ORACLE BOUND (roofline collect): gm 1.4349 / mean 1.4570 / min 0.955, 1/56 <1.0 (was 1.37/0.967/2). Roofline lever real but bounded.
a4 production: gm 0.583 — REGRESSION vs a3 (0.590) at K=2048: STAGE 6144->4096 with T=2K=4096 == STAGE trips chunk clips -> rescue storms on v32 (BS1024 0.672->0.594). Fix for relay: K-aware STAGE (template: 4096/5-occ for K<=1024, 6144/3-occ for K=2048) or T=1.5K for K=2048.
LOCK 1 (arm-family bound vs the 1.8 bar): even with FREE thresholds this collect shape measures gm 1.43 on the battleground — the mean>=1.8 envelope bar cannot be met by threshold engineering alone. Remaining candidate shapes within GVR framework before double-lock: (a) multi-row-per-CTA small-cell amortization (sub-L2 band cap is chain latency, oracle 1.2-1.5 there), (b) collect BW push beyond 4x-ILP (async copy / TMA bulk), (c) relaxed-constraint control (unlimited passes / no exactness) to complete the double lock if (a)/(b) stall.

## iter 9b/10 — 2026-07-23 — CLOSE-OUT (double lock complete)
a6 (post ledger-revert): production gm 0.6297 (best), min 0.378, no storms.
LOCK 1: oracle bound gm 1.4349 (f2). LOCK 2: envelope UB w/ measured constants (2.47 DRAM cap x 51 + per-band oracle MAX) = mean 1.738 < 1.8, feasibility-favoring.
VERDICT: 1.8-mean double-locked infeasible for this arm family; zero-regression exactness met (750/750); combined dispatch harvest = gm 1.3049/mean 1.3428. Full verdict: RESULTS.md.

## iter 10 — 2026-07-23 — GO (fallback-select skip)
a7: gm 0.6758 / mean 0.7112 / min 0.412 (a6 0.630/0.666/0.378) — mid-cell K0 second select removed, tax 2.3x -> 2.1x vs oracle 1.435. Named residual: K0 sampling body, K1 2K-candidate diet (blocked by r>=64 line — needs count-feedback, not tighter sampling), K2 empty launch, BS16-64 reducer exposure.
NOTE: e1 envelope sweep used FIXED chunks=592//bs; screens use a per-case mini-ladder — e1 understates the arm. e2 rerun with ladder queued.

## iter 12 — 2026-07-23 — GO (BS-dispatched ILP: 8 for BS<512, 4 for BS>=512)
e3 (post-iter11 envelope rerun) EXPOSED iter11: ILP-8's event-axis +5-6% did
NOT transfer — nsys envelope BS1024 arm times +9.5% gm (12/12 big-N cells
+3.6..13%, pr anchors fixed so no drift), BS512 +2%, BS2-256 wash. FALSIFIED:
(uniform ILP-8 collect; domain: BS>=512; evidence: e2-vs-e3 paired envelope).
Root: e2 baseline batch loop was ILP-4 (iter11 replaced 4->8 wholesale).
Missteps logged: (a) e4 first tried ILP=2 fallback (fell into the slow 2-wide
tail loop, BS>=512 still +7% vs e2); (b) `setsid` in a bg wrapper returns
immediately — poll shard logs, don't trust the wrapper exit.
Final shape: `arm_kernel<RESCUE, ILP>` generic batch loop; launch dispatch
BS>=512 -> ILP4, else ILP8. Gate 0/225 + adversarial OK (twice).
e5 verdict: combined gm 1.3150 / mean 1.3532 / min 0.7665, 0/750 inexact,
arm-beats-v3 wins 52 (e2 51, e3 44). BS1024 arm gm 0.846 / BS512 0.749 (best);
e5/e2 per-BS: BS>=512 parity 0.997-0.998, BS64-256 keep ILP-8 ~1%.
flash_512k BS1024 2.56x / BS512 2.21x; flash_1024k BS1024 2.12x.
NCU (results/ncu_bignt.txt, flash_512k BS1024 chunks=2): collect 75.4% DRAM /
45% SM / issue 0.49 — still ~25% from the 2.47-cap DRAM roofline; reducer
kernel latency-bound 6.1us (issue 0.13). Next named levers unchanged: collect
BW push (async-bulk/TMA) toward the 2.47 cap; K2 empty launch; BS16-64
reducer exposure.

## iter 13 — 2026-07-23 — TA falsified, __ldcs GO (big-N collect BW push)
FALSIFIED: (cp.async per-thread double-buffer collect; domain: DRAM-band
npad>=262144; evidence: paired event A/B ab_ta_iter13.log) — 0.93-0.98 on
23/24 cells, 1 wash. smem round-trip + 5->4 occupancy (extra 16KB) beat the
latency-hiding gain; the 25-reg/5-occ collect is already near the practical
streaming ceiling.
iter13b GO: __ldcs evict-first streaming loads on the TA path (ILP-4), zero
occupancy/register cost. Paired A/B: +2-9.5% on 22/24 big-N cells (peaks
flash/pro 1024k BS256 1.095), wash 2. Auto gate npad>=262144 (ARM39_TA env
override). Gate 0/225 + adversarial OK.
e6 envelope: combined gm 1.3179 / mean 1.3564 / min 0.7665, 0/750 inexact,
wins 53 (e5: 1.3150/1.3532/52). Big-N >=256k BS>=256 arm +2.0% gm vs e5;
flash_1024k BS1024 2.12->2.24. Residual to the 2.47 cap now mostly the cap's
own optimism (assumed 7 TB/s; pr itself achieves ~5.8): collect ldcs puts the
arm at parity-or-better streaming efficiency vs pr.
Named next: K2 empty-launch (1.3us x all cells), BS16-64 reducer exposure.
