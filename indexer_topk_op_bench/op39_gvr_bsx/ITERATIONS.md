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
