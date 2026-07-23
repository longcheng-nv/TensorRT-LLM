# op39 RESUME (updated 2026-07-23, iter12 done — harvest phase)

## CURRENT STATE (iter12)
Campaign verdict unchanged: 1.8-mean double-locked infeasible (RESULTS.md).
Harvest best = e5: combined gm 1.3150 / mean 1.3532 / min 0.7665, 0/750
inexact, 52 arm-beats-v3 cells. Kernel: arm_kernel<RESCUE, ILP> with BS
dispatch (ILP8 < BS512, ILP4 >= 512) — uniform ILP-8 falsified at BS>=512
(e2-vs-e3). Verdict scripts: verdict_e5.py; data results/e5_data.csv.
Next named levers: collect BW toward 2.47 cap (TMA/async-bulk; NCU 75.4%
DRAM at flash_512k BS1024), K2 empty-launch, BS16-64 reducer exposure.

## Older header (iter2 era, kept for context)

## 1-minute context
User /goal-locked: vs PR head (bs_real_layers.csv nsys anchors), §7b fp32
envelope 75 cells x BS2-1024, **mean >= 1.8x AND all >= 1.0x**, GVR framework.
op38 double-locked the old r3_v11 ladder — op39 is a NEW ARM: fused
tile-parallel single-pass collect (grid (chunks,BS), smem staging, last-CTA-
per-row candidate top-K reduce, self-cleaning counters). Oracle-threshold
microbench GO: pro_1024k BS1024 1.84x event-axis; parity+ at 64k cells.
Small cells BS<=256 pending nsys axis (event launch tax masks truth).

## Preflight
- node umb-b200-045, GPU0 used for probes; git branch omni/op21-gvr-prod
- git log -1 should show "[op39] iter2 GO"; workspace indexer_topk_op_bench/op39_gvr_bsx
- /tmp/gvrlayers/cutlass450 overlay must exist (1-line rebuild in memory
  env-tmp-gvrlayers-wiped) — only needed for PR-head paired probes
- ncu at /opt/nvidia/nsight-compute/2026.1.1/ncu (PATH symlink is ELOOP-broken)

## State
- iter0: cruxes — DRAM-bound lever bounded (2.05x pass ratio, 51/750 cases);
  battleground = L2-resident BS>=16 latency wall (all SOL<40%), 699 cases.
- iter1: smem-staged collect GO; ballot-per-component variant falsified.
- iter2: fused kernel GO (src/mb_fused). chunks ladder per (cell,BS) matters.
- IN FLIGHT: results/nsys/f1 screen (scripts/mb_nsys.py, 8 cells x 7 BS,
  oracle thr) -> parse with scripts/mb_verdict.py.

## Next (iter7 relay — READ THIS FIRST)
Screens: f2 = oracle bound w/ roofline collect gm 1.4349 (results/nsys/f2,
parse: mb_verdict.py --rep .../f2.nsys-rep); a4 = production gm 0.583.
1. FIX K=2048 STAGE regression (a4): K-aware STAGE template (4096/5-occ for
   K<=1024, 6144/3-occ for K=2048) or T=1.5K at K=2048; re-gate + re-screen.
2. Decisive for the 1.8 bar (LOCK 1 in place: oracle gm 1.43 < 1.9):
   (a) multi-row-per-CTA shape for sub-L2 cells (chain-latency cap there),
   (b) TMA/async-bulk collect BW push at big cells,
   (c) if (a)+(b) stall: relaxed-constraint control run -> double-lock verdict,
       then harvest = combined dispatch (op38 v3 + arm win band large-npad
       BS>=256 at 1.16-1.43) and a bounded close-out per AUTONOMY.md.
3. Production threshold tax levers still open: K0 6->2us (fold hint+sample,
   single select), K2 empty-launch 1.3us.

## Older context (iter5 relay — production arm v2/v3c in src/arm_v2)
State: oracle bound (results/f1_verdict.txt) = gm 1.37/min 0.967 nsys over the
8-cell battleground. Production arm (K0 hint-min + strided-sample quantile ->
K1 fused collect+reduce -> K2 rescue) is 0/225 exact incl. adversarial, but
carries a ~2.5x threshold tax on sub-L2 cells: nsys split pro_64k BS16 =
K0 6.1 + K1 17.5 + K2 1.35 vs oracle-fused 9.2 vs pr 10.7.
1. K1 reduce: single-level 2048-bucket (11-bit) select for n~2K candidates
   (current 4-level x 256 costs the exposed low-BS reducers).
2. K0: S=2048 samples, fuse hint+sample phases (one sync chain), skip sample
   when npad <= CAP; target <= 2us.
3. Verify win band with production thresholds: pro_1024k BS256 event 148us vs
   oracle 76us — suspect threshold/rescue behavior on low-hit L46; nsys split.
4. Re-screen: scripts/mb_nsys-style run with arm_v2 (write arm_nsys.py), then
   full-envelope sharded sweep + [worst,real,best] axes + zero-regression
   dispatch vs op38 v3 (arm only where it wins).
Falsified this campaign (do NOT re-propose): min-hint-only threshold (31/75
overflow); t2 = K-th of stored subset (positional bias, real data);
per-component ballot collect (0.2-0.6x); thread0 serial bucket scan.

## Gotchas
- exact gate must be tie-aware multiset (probe.exact_rows) + real captures
  (bundle loaders) + adversarial ties; oracle-thr mb skips threshold estimation.
- CAP=8192, STAGE=6144 (48KB smem, 3 CTA/SM at 512thr) — v32 K=2048 rows with
  overshoot thresholds can exceed STAGE per chunk at chunks=1 (cnt clamps, but
  production must handle overflow: raise threshold or spill).
- Event-axis vs nsys-pr comparisons are pessimistic by ~5-8us (1 launch).
