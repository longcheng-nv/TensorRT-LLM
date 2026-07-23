# op39 RESUME (updated 2026-07-23, iter2 done, nsys screen in flight)

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

## Next (iter5 relay — production arm v2/v3c in src/arm_v2)
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
