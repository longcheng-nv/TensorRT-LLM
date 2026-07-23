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

## Next
1. Parse f1 nsys verdict: small-cell BS16-256 truth; identify residual losses.
2. Production arm: threshold-from-hint (preIdx gather -> conservative t_lo),
   undershoot fallback (count<K -> re-scan with lowered t or escape to v3
   dispatch), tie-exact tail (reuse GVR P4 machinery on candidate set; the
   mb reduce's bucket-tie fill is NOT tie-exact yet).
3. Wire as arm into op38 kernel_bs dispatch (BS>=16 keys only), full-envelope
   sharded nsys sweep (drive_v2_sweep.sh pattern) + verdict.py.

## Gotchas
- exact gate must be tie-aware multiset (probe.exact_rows) + real captures
  (bundle loaders) + adversarial ties; oracle-thr mb skips threshold estimation.
- CAP=8192, STAGE=6144 (48KB smem, 3 CTA/SM at 512thr) — v32 K=2048 rows with
  overshoot thresholds can exceed STAGE per chunk at chunks=1 (cnt clamps, but
  production must handle overflow: raise threshold or spill).
- Event-axis vs nsys-pr comparisons are pessimistic by ~5-8us (1 launch).
