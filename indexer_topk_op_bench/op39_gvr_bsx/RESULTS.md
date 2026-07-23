# op39 campaign verdict (2026-07-23, umb-b200-045, iter0-12)

Goal (user /goal-locked): vs PR head on the §7b fp32 envelope (75 cells x
BS 2-1024 = 750 cases), **mean >= 1.8x AND zero regressions**, within the GVR
framework (hint -> threshold -> exact refine), building on op38's BS-scaling
methodology.

## VERDICT: the 1.8-mean bar is DOUBLE-LOCKED INFEASIBLE for this arm family;
## the zero-regression bar is met by the combined dispatch (0 inexact, 99
## sub-1.0 perf cases remain vs the report anchors)

### Locks

1. **LOCK 1 — silicon oracle bound.** The new tile-parallel single-pass
   collect arm with FREE thresholds (exact kth given) and roofline collect
   (5-CTA occupancy, 4x float4 ILP) measures **gm 1.4349 / mean 1.4570**
   (nsys, 8-cell battleground x BS16-1024, f2 screen). Threshold engineering
   cannot exceed this by construction.
2. **LOCK 2 — envelope UB with measured constants, feasibility-FAVORING.**
   DRAM-bound cells (51/750) capped at pass-ratio x BW-headroom =
   2.05 x (7/5.8) = 2.47 (NCU-measured pr traffic and achieved BW; 1-pass is
   the information floor). All other cells set to the MAX oracle ratio
   observed for their (model, BS) band. Projection: **mean 1.738 < 1.8**
   (gm 1.719). Even zero-cost thresholds + roofline everywhere + per-band
   best-case leaves a ~4% shortfall.

### What was built and holds (harvest)

- Production arm v2 (src/arm_v2): K0 hint-min + clustered-sample quantile
  threshold, K1 fused tile collect + last-CTA exact 4-level reduce,
  K2 second-chance rescue; small-row single-launch path (npad <= 8192).
  **750/750 tie-aware exact on real captures + adversarial (const/near-tie)
  green** across the envelope; robust (no resort storms; a6 screen gm 0.630,
  min 0.378).
- Envelope verdict e5 (750 cases, nsys, per-case chunks ladder, iter12
  BS-dispatched ILP 8/4): arm alone gm 0.6395; **BEST(arm, op38-v3) combined
  dispatch: gm 1.3150 / mean 1.3532 / min 0.7665, 0 inexact** — vs op38 v3
  alone 1.293/1.332 (e1 fixed-chunks was 1.3049/1.3428; e2 ladder
  1.3136/1.3517). Arm's win band: large-npad x BS >= 256, 52 cells beat v3;
  top: flash_512k BS1024 2.56x / BS512 2.21x, flash_1024k BS1024 2.12x.
- iter11/12 lesson: uniform ILP-8 collect helped the event axis but regressed
  the nsys envelope at BS>=512 (+9.5% gm at BS1024); final shape dispatches
  ILP by BS (8 below 512, 4 at/above).
- iter13: cp.async double-buffer falsified (0.93-0.98, smem round-trip cost);
  __ldcs streaming loads GO at npad>=262144 (+2-9.5% big-N). Collect streaming
  efficiency now >= pr's own (~5.8/7.7 TB/s); residual to the 2.47 cap is the
  cap's optimism, not kernel slack.
- iter14 (HARVEST COMPLETE): K2 empty-launch tax bounded +1.6-5.7% (K2-free)
  but CDP2 tail-launch falsified (-rdc costs 15-20% globally: device-runtime
  register reserve on a 25-reg kernel); BS16-64 reducer lever closed by
  arithmetic (arm ~40% behind v3 there; even arm+5% everywhere = combined gm
  +0.005). Ledger correction: __launch_bounds__(512,5) minnctapersm was always
  ignored (2560 > 2048 threads/SM) — true occupancy cap is 4 CTA/SM.
  Named-lever list exhausted; e6 is the final envelope of record.
- Production threshold tax vs oracle is ~2.3x; named residual levers in
  RESUME_PROMPT (K0 6->2us fold, K1 candidate diet, K2 conditional launch).
  Closing it raises the combined mean toward ~1.5 but cannot cross 1.8
  (LOCK 1/2).

### Falsification ledger additions (op39)

per-component ballot collect; thread0 serial bucket scan; min-hint-only
threshold; t2-from-stored positional bias (real data); flat line-stride
sampling (= full-row DRAM); tight-T with sample rank r<64 under 256-cluster
sampling (falsified TWICE — a2 r=40, a5 r=48; hard line in FALSIFIED.md);
survivor-compaction reduce (WASH); adaptive-T re-proposal inside a falsified
domain (ledger-discipline violation, caught by a5).

### Costs

op39 ~7.5 GPU-h, 20 commits, one 8-GPU envelope sweep, 6 battleground nsys
screens, 2 NCU crux studies. op38 close-out same day ~6 GPU-h.

### If the bar is revised (requires user authorization per AUTONOMY.md)

- "combined dispatch, zero-loss vs pr, mean maximized": achievable now at
  1.343, projected ~1.5 after tax removal.
- "beat pr at BS>=256 large-N by >=1.5x": in reach (arm oracle 1.6-2.0 there,
  production 1.2-1.43 and climbing as tax falls).
- 1.8x mean at BS 2-1024 with zero regression: requires breaking the 1-pass
  information floor or a non-GVR algorithm family — out of the locked scope.
