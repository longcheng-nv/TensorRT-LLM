# hit rate vs GVR gain — op36 grid statistics + mechanism (2026-07-18)

Script: `hit_gain_analysis.py` (per-batch table + within-model N-detrended
residual correlation). Controlled-experiment context: op24 RESULTS.md
(fixed-shape hr sweep, HLS/op21 family) + op30 calibration (GVR-base
absolute extremes).

## Findings on the op36 real §7b grid (25 batches, 275 cells)

1. Raw split (§6c): pr/sgl gm 0.684 @hit>0.4 vs 0.783 @hit<=0.4 — an
   APPARENT negative hit effect.
2. N-detrended within model, corr(residual, hit) on log(pr/sgl):
   flash +0.33 / pro ~0 / v32 -0.34..-0.46 (all-BS / valley); effect per
   +0.1 hit within ±2%. **Once N is controlled, hit explains almost
   nothing on this grid — the §6c raw split is an N-structure confound**
   (high-hit batches skew small-N = the 4-16k hole).
3. The only sign-consistent real-data trace of the known high-hr penalty:
   the three hr≥0.84 batches (v32/8k 0.842, v32/256k 0.865, pro/4k 0.998)
   all sit below their model's N-trend (v32/256k g_all 0.904 < v32/128k
   0.995 despite larger N) — direction matches op30, magnitude small
   next to the N effect.

## Mechanism (from the fixed-shape sweeps; real grid barely samples the poles)

GVR seeds its rung thresholds from prev-step top-K values (P1 gather at
preIdx, P1b histogram). sglang_v2/radix are hit-agnostic — so the hit
dependence of the GAP is entirely GVR-side, and it is an **inverted-U**:

- hr≈0.05 (adversarial): gathered values sit below the current top-K band →
  every rung admits count>K (all_ge) → per-row fallback / extra full-row
  passes. op24: o/s 0.76-0.86, radix ~2.5x faster; K-flat.
- hr≈0.55 (sweet spot): several ladder rungs land inside the fast-path
  acceptance window f∈[h, 2.5h] → single-pass admission. op24: best combos
  gm 1.75-2.14, beats radix at 65-262K.
- hr≥0.85 (worst for GVR-base): thresholds land too tight → admission
  undershoots (<K; the known real-data failure mode final_count<kK) +
  init poisoning (op22 pair01/band_gt_kC) → extra refine evals. op30
  calibration: GVR-base worst at hr 0.85-0.90 (worst/best 1.5-2.2x per K);
  on that data sglang_v2 is 2.25x vs GVR-base (vs 1.44x on low-hr data).

Note op30 (base absolute: "low-hr fast / high-hr slow", worst 0.85-0.90)
and op24 (op21-family relative: worst at 0.05) measure different arms —
combined they give the inverted-U for the shipped R0-ladder GVR.

## Why the ship table neutralizes this

Rule 3 routes small-N and extreme-BS to sgl_bx (hit-agnostic); hit
sensitivity survives only on the 26 pr/a2-routed cells. §6c: ship composite
≥1.0 in BOTH hit domains (1.011 / 1.025). Dispatching on hit remains a red
line (unknowable at inference) — and this analysis shows it is also
unnecessary on the deployment axis.
