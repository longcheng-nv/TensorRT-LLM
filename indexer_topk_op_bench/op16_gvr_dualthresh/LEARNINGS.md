# op16_gvr_dualthresh — Learnings

## Verdict (iter 0, B300, measured): two-threshold band-refine is TAX-BOUND
The user's mechanism (find `threshold_1` with count `M<K` in P2 via secant, so P4
only refines the band `[threshold, threshold_1)`) cannot beat the baseline in the
DSv4 decode regime, for the same root cause as op12/op13/op14.

## Why (the algebra + the measurement)
- **Band = cand − c_hi, and c_hi < K ⇒ band > cand − K.** To make band small
  enough for a cheap single-warp select (≤256), cand must be ≈K, i.e. the
  threshold must be pinned tight (kC≈K).
- **Pinning a tight threshold explodes P2** (full-N secant `count_ge` passes):
  measured 1.3–2.5× P2 cycles on B300 (K512 fp32). With eval-optimal kFTarget the
  P2 tax ≈ the P4 saving at small N (WASH) and EXCEEDS it at N≥65536 (LOSS).
- **rank-scatter P4 is mostly floor-bound**: only ~15% count-reducible at small N;
  collapses to ~0 only when cand→K (which needs the max tax). Confirms op12.

## Ceiling (even if band-select P4 were FREE)
−27% @N=4096, −11% @16384, ~0 @65536, LOSS @≥131072. Mild mid-N only.

## The +40%-over-Radix-at-95%-of-cases target is structurally unreachable
Independent of any P4 work:
- **Small N**: Radix floor (5–6.5µs) is BELOW GVR's kernel floor (~8–9µs, shared
  by every GVR variant). Best-case −27% still loses to Radix.
- **Large N (131K/262K)**: GVR is P2+P3 full-N-streaming bound (60–70%), gvr=35µs
  vs Radix flat 19µs @262K. A threshold-streaming kernel cannot beat flat
  radix-select there even with P4→0. (~26/60 cases.)
- GVR only wins Radix in the mid-N crossover (16K–32K); currently 6/60 cases beat
  BOTH baselines.

## The ONLY real lever (op13's unbuilt H2): cheaper P2
Reduce the number of full-N `count_ge` passes to pin a tight threshold:
sampling-based quantile init (read a subsample, estimate the K-th value), a
CDF-aware / higher-order root-finder, or a much better preIdx-seeded t0. This is
the only thing that:
  (a) makes cand→K cheap → unlocks the P4 collapse (small/mid-N win), AND
  (b) directly cuts the large-N P2 cost that makes GVR lose to Radix.
It subsumes the two-threshold goal. Uncertain, larger effort.

## Method notes
- clock64 tot ≈ cold wall at single-CTA sizes (18464 cyc ≈ 10µs ≈ 9.7µs sweep) →
  cycle deltas are trustworthy for floor-vs-tax without wall-us conversion.
- `scripts/p4_scaling.py`: kC/kFTarget override subclass of the timed rs kernel;
  the clean way to isolate P4-vs-cand and the P2 tax without editing the kernel.
- eval-optimal kFTarget (near kCC) matters: turns op13's "loss" into "wash" at
  small N. Never judge the tax with kFTarget=K.
