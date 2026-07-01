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

## iter 6 CLOSE (nsys, both K measured): Scheme X NO-SHIP
- **K2048 fp32**: X/rs 0.95–1.02× (net-neutral). **K512 fp32**: X/rs 0.86–0.98×
  (net LOSS at EVERY N). K512 is worse because band is large (M0≈1.8–2.6K,
  M(free)≈360 ⇒ band ~1.4–2.3K, only ~15% below baseline M0) so the rank-scatter
  floor barely collapses, while phase4_partition (M0-wide 2-pass + 2-counter smem
  atomics) is pure added cost — largest fraction at small/mid N (0.86–0.91 @ 4K–32K).
- Scheme X is EXACT + baseline byte-identical (flag off); a net loss with flag on.
  **HEAD stays at baseline. op16 closed.** Sole untried lever = warp-aggregated
  partition, ceiling K2048-large-N only, cannot fix K512 → not worth pursuing.
  Real lever remains op13's cheaper-P2 (separate ticket).

## Resume gotcha: stale torch cpp_extension baton lock hangs imports forever
On resume after a host expired mid-run, `nsys_ab.py` hung ~22 min with CPU frozen
at 10s (blocked, not compiling), header never printed, jsonl empty.
- **Diagnosis**: `py-spy dump --pid <PID>` → `wait (torch/utils/file_baton.py:51)`
  ← `_jit_compile` ← `radix_cuda_op._module()` ← `import sweep`. The prior host
  died mid-JIT and left a 0-byte `lock` baton in the torch-extension build dir.
  `file_baton.wait()` polls for that lock to vanish → infinite wait (the `.so` was
  already fully built, so no compile was even needed).
- **Fix**: remove `indexer_topk_op_bench/_build/radix_cuda/lock` (build_directory
  set in `harness/radix_cuda_op.py`), then re-run — imports drop from ∞ to ~6s
  ("ninja: no work to do"). Confirm with `ps -o etimes,time` (elapsed climbs, CPU
  frozen ⇒ hang, not slow-compile) then py-spy for the exact wait site.

## Method notes
- clock64 tot ≈ cold wall at single-CTA sizes (18464 cyc ≈ 10µs ≈ 9.7µs sweep) →
  cycle deltas are trustworthy for floor-vs-tax without wall-us conversion.
- `scripts/p4_scaling.py`: kC/kFTarget override subclass of the timed rs kernel;
  the clean way to isolate P4-vs-cand and the P2 tax without editing the kernel.
- eval-optimal kFTarget (near kCC) matters: turns op13's "loss" into "wash" at
  small N. Never judge the tax with kFTarget=K.
