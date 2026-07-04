# op13_gvr_p2cand — Learnings

## Architecture / measurement facts
- Base = plain **snap-P4** `GvrTopKKernel` (NOT op#7 rank-scatter). Phases:
  P1 preIdx-stats → P2 secant (full-N count_ge ×evals) → P3 collect (v≥thr →
  smem_keys[fp32]) → P4 histogram+snap over cand candidates.
- `self.kC` (= kCC, candidate cap AND P2 acceptance upper bound), `self.kFTarget`,
  `self.kNumBins` are plain ctor attrs from `GvrParams.get(dtype,K,cr)`, read as
  const_expr at compile → **overridable in a subclass before `cute.compile`**
  (see `scripts/phase_ab.py::GvrTimedOverride`). No need to edit the vendored file.
- Host replay `src/p2_replay.py` is faithful (720/720 vs kernel baseline) → use it
  for cheap param search; only port to a real kernel once host predicts a win.
- Exactness criterion (value-equiv to torch.topk): `K ≤ count_ge(thr_final) ≤ kCC`.
  Below K → misses top-K; above kCC → cap truncation drops top-K. So
  "no-fallback + cand∈[K,kCC]" IS the exactness guard.

## DECISIVE finding (iter 2) — corrects op12
- **snap-P4 cost scales with candidate count** (P4 9.3→5.7µs @N4K, 10.4→0.2µs
  @N65K when cand 4.05×K→1.04×K). op12's "P4 is barrier/latency-floor bound, NOT
  candidate-count bound" was measured on the **rank-scatter** P4 op#7 — it does
  NOT transfer to the snap kernel. The user's premise (cut cand → cut P4) is
  TRUE for the snap base op.
- **BUT the only candidate-reduction lever that works (narrow kCC) costs P2 full-N
  scans that ≈ cancel the P3+P4 savings** → ΔTOT≈0 (one +2µs win at N=65536).
  The P2 secant eval is the tax; the P4 saving is real but pre-paid.

## Effective so far  (iter 3-5, nsys pure-kernel)
- **Moderate cand target (kCC≈2–3×K) with EVAL-OPTIMAL kFTarget = +1 eval (not
  +3)** — host pre-pass (`kcc_host_prepass.py`) picks the kFTarget minimizing P2
  evals per kCC. This is the cheap-reduction path.
- **K=512 fp32: net ~10% pure-kernel win at N≤16K + 65K, EXACT, no fallback**
  (nsys, launch floor stripped). The candidate-reduction premise IS realizable on
  the snap kernel in the small/mid-N regime.
- **N-dispatch mandatory**: large N≥131K loses (P2-eval explosion). Narrow kCC
  only for N≤~65K.

## Measurement methodology (critical)
- **Event-timed cold-L2 wall is USELESS at small N here**: ~16µs CUDA-graph launch
  floor swamps the ~1.5µs kernel delta; medians quantize to ~1.024µs multiples.
  MUST use nsys pure-kernel (nvtx_kern_sum) for small/mid-N kernel deltas. Event
  timing is fine only for the large-N losses (deltas ≫ floor).
- **phase_ab.py absolute µs is production_wall × modified_fractions** — only the
  FRACTION split is valid for a modified kernel; never read its ΔTOT (this voided
  iter-2's "wash"). Use `kcc_walltime_ab.py` (times the real modified kernel) or
  nsys for absolute A/B.
- Single-batch nsys has ≥0.5µs run-to-run variance → median of ≥3 batches for a
  ship claim.

## SHIP outcome (iter 7) — K512 fp32 ONLY
- Built `src/gvr_p2c_op.py` (mirrors gvr_cutedsl_op.py; `GvrP2C` subclass override,
  no vendored edit). **480/480 exactness** (value-equiv + done==1, no fallback).
- nsys ×3-median **p2c op vs baseline op**: K512 fp32 robust WIN (N=4K −14.7%, 8K/16K
  −11%, 65K −5.5%, 32K ~tie); N≥131072 ties baseline EXACTLY (dispatch→baseline) so
  zero large-N regression. Crossover pinned in (65536,131072].
- **K1024 fp32 NOT shipped**: ×3-median noisy with REAL regressions (N=4096 +15.8%,
  N=65536 +12.4%) despite mid-N wins. A config with losses fails the falsification
  bar → demoted to baseline. Lesson: single-batch K1024 nsys is too noisy to trust;
  the ×3-median is what exposed the regressions a single batch hid.
- Final: `_NARROW_TABLE={(fp32,512):(kCC=1536,kFTarget=1280)}`, NARROW_N_MAX=65536.

## Win is dtype/K-specific (physics)
- Win = (P4 fraction × cand-cut) − (P2-eval tax). Maximized for **fp32** (P4
  expensive) × **K=512** (4×K over-collect) × **small/mid N** (P4-dominant, cheap
  evals). **bf16 only ~3%** (ties collapse → P4 already cheap). K=2048 baseline
  already lean. P3 is full-N-READ-bound so it never shrinks much → no "drastic" cut.

## Ineffective / dead ends
- **Init-only threshold (lerp/pquantile)** for K512/K1024: backfires in the wide
  default window (overshoot below K → cand goes UP to ~5×K). Only useful combined
  with a narrowed window, where it does not reduce eval count.
- **Aggressive kCC=1.25×K**: cand→1.04×K but +3 P2 evals → net wash (iter 2).
- (Inherited from op12, on rank-scatter P4 — re-confirm if reused: opt-1 fp16
  traffic no-op; rs non-exact fails on continuous Beta.)

## Open hypotheses (iter 3)
- H1: kCC≈2×K + N-dispatch (cand-cut only at N≤~16-32K) → net-positive small/mid N,
  baseline large N. Measure ΔTOT per N with `phase_ab.py` kCC sweep.
- H2: cheaper P2 root-finder (regula-falsi/Illinois or slope-model) hits the
  narrowed window in ≤3 evals instead of 5.4 → converts the wash into a win at
  fixed cand. Prototype in host replay first.
- H3: large N is P2-dominated (eval ~8-12µs each) → cand reduction can NEVER pay
  there; the achievable win is a small/mid-N-only feature.

## Floor caveat
- At small N the total is ~16µs of which ~2.5µs P1 + a shared CUDA-graph launch
  floor are fixed. Even a free P4 collapse caps the small-N gain. Track whether
  the moderate-target win survives nsys pure-kernel (event includes ~launch).
