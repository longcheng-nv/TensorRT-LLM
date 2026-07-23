# op41 verdict (SUPERSEDED then SHIPPED — see bottom)
(2026-07-23, umb-b200-045)

> **UPDATE (same day): the saturation verdict below held only for the
> replicated-row envelope. User-directed BS>1 heterogeneous-row verification
> REOPENED the campaign (8.4% rows need extra secant passes; straggler tax
> up to 1.35x) and it closed with a SHIP — v3mt per-K rung fractions.
> Full chain in ITERATIONS.md phases 2-3; ship verdict at the end of this
> file.**

## Brief (user, 2026-07-23)
Keep the GVR skeleton (preIdx heuristics -> multi-threshold estimate ->
iterate high-quality thresholds -> exact refine); equivalent per-phase
algorithm changes allowed. Continue optimizing the op39 envelope
(75 cells x BS2-1024 vs bs_real_layers pr anchors; combined record e6
gm 1.3179 / mean 1.3564, 0/750 inexact).

## VERDICT: the brief's lever is already fully banked by the current
## combined dispatch — measured, not argued.

### Evidence chain
1. **preIdx is highly informative** (results/hint_study.csv): the (h*K)-th
   largest hint value is a near-perfect v_K estimator (count 506-512 vs
   K=512 on ALL 75 cells). A 6-rung hint ORDER-STATISTIC ladder
   {K/8,K/4,K/2,3K/4,7K/8,K} gives one-pass count-in-[K,8K] coverage
   **75/75** (74/75 at kC=4K). No FIXED rank is viable alone (best 54/75;
   h spans 0.057-0.998) — iteration/laddering is mandatory, as the brief says.
2. **v3 (the combined dispatch's champion on 697/750 cells) already
   implements exactly this design**: P1 = hint-value-CCDF rung ladder
   (AR4/6/8 quantile rungs + above-hmax rung + hmin floor), P2 = ONE
   multi-threshold count pass + exponential-CCDF secant iterate,
   P4 = exact refine.
3. **Measured saturation** (results/v3_passes.csv, DBG_PASSES build):
   stock v3 converges with **0 extra secant passes and 0 exact-descent
   falls on 45/45 hint-path envelope cells** (npad > 12288; the other 30
   cells take the hintless direct path, which is launch-latency-bound).
   A better ladder can save literally nothing here.
4. **The op39 arm's win band cannot adopt count-feedback iteration**:
   big-N cells are DRAM 1-pass bound — any iterate = a second full row
   pass; hint-rank-as-primary without feedback has unbounded count
   variance in h (measured span 548 -> 189051 at fixed rank), i.e. the
   falsified min-hint-blowup domain. The arm's clustered row-sampling
   (targets ~2K candidates INDEPENDENT of h) is load-bearing; ledger lines
   (r>=64 sampling rank, min-hint-only) stand.

### What this means
- Sub-L2 band: hint-ladder + exact-count iterate == v3's existing design,
  already the dispatch winner there. Saturated.
- DRAM band: op39 arm's sampling+rescue is the 1-pass-discipline equivalent
  of "iterate", already at streaming parity with pr (op39 iter13).
- Further gains on this envelope require a NEW algorithm family (op39
  RESULTS close-out), not better thresholds within the GVR skeleton.

### Assets for future campaigns
- scripts/hint_study.py — offline hint order-stat vs v_K analysis.
- scripts/v3_pass_probe.py + src/v3dbg (DBG_PASSES) — secant pass-count
  instrumentation for any r3_v11-family kernel.
- The 75/75 one-pass ladder result: if a future workload shifts h or the
  envelope (e.g. much lower hit rates, live BS>1 with DIVERSE rows where
  per-row convergence variance matters), re-run v3_pass_probe first — a
  nonzero pass histogram would re-open this campaign with a ready design
  (order-stat rungs replacing value-quantile rungs, one-line P1 change).

---

## SHIP verdict (phase 3, @bce921d0b1): v3mt per-K rung fractions

Change (src/v3mt, constants only, zero P2 cost, skeleton untouched):
- AR4 quantile fracs: K2048 -> {55, 88} (gated npad<49152 || >98304;
  measured exception npad~65600), K1024 -> {35, 70}, K512 -> stock {25, 65}.
- AR6 fracs: K2048 -> {25, 50, 75, 92}, else stock {15, 40, 70, 92}.
Basis: per-row P2 convergence is a LAYER property; optimal rung placement
tracks K (model-family hit-rate distribution). No global set dominates
(falsified). Dispatch-exact simulation -> focused wall-clock scans.

Verdict (ab_v3mt_v2, paired event axis, stock v3 baseline):
- Exactness: 0 fails (75 cells x BS{2,16,256} replicated + hetero per-row).
- Hetero-batch axis (production-realistic): 32/32 cells >= 1.00 —
  v32_256k 1.10-1.18, v32_16k 1.10-1.16, pro_512k 1.13-1.15,
  pro_256k 1.18 @BS256, pro_1024k 1.07 @BS1024, flash ~1.00 (its lone
  straggler L10 is a kC-window straddle — no ladder fixes it).
- Replicated axis: no cell < 0.97; positive outliers to 1.20 (pro_64k).

Follow-up options: swap v3mt into the op39 combined dispatch + envelope
re-run; port the per-K fracs upstream (production GVR uses the same
skeleton and serves hetero batches ALWAYS).
