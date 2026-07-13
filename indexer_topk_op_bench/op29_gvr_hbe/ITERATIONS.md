# op29 ITERATIONS

## iter 1 — 2026-07-13 — GO (crux, rung 0)
Hypothesis (ledger check: none hit — not Opt-L/op15/P1-model-seed; uses hint
directly + plain cursors): the preIdx hint can place a speculative collect
threshold bin b_hat <= b* (one-sided!), enabling a fused 1-pass
collect+inline-histogram vs sglang_v2's 2 passes; miss -> redo pass = rival
parity.
Probe: scripts/crux_hint_bin.py + crux_hint_quantile.py on all 27 op22rr fp32
bundles (GPU3).
Result:
- min(hint) predictor DEAD: one deep miss drags b_hat ~2000 bins low ->
  cand up to 192xK (useless).
- Success condition is ONE-SIDED (b_hat <= b* suffices; exact b* recovered
  from the inline histogram; resolution happens inside the candidate buffer).
- ORACLE cand(b*) = 1.01xK med / 1.06xK max -> 12-bit bins are razor sharp;
  buffer size is the entire game.
- Quantile predictor q=0.9: 100% one-sided success on ALL scenarios;
  cand/K med 2.7 (real) / 3.6 (best) / 6.5 (worst), p90 ~29, max ~116
  (K2048 large-N: hint quantiles land ~2000 bins low on V3.2 marginals).
- q=0.75: cand med 1.7-3.5xK, success 75-80% real / 100% best,worst.
Design implication: dual column (A=q0.75 tight, B=q0.9 safe) + per-column
smem caps ~2-4xK + flashinfer-style global overflow for B; K2048 large-N
falls back to 2-pass (rival parity; shape-keyed dispatch, allowed).
E[DRAM passes]: real/best ~1.0-1.3, worst ~1.0-1.4 vs rival's fixed 2.0.
Verdict: GO -> iter0 fork, then iter2 implement fast path.
Next: iter0.

## iter 0 — 2026-07-13 — (in progress)
Fork vendored sglang_v2 -> src/gvr29 (baseline immutable; new op files).
Gate + L1 parity vs sglang_v2 arm expected +-3%.
