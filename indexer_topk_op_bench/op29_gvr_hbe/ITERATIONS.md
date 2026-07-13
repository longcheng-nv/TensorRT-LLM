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

## iter 0 — 2026-07-13 — SHIP (fork parity)
Fork gvr29 (src/gvr29) builds; gvr29_off/rival same-batch nsys = 0.993-1.032
(54 cells, geomean ~1.000) — fork and harness trusted.

## iter 3 — 2026-07-13 — PIVOT (pilot verdict)
Probe: nsys same-batch 3 arms x 54 cells (results/pilot/). GO/NO-GO mixed:
- WIN pocket: K512 N>=131072 BS>=1024: rival/hbe 1.17-1.47x (real 1.34/1.45,
  best 1.17/1.47/1.35) — the fused-pass DRAM saving is real at bandwidth bound.
- LOSS everywhere else. Diagnosis (mechanism):
  (1) L2-trap (Phase 1.4 veto, should have pre-computed): BS*N*4B <~ L2
      (126MB) => rival's 2nd pass is L2-hot, 1-pass saves no DRAM; fused-pass
      smem-atomic tax + hint phase then lose outright (K512 32768 cells 0.55-0.81).
  (2) capA=2xK too small for worst (crux cand med 3.4xK) -> tier miss ->
      redo = 2 passes + overhead (worst 131K/262K 0.85 instead of winning).
  (3) hint gather K*BS random reads = fixed tax (65536x2048: 1M gathers, 0.91).
  (4) K2048 hint quantiles unreliable (crux) -> always-miss ~0.5x. Dispatch off.
Ledger write-back: FALSIFIED.md += (HBE fused pass at BS*N*4B < ~1.5x L2,
{fp32, streaming}, nsys) structural-wall L2-trap.
Next: iter4 = capA 4xK (K<=1024), hint 4x subsample, dispatch guard
(K<=1024 && N>=131072); re-pilot.

## iter 5 — 2026-07-13 — WASH (global spill; worst 262144 0.80->0.88-0.96, not enough)
## iter 6 — 2026-07-13 — FALSIFIED (hint x sample max() breaks one-sided safety; 131K 0.70-0.74)
## iter 7 — 2026-07-13 — FALSIFIED-BY-SILICON (cand-targeted cols PERFECT on host
   replay [binA<=b*, candA 1.1-2.6xK] yet slow: strided sample gather = DRAM-burst
   waste ~half a pass at BS=1024)
## iter 8 — 2026-07-13 — PARTIAL (chunked coalesced sampling: 262144 positive
   1.02-1.14, 131072 still 0.88-0.91)
## iter 9 — 2026-07-13 — SHIP-CANDIDATE (breakthrough)
Hypothesis: NCU attribution — fused pass read 545MB (1.06 passes, DRAM goal met)
but 378us vs rival 245us => ISSUE-bound (1.4TB/s): the inline full histogram
(F2F+twiddle+smem atomic per element) is the bottleneck, and it is UNNECESSARY:
validity = cnt(>=vA) >= K (count proves top-K containment); b* recoverable from
a candidate-only mini-hist in smem at resolve.
Result: nsys same-batch, all 3 scenarios: 262144x1024 1.470-1.498, 262144x2048
1.364-1.398, 131072x1024 1.089-1.119; non-engaged parity 0.997-1.003; gate
216/216. SCENARIO-INVARIANT (hint-free sample estimator).
Diagnosis: fused pass now ~2 cmps/element (lighter than rival's hist pass);
131072 residual gap = fixed per-CTA overheads (sample + 2x find_threshold +
resolve) vs shorter main pass.
Ledger: WALLS.md += inline full-row histogram in a fused collect pass is
issue-bound-prohibitive on B200 (per-element smem atomic + F2F); the fix is
count-validity + candidate mini-hist.
Next: iter10 = per-K caps to re-enable K2048; widen guard to
batch*maxseq >= 128M elems (adds 65536x2048); re-pilot; then cluster-path HBE
+ full-grid sweep.

## iter 10 — 2026-07-13 — EXPANSION FALSIFIED (guard reverted)
K512 proven cells unchanged (1.06-1.49). New cells lost: 65536x2048 0.63
(per-CTA fixed overheads vs 65K rows), K2048 all 0.56-0.88 (rival +13us going
K512->K2048 at 262144x1024, HBE +188us — K-proportional cost unattributed;
suspects: cand target 2*K=8192 > capA 4096 => universal spill r/w, resolve
scaling, tie machinery). Guard reverted to K<=1024 && N>=131072.
Ledger: FALSIFIED += (HBE at N<=65536 even when batch*N>=128M, {fp32}, nsys)
— fixed overheads; revival = shrink sample/resolve fixed costs.
Next: NCU-attribute K2048; cluster-path HBE; full-grid sweep + REPORT arm.
