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

## iter 11 — 2026-07-13 — ATTRIBUTED (K-cost = candidate-band inst in issue-bound pass)
Hypothesis (ledger check: iter10 entry cites spill/resolve suspects): the K2048
+188us is universal spillA r/w (cand 2*K > capA) or resolve/tie scaling.
Probe: rung-1 host replay (scripts/replay_hbe_cand.py, 18 cells r/b/w x K x N)
+ L3 NCU 2x2+2 grid (scripts/ncu_attrib_iter11.sh + GVR29_FORCE_HBE=1 bypass
for the guard-excluded K2048; results/iter11/).
Result — BOTH iter10 suspects FALSIFIED, real mechanism found:
- Replay: tier A 18/18, one-sided 18/18; spill ovA max 251 entries (0.57% of a
  pass), ovB never read (tier B never fires). Spill is NOT the cost.
- NCU (262144, BS=1024, real): dram_r 1.088/1.089/1.089 GB (1.01 pass) at
  K512/1024/2048-forced — DRAM goal holds at ALL K. occ 93-94% flat. dur
  423.6/601.2/765.4us, inst 227.9/326.9/427.2M, issue 81-84% (issue-bound).
- Cross-check vs replay cand counts: +16.3-16.4 inst PER candidate-band
  element at both K deltas — every element in [vB,vA) or >=vA costs ~16 inst
  (single-address atomicAdd + bounds + store); band is ~8*K/row BY DESIGN
  (B insurance col ~6*K + A col ~2*K) => K-proportional issue-cost wall.
- BONUS FINDING (needs L2): K1024 is INSIDE the shipped guard but was never
  pilot-measured (KS=[512,2048]); NCU-axis dur 601us vs stock ~460-490us
  suggests HBE may LOSE at K1024 N=262144. nsys verdict required in iter12.
Diagnosis: fused pass is issue-bound; the tier-B safety column (rank 8*rS_K)
buys nothing observed (0/18 fires) and costs ~6*K*16 inst/row.
Ledger write-back: FALSIFIED.md iter10 K2048 entry re-attributed (was
"measurement gap"); fix candidates = B-off (1 cmp/elem, single
find_threshold) or B-narrow (rk_b 8->3-4*rS_K, no global spillB).
Next: iter12 = kColB compile-key A/B (B-on vs B-off), gate, nsys pilot with
K1024 added to KS (close the guard hole) + K2048 forced.

## iter 12 — 2026-07-13 — SHIP (B-off default; guard widened to N>=65536, all K)
Hypothesis (ledger check: iter11 tier-B entry cites this as the fix; no
red-line hit — removes a threshold, does not add one): dropping the tier-B
insurance column (kColB=false: 1 cmp/elem fused pass, ONE find_threshold,
tier-A-or-stock-fallback) recovers the K-proportional issue-cost and the
short-row fixed overheads.
Probe: rungs merged into implementation (mechanism already silicon-attributed
in iter11); kColB compile-key, col_b host flag, spill/dyn-smem col_b-aware.
Result:
- Gate 324/324 (3 scen x 3 K x 4 N x 3 BS x [hbe-B, hbe-noB, off], forced
  engagement everywhere incl N=32768).
- nsys same-batch 4-arm pilot, 27 cells x 3 scenarios (GVR29_FORCE_HBE=1)
  + no-force real confirmation (results/pilot/iter12/):
  nob vs rival: 262144x1024 1.71-1.73 (K512/K1024) / 1.62-1.64 (K2048);
  262144x2048 1.60-1.75; 131072x1024 1.46/1.48/1.33-1.36; 65536 cells ALL
  positive 1.03-1.13; 32768 still 0.85-1.00 (stays outside guard).
  vs B-on arm: e.g. 131072x1024 K1024 0.84 -> 1.48; K2048 262144x1024
  0.87 -> 1.64. Scenario-invariant (real/best/worst within ~1%).
- K1024 GUARD HOLE (iter11 bonus finding) CONFIRMED on nsys and FIXED:
  B-on K1024 was 0.84 (131072x1024) / 1.08 (262144x1024) INSIDE the shipped
  guard; nob 1.48 / 1.72.
- No-force run: N=32768 hbe/nob/off all 0.99-1.02 (guard exclusion clean);
  engaged cells match the forced run.
- RESUME step 2 (131072 residual) resolved as a side effect: 1.09 -> 1.46.
- Fork parity intact (off/rival 0.99-1.02 median ~1.00).
Diagnosis: candidate-band width was the K-cost AND most of the short-row
fixed cost; A column alone (2*rS_K target) keeps one-sided safety via the
count-validity check, miss falls to stock fallback (never observed in gate).
Ship shape: guard = !cluster && streaming && N>=65536 (2 rules); col_b=False
default; kColB=true retained for A/B; spill buffer halved (spillA only).
Ledger write-back: N<=65536 entry re-scoped to N<=32768 (65536 REVIVED);
tier-B entry marked RESOLVED-SHIPPED; K2048 entry closed by iter12.
Next: cluster-path HBE (BS<=512); short rows N<=16K (P5); full-grid sweep +
REPORT arm; production dispatch-tier decision (user).
