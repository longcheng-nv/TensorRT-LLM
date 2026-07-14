# op34 ITERATIONS

Verdict vocab: SHIP / FALSIFIED(+domain) / WASH / PIVOT / GO / NO-GO (probe rungs).
Incumbent-to-beat = sglang_v2. Start kernel = op26_r0auto. Ship goal = GRAND BS=1
fp32 geomean <= sglang/1.30 (<=6.06us; need 2.03x over r0auto 12.31us).

## iter0 — 2026-07-14 — CHARACTERIZATION (analysis/, committed)
- Data props: pro hr 0.6-1.0, flash hr 0.3-0.76 (last-step); boundary gap_rel ~1e-5.
- Gap map (BS=1 fp32 cold geomean): sgl 7.88us / r0auto 12.31 / base 18.60; 0/459 win-layers.
- Priors (FALSIFIED.md): pass-count+occupancy wall (op#8), short-N phase-chain floor (op32),
  hint-on-cluster parity-only in-envelope (op31). Target probably structurally hard; needs
  fresh v4cap double-lock (math floor + relaxed-constraint control) before any infeasible ruling.

## iter1 — 2026-07-14 — USER HYPOTHESIS: single-scan bracket + fast-write certain-winners
Hypothesis (ledger check: Opt-L "fuse collect into count ≈ full pass" — CONDITIONAL REVIVAL:
user's variant adds (a) hint-predicted TIGHT bracket, (b) count<K fast-write short-circuit that
SHRINKS P4 [live lever: shrink cand_count], (c) explicit low-prob fallback only for ~zero-temporal
-correlation rows). op26_r0's R0 ladder already does ~97% one-scan threshold admission; novel part
= fuse collect + fast-write + P4-shrink.
- Probe rung 0 (CRUX, host, analysis/crux_singlescan.py): on real v4cap bundles, can a HINT-derived
  single bracket [t_lo,t_hi] achieve count(>=t_lo)∈[K,kC] AND count(>=t_hi)<=K on ~all cells (=>
  fallback prob ~0)? And what is the certain-winner fraction (fast-write share) + contested band
  size (P4 workload) vs hit_rate?
- STATUS: **GO** (crux_singlescan.json). Real-data headroom:
  - one-scan happy = 100% on all cells except pro/4k (degenerate N≈K=1024; fallback = trivial
    near-full-selection path, NOT a low-hr failure — even flash/256k hr=0.30 is 100% happy).
  - fast-write share median **0.887** (89% of top-K are certain-winners → skip P4).
  - contested band/K median **0.29** → P4 cand_count 6K→~0.29K (~20× shrink; live lever).
  - oracle ceiling 0.998 (better threshold predictor => more fast-write headroom).
- Ledger reconciliation: DIFFERENT from Opt-L (which fused the full kC collect). Here the fused
  append is bounded ~1.2K (fast-write + tiny band), and the saving is P4-shrink + P3-removal, NOT
  the memory pass Opt-L measured. Revival condition satisfied.
- Prize sizing (op26 iter7 NCU, cluster fp32/65536 seg split): P3-collect 11-14%, P4 = large
  remainder. Removing P3 + shrinking P4 ~20× => large potential saving, esp. large N. Build it.
- Next: rung 3 kernel — GvrOp34SingleScanKernel ⊂ GvrOp26R0Kernel: fold P3 collect + certain-winner
  fast-write into the R0 multi-count pass; P4 over the contested band only; fallback to op26_r0 path
  when no happy rung (measured counts => always correct). Behind a flag (default byte-identical).
