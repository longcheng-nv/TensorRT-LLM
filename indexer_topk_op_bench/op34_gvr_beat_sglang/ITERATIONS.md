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

- Probe rung 2 (kC-probe, nsys BS=1 real, scripts/nsys_op34.py + results/kcprobe): time op26_r0auto
  at kC∈{stock..768} — kC caps BOTH collect-writes AND P4 input, so this sizes the P4/fast-write
  prize WITHOUT the kernel rewrite. Result (us_cold geomean, exact vdiff=0 all arms):
    flash/32k r0=12.34 bestkc(1024)=11.77 (-4.6%); flash/256k stock best; flash/1024k stock best
    (smaller kc WORSE); pro/32k bestkc(1536)=13.92 vs 16.15 (-14%); pro/256k stock; pro/1024k
    smaller kc MUCH WORSE (kc1280=64 vs 42).
  => **iter1a (P4-shrink / fast-write) FALSIFIED at BS=1**: shrinking the candidate cap yields
     ≤14% (small N) to NEGATIVE (large N, fb_fix fallback fires). **P4 is NOT the BS=1 cost — the
     two full-N scan passes (R0 count + P3 collect) dominate** (single-CTA 1/148 SM bandwidth;
     matches op#8 root cause). The user's fast-write attacks the wrong phase for BS=1.
  Ledger write-back: FALSIFIED (kC-diet/P4-shrink as a path to 2x, {BS=1 fp32 real v4cap}, nsys) —
     complexity/structural: P4 is a small fraction at BS=1, full-N passes dominate; small kC triggers
     fb_fix extra passes at large N. (kC-diet remains a known ~4-14% small-N lever, not a 2x path.)
  Anchor note: probe used 3 layers/cell vs §10's 21-layer geomean + different node (074 vs 039);
     ABSOLUTE us differ ~19% but WITHIN-run ratios (the kC verdict) are valid (same GPU/session/layers).

## iter2 — 2026-07-14 — PIVOT: single-scan PASS-FUSION (the real lever the probe exposed)
Hypothesis: fold R0 count + P3 collect into ONE full-N pass, removing the 2nd full-N read (the
confirmed BS=1 bottleneck). Ledger check: Opt-L "fuse collect into count ≈ full pass, even
FORCE_HAPPY no speedup" — CONDITIONAL REVIVAL: crux shows count(>=t_lo)≈1.2K, so the fused append
is bounded ~1.2K (vs Opt-L's full-kC ~6K online slot-reserve); test whether a cheaper bounded-append
fused scan beats Opt-L's verdict on real v4cap. This is the honest test of the user's core single-scan
idea against the CONFIRMED bottleneck (not the falsified P4 half).
- STATUS: designing rung-3 kernel.
