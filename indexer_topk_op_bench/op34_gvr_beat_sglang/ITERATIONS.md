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
- STATUS: SUPERSEDED by iter2b re-anchor + NCU CRUX (below). Pass-fusion is a dead lever
  under cold-L2 — see analysis/NCU_CRUX_048.md.

## iter2b — 2026-07-15 — RE-ANCHOR (node 074->048) + DECISIVE NCU CRUX (reframes campaign)
Session resumed on node umbriel-b200-048 (was 074). Re-established anchor + ran the decisive
NCU attribution the paper feasibility analysis never had. analysis/{ANCHOR_048,NCU_CRUX_048}.md.
- Anchor (pro/256k N=65539 L32 hr=0.489, cold-L2 wallclock ×30): sglang 21.6us / op26_r0auto
  31.2us = 1.446x; op26_r0 EXACT (vdiff=0). Ship goal here = 1.88x over r0.
  GOTCHA fixed: GVR seq_lens must be N*cr (uncompressed); N alone → scans N/4 → recall 0.
- NCU CRUX (op26_r0 vs sglang, same cell, ncu --cache-control all = cold):
  * op26_r0: grid (1,1,1)=1 SM, 43.3us, DRAM 0.11% peak, SM 0.23% peak, occ 50%.
  * sglang:  grid (1,8,1)=8 SM, 28.2us, DRAM 0.17% peak, SM 0.83% peak, occ 50%.
  * BOTH read ~360KB = 1x row (pass count NOT the wall under cold-L2). BOTH <1% DRAM AND <1% SM
    ⇒ **LATENCY-BOUND, not bandwidth/compute-bound.** sglang's ONLY structural edge = 8-CTA MLP
    (8x outstanding loads to hide latency), 1.54x for 8x SMs (sub-linear).
- Ledger write-back (FALSIFIED): pass-fusion / single-scan collect-into-count as a BS=1 lever —
  FALSIFIED on the MECHANISM (cold-L2: both do 1 HBM read; fusing saves only an L2-hot pass =
  op29's measured 1.03-1.13x). Domain {BS=1 fp32 real v4cap cold-L2}, evidence NCU. This closes
  the user's original single-scan idea as a 30% path (it was already partially ledgered as Opt-L).
- Live levers now precisely named: (a) multi-CTA-per-row >8 to exceed sglang's MLP (147 idle SMs
  at BS=1), (b) intra-CTA MLP (deeper unroll/sw-pipeline). WALL risk: GVR multi-phase barrier
  chain (P1..P4..fb) doesn't shrink with CTAs; sglang is lean. 
- Next: iter3 CRUX-A (multi-CTA scan-scaling microbench: does MLP keep scaling past C=8?) +
  CRUX-B (op26_r0 phase breakdown for the multi-CTA Amdahl ceiling).

## iter3 — 2026-07-15 — CRUX-A GO (MLP scans past C=8) → build multi-CTA
CRUX-A (scripts/crux_a_mlp.py, analysis/CRUX_A_MLP_048.md): bare count scan keeps speeding up well
past C=8 (N=262144: C8→C64 = 34.6→9.3µs = 3.7×; sglang fixed at C=8). CRUX-C proxy (crux_c_proxy.py)
GO: multi-CTA collect C64 = 14–21µs NCU, EXACT on real data. Napkin said tail budget 7–9µs vs
sglang. Verdict GO → build the kernel. (Caveat later proven decisive: NCU inflates absolute µs;
real nsys sglang is 12–19µs, not 28–39µs.)

## iter4 — 2026-07-15 — BUILD multi-CTA + nsys A/B → FALSIFIED, then DOUBLE-LOCK
Built src/op34_mcta_op.py: multi-CTA single-pass GVR (grid=C CTAs/row; each CTA stream-compacts
elements >= hint threshold via block prefix-sum + 1 atomic/block-iter; tail = exact top-K on
candidates). EXACT (vdiff=0) on real pro grid, all 18 cells.
- nsys A/B (results/harvest_pro): op34_mcta = 76–125µs = **4–8× SLOWER** than sglang (12–19µs).
  Root cause: t=hint.min is exact but admits M=16K–100K candidates on real data → heavy collect +
  heavy tail. A tighter hint-quantile rung gives small M but MISSES exactness (count<K) — the GVR
  threshold-miss that forces op26's 2-pass structure.
- DECOMPOSITION (results/decomp2, analysis/DOUBLE_LOCK_048.md): oracle-threshold collect-only
  (col_orac, UB best case, C=64, no tail) = 16–17µs @1024k / 12µs @256k ≈ sglang's ENTIRE kernel.
  ⇒ **LOCK 1 (UB): the leanest possible GVR scan+collect merely EQUALS sglang; the mandatory rank
  tail then pushes it over. No 30% headroom at the information floor.** LOCK 2 (relaxed controls):
  oracle-full 2.7–4.0×, hint 4–8×, op26_r0 2.2–4.6× — all parity-or-worse, none at +30%.
- **VERDICT: STOP — double-locked INFEASIBLE (pre-authorized negative conclusion, AUTONOMY.md).**
  sglang_v2 remains best at BS=1. No conditional +30% region (even oracle col-only never beats
  sglang by 30%). Ledger + WALLS updated. Remaining: full-grid regime map + bilingual HTML report.

## iter5 — 2026-07-15 — CONVERGED: full-grid regime map + report (STOP)
Full 18-cell grid (both models × 9 ISL × 3 arms × 3 layers, nsys cold, results/grid, 162 recs 0 err):
- op26_r0auto vs sglang: 1.56–3.38× SLOWER, GRAND geomean **2.11×** — loses at EVERY regime.
- op34_mcta vs sglang: 5.58–12.03× SLOWER, GRAND geomean **8.58×** (worst at small N = merge
  overhead + phase floor) — loses at EVERY regime.
- NO cell where any GVR arm beats sglang. Goal was <0.77×. Confirms the double-lock across the full
  envelope; small-N walled a fortiori (op32).
Deliverable: report/op34_report.html (bilingual EN/中, CSS-only toggle, static SVG, data-driven via
report/gen_report_op34.py). Memory saved. RESUME_PROMPT + COST updated.
**CAMPAIGN CLOSED — sglang_v2 remains best BS=1 top-K; +30% double-locked infeasible in GVR skeleton.**
