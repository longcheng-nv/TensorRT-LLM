# ITERATIONS.md — op40_omni_gvr

(append-only; one entry per iteration, fixed verdict vocabulary
SHIP / FALSIFIED(+domain) / WASH / PIVOT)

## iter 0 — 2026-07-23 — FINDING (baseline defect, not an optimization iter)
Hypothesis: none (Phase 4 gate discovery during harness bring-up).
Result: vendored e612 baseline FAILS gate40 plateau track (giant fp32 tie class
~60% of row at one value): duplicate output indices (uniq 339/512 @K512 N8192,
347/1024 @K1024 N8192, 1986/2048 @K2048 N65536). Deterministic. Real captures
66/66 green; randn/narrow/neartie/hit/miss green.
Bisect: cs=1 cases flag-INSENSITIVE (p4tt/p4wr/p2wr/kcdiet/r0/p4rse all fail);
cs=4 case fixed ONLY by enable_r0=False. Tie value not at the K-boundary —
suspect candidate-collect/dedup under massive sub-threshold tie class.
Consequence: (a) reportable PR#16457 defect candidate (upstream tests use
continuous randn only, never exercise this); (b) op40 variants must pass
plateau even though baseline does not; baseline plateau cells excluded from
paired perf ratios (correctness there is undefined).
Ledger write-back: FALSIFIED.md none; defect logged here + repro_plateau.py.
Next: root-cause during P4/P3 characterization (task #4).

## baseline bl0 — 2026-07-23 — DONE (denominator for the campaign)
Fresh full-865 nsys re-measure of vendored e612 baseline on umb-b200-239
(8-way GPU shards, cold-L2, median of 20 reps/cell via parse_nsys_full).
865/865 cells measured, real-data exactness green, zero batch errors.
Overall gm 14.071 us. By (model, band) gm us:
flash 8.61/13.69 · pro 9.93/14.93 · v32 12.74/19.22 (small/large ISL).
Goal 1.60x == overall gm ~8.79 us. Observed grid min 5.33 us (flash 4k).
ANCHORS (expected us, drift >3% => re-baseline): pro_64k_L30=11.78 (cs1),
pro_256k_L30=15.13 (cs4), v32_128k_L14=18.31 (cs8), flash_128k_L42=17.63 (cs1).
Raw: results/bl0/cells.csv (committed); nsys reps local-only (gitignored).
