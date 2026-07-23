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

## probe: launch floor — 2026-07-23 — PRIOR ESTABLISHED
nsys floors on b200-239 GPU4 (20 cold reps, median): trivial fill 1.30 us;
GVR degenerate identity path (prologue+emit, N<=K) 1.68 us; GVR smallest real
shape (K512, N=1024, 4KB) 8.53 us. => small-N cell time is ~80% ALGORITHM
latency chain, NOT an immovable launch floor. Overwrites the inherited
"~10 us structural floor" wall candidate: the floor is ~1.7 us + algorithm.
WALLS.md updated accordingly.

## characterization — 2026-07-23 — DONE (bottleneck map)
Phase table (26 cells, clock64 twin, us anchored to bl0): P4 dominant
everywhere — cs1 3.5-6.2us (40-47%), cs4/8 7.2-12.2us (~50%+, DSMEM
gather+wait); P2 1.0-4.5us and P3 0.9-3.9us scale with N; P1 gather flat
1.5-2.3us; P1b 0.4-1.1us (scales with K). Floor probe: true floor 1.7us.
e612 rank_scatter coarse search still leader-serialized; d2a/d2b/d1a flags
ABSENT from head (live in draft #16715) => lever pool available.
Hypothesis queue (ledger-checked): H1 cs1-P4 chain rebuild (d2a/d2b class),
H2 cluster-P4 peer-push (d1a class), H3 P1+P1b gather fuse, H4 P2 large-N
count cost, H5 P3+P4 fuse (hist during collect), H6 admission tightening
(undershoot-guarded), H7 plateau exact tie-fill (correctness).
Raw: results/phase_base_40.{log,json}, results/floor/.

## iter 1 — 2026-07-23 — IN PROGRESS
Hypothesis (ledger: none against; op37 primitives silicon-verified on
predecessor head): re-splice d2a (rw coarse/fine rank search) + d2b (tiny-bin
fine skip) + d1a (cluster peer-push) onto e612 rank_scatter as ctor flags,
default-off. All three splices landed with exact anchors (gvrpkg40v1).
Probe: rung-2 equivalent (primitives pre-verified) -> straight to gate + A/B.
Gate base,v1 running.

## iter 0b — 2026-07-23 — FINDING (baseline nondeterminism, neartie cs=8)
gate run 1 (base only): neartie_K2048_N262144 PASS. gate run 2 (base,v1 same
GPU, identical deterministic input): FAIL on BOTH arms including base.
=> e612 baseline is NONDETERMINISTIC on near-tie data at cs=8 (cluster path).
Suspect class: DSMEM visibility race (ledger: cluster_arrive_relaxed has no
release; reading a just-written scalar via DSMEM can see stale data; symptom
clusters wrong picks by CTA slice). v1 adds no new failures (failure set
identical to base). Flake probe (probe_neartie_flake.py, 100 reps x cs sweep)
queued to run AFTER ab_v1 grid finishes (no probes during timing runs).

## iter 1 — 2026-07-23 — SHIP (arm v1 = campaign running best)
Result: full-865 paired nsys (8-GPU cell-sharded, pairs same-GPU): gm 1.1261,
0 regressions (<0.97), worst 0.9892 @ pro_4k_L16, 865/865 real exact.
flash 1.1497 / pro 1.1588 / v32 1.0942; 32k-1M 1.1220 / 4k-32k 1.1328.
Reproduces op37's 1.1284 on the newer e612 head — levers independent of the
head refresh (p4tt/p4wr absorption did not eat them).
Cumulative vs goal: 1.126 of 1.60 (remaining x1.421).
Next: iter2 target selection from cs-level residual map.

## iter 0b CORRECTION — 2026-07-23
Pre-draw sweep (100 seeds x cs {1,4,8}): neartie_K2048_N262144 fails on 54/100
pre-draws with IDENTICAL failing seed sets across cs => deterministic,
cs-INDEPENDENT, data-dependent defect (not a DSMEM race). Earlier
"nondeterminism" was my gate's own seeding bug: hash(str) is
PYTHONHASHSEED-salted across processes (fixed: crc32). Defect class = P4
boundary tie handling on 1-2 ULP-spaced near-tie bands (same family as the
plateau duplicate-index defect). Baseline real-data green stands.
Consequence: P4 tail redesign is now correctness-obligated AND the perf
lever (H1) — one redesign serves both.

## probe: cluster-P4 gather split — 2026-07-23 — T1 QUANTIFIED
t5-relocation twin (gvrpkg40v1g) vs v1t differential on 9 cs>=4 cells:
cluster P4 = DSMEM gather+straggler-wait 2.4-4.4us + leader-only P4 compute
2.8-6.1us (both WITH d1a peer-push ON). Worst: v32_64k cs4 = 4.35 + 6.14.
=> iter4 design "distributed P4": per-CTA local coarse hist during P3, DSMEM
hist-reduce (256 ints instead of bulk candidates), redundant k-th-bin on all
CTAs, distributed global-atomic scatter, only straddle class crosses to
leader for exact tail. Expected: gather -> ~0.5-1us, compute parallelized.
Raw: results/phase_v1g_40.{log,json}.
