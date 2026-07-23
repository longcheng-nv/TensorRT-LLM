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

## iter 2 — 2026-07-23 — FALSIFIED (K512/K1024 domain) / partial harvest (K2048)
Hypothesis (ledger hit acknowledged: upstream audit "extra count column costs
3-7%", revival tried on real-865-only envelope): wider R0 ladders raise hit
rate enough to beat the column tax.
Result: full-865 4-arm paired nsys, exactness green. v2lad vs base gm 1.1136
(< v1's 1.1261); vs v1 gm 0.9894 — flash 0.9617 / pro 0.9669 / v32 1.0180.
The upstream audit verdict REPRODUCES on the real envelope for K512/K1024.
K2048 4-rung ladder (0.8,0.6,0.4,0.25) is a keeper: +1.8% over v1 on v32.
v2c (p1b_cache fp32 K512/1024): WASH vs v2lad (delta < 1%).
Ledger write-back: FALSIFIED.md entry (ladder widening, K512/K1024, real fp32
BS=1, nsys, complexity-backfire); K2048 ladder harvested into v3k.
Next: iter3 = v3 distributed radix P4 (smoke 5/5 exact incl cs4/8) — gate.

## iter 3 probe update — 2026-07-23 — neartie root cause RELOCATED to P2
v3 (fully rewritten radix P4) fails neartie with the IDENTICAL 54/100 seed
set as base => defect is NOT in P4. Root cause: P2 admission fail-soft —
near-tie bands need ULP-level bracket resolution; fb log-falsi caps at 8
iters -> undershoot (count < K) -> pad/dup. Same class as plateau. One fix
covers both: p2_radix_fallback (full-row distributed radix select on
fail-soft, exact by bit-level digits). Promoted to the iter5 keystone.

## iter 5 (keystone, implemented ahead of iter3 verdict) — 2026-07-23 — smoke GREEN
v4 = v3k + p2_radix_fallback: exact full-row distributed radix select
(radix_select_row) replaces the P2 fail-soft undershoot; P3/handoffs/P4
skipped via s_iscalars[1]==4 sentinel when the fallback emitted.
Smoke: plateau 3/3 FIXED (base/v1/v3 all fail these), neartie 0/20 pre-draws
inexact at cs1 AND cs8 (base ~54%). Full gate + grid pending iter3 grid end.
This is the first arm that passes every adversarial track — baseline defect
class (P2 fail-soft under-fill) closed with a bit-exact bounded-cost path.

## iter 5 gate — 2026-07-23 — GATE GREEN 69/69 (first fully-green arm)
v4 passes all three tracks including plateau + neartie adversarials that the
e612 baseline fails. Exactness obligation of the ship rule now satisfiable
strictly ABOVE baseline correctness.

## iter 3 — 2026-07-23 — FALSIFIED (distributed radix P4 as the perf path)
Result: full-865 4-arm grid. v3 vs v1 gm 0.8696 (flash 0.8979 / pro 0.8773 /
v32 0.8517); v3k vs base 0.9893. Losses worst at cs8 (v32_128k 0.64) but
cs1 also -10%.
Diagnosis: real fp32 keys almost never exact-fit -> full 4-level descent is
the common path; per level = 2 cluster barriers + 2 candidate scans; total
8 cluster barriers + 8 scans vs the old 1 gather + coarse+fine. Pass-count
economics beat DSMEM-traffic economics here (echo of op14's lesson, now on
the SMEM/barrier axis).
Ledger write-back: FALSIFIED (multi-level distributed radix as P4 perf path,
all cs, real fp32 BS=1, nsys) — complexity-backfire; revival condition:
hybrid with <=1 distributed level + small-class handoff. RADIX MACHINERY
RETAINED as the exactness fallback (v4) where it is unconditionally correct
and cost-bounded to pathological rows.
Next: iter6 = P1+P1b register fusion (kill the 2nd gather + a barrier on ALL
cells, ~0.4-0.7us each); iter7 = T2 scan-ILP microbench probe; iter8 = v3b
single-level float-bin distributed hist + class-only gather (cs>1).

## probe: cs-choice sweep — 2026-07-23 — WASH (config already near-optimal)
v5best-class arm, 11 large-N cells, cs {1,2,4,8} paired nsys: pick_config's
choices are best or within noise everywhere; only v32 32-64k shows cs8 +2-3%
over the picked cs1/cs4. cs2 never wins. Deferred (simplicity criterion).

## iter 7 — 2026-07-23 — FALSIFIED (mt_unroll=8)
v5mt8 vs v5best gm 0.9625 (worst flash 512k/32k 0.86-0.88). Register pressure
/ issue economics beat the extra in-flight loads at T=512/1024. Ledger entry.

## iter 6 — 2026-07-23 — verdict: K2048 ladder DROPPED (ship-rule tail loss)
v5best vs base gm 1.1312 BUT 24 cells < 0.97 (v32 16-64k, worst 0.9302) —
and regressions exist even inside the gain bands (v32_128k min 0.8932):
ladder benefit is cell-content-dependent, not N-dispatchable (hit-rate
dispatch forbidden). op33 pattern (mean win, tail loss) => DROPPED.
Fallback code-presence tax measured: flash -0.87% (paired same-rep), pro
+0.1% — accepted for correctness (both baseline defect classes fixed).
Ship candidates now v7 = v1 + p2_radix_fallback; v8 = v7 + p3_hist_fuse.

## probe: T-sweep 512 vs 1024 — 2026-07-23 — WASH (content-dependent)
Mid-N cs1 cells: T1024 helps v32_32k +5.3% / pro_128k +3.3%, hurts
flash_64k -7% / pro_32k -5.2%. Content/K-dependent, small aggregate; deferred
(simplicity). Raw results/tsweep/.

## probe: H9 speculative-P3 stats — 2026-07-23 — PARKED (EV insufficient)
Admission-column replay: vseed admits only 25-32% of R0 hits (15% of cells);
q-rung speculation ~50-65% hit but requires conditional stores inside the
latency-sensitive multicount loop (mt8 falsification shows sensitivity).
Parked; revival = if P3 remains the top residual after v8.

## iter 8 — 2026-07-23 — v7 SHIP (running best) / v8 fusion N-profile pending
Full-865 4-arm grid, exactness green:
- v7 (v1 + p2_radix_fallback) vs base: gm 1.1250, ZERO regressions (<0.97),
  worst 0.9732. Fallback net tax ~0.1% vs v1's 1.1261. SHIP candidate: only
  arm that is regression-free AND green on every adversarial track.
- v8 (+p3_hist_fuse) vs base gm 1.1323 but 3 cells < 0.97; vs v7: +3.3% on
  the 4k-32k band, -8% cells across 128k (range-skew: values above pmax crowd
  the top bin -> fine recursion tax). N-band analysis running to decide an
  N-gated fusion (dispatch budget has room: 0 rules used so far).

## iter 8b — 2026-07-23 — v8 FALSIFIED (fusion tails, no admissible gate)
N-band v8/v7: gm +3.4-4.6% at N<=8k BUT content-dependent losers inside
every band incl N<=2k (pro_4k_L16 0.9287 vs base); 128k band -1.4% gm with
0.92 cells (pmax range-skew -> top-bin crowding -> fine recursion tax).
No static N-gate clears the 0.97 rule. Mean +0.65% not worth tail risk.
Ledger: FALSIFIED (p3_hist_fuse as static or N-gated config, real fp32 BS=1);
revival: tighter hi-bound estimation or top-bin split — parked.
STALL PROTOCOL: >=3 flat iters since v1 -> meta-analysis (UB bounding,
convergence assessment). v7 = running best: gm 1.1250, 0 reg, gate green.

## probe: enable_smem_cache — 2026-07-23 — IN PROGRESS, already DISQUALIFIED
smem_cache=on is INEXACT at flash_1024k cs8 (latent defect in the default-off
flag; not production-affecting). Lever dead regardless of timing.

## NCU attribution — 2026-07-23 — NEW MECHANISM: icache/fetch-bound
pro_64k v7 (NCU full): grid=1 CTA/148 SMs (structural), 80 regs, occupancy
25% of its one SM, DRAM 0.1%, SM 0.09%, IPC 0.68 — and **47.6% of stall
cycles = instruction-fetch / icache miss** (NCU est. speedup 47.6%). The
mega-kernel's code size (all phases + R0 + fb + P4 coarse/fine/tail +
radix fallback inlined) blows the icache on a latency-bound single-CTA
kernel. Explains: fallback code-presence tax, mt8 backfire, op21's
"never-executed code still costs".
iter9 = unroll-reduction probes (v9a mt2 / v9b no-p3unroll / v9c both+no4)
— counter-intuitive lever: LESS memory-ILP unrolling may net-win.

## iter 9 — 2026-07-23 — FALSIFIED (unroll reduction vs icache wall)
v9a (mt2) gm 0.9821 / v9b (no-p3unroll) 0.9985 / v9c (all-off) 0.9472 vs v7.
Small-N gains (+3-6% flash_4k/pro_8k) swamped by mid/large-N scan losses —
memory-ILP unrolling earns its icache cost where scans dominate. The 47.6%
fetch-stall wall stands: structural fix = splitting the mega-kernel, not
expressible in the single-kernel cuteDSL GVR structure. -> WALLS.md.
CONVERGENCE: lever pool exhausted (9 iters: 1 SHIP, 6 FALSIFIED, rest WASH/
PARKED). v7 = final arm. ab_iter8 grid = terminal L2 verdict (gm 1.1250,
0 regressions, 865/865 exact, all-adversarial gate green).
