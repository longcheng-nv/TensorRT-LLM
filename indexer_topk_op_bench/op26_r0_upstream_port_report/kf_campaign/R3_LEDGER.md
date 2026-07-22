# R3 campaign ledger — gvr-topk-r3 (beyond-champion)

## CLOSED 2026-07-22 — SHIP: compB

- Campaign cancelled at operator close-out (round 4; its best 1.1111 internal
  below the harvested composite — kill-line consistent). Final platform cost
  **$764.66** (4 rounds, 27 agents launched, 628M in / 4.3M out tokens).
- **Final verdict (fresh full-865, clean anchors med 1.006/p95 1.063):
  compB = 1.8267× geomean vs PR#16457 current head b14ec40e1b; 0 regressions
  (min 1.140, NO borderline); 865/865 exact. +8.1% over campaign-1 champion.**
  Bars: 1.8267 ≥ 1.60 ✅ / zero-reg ✅ / exact ✅.
- Rivals (PR-arm-normalized): sglang v2 1.215×, radix_cutedsl 1.760×,
  flashinfer 1.585×.
- Ship branch: `fork/kf/gvr-topk-compB` @9dbd6ee20a (code-only, stacked on
  campaign-1 ship branch). Report: KF_PROCESS_LOG.html §7 (gen_r3_section.py).

Campaign id: `e5q1zgrfhs0z57dj6850kc444r` (KF managed B200, effort high,
6 agents/round: 2×fable-5(high) + 2×gpt-5.6-sol(high) + 2×n3-opus-4.8;
max_rounds 20, max_duration 8h, max_cost $800, stagnation 4).
Node: umbriel-b200-027 (8×B200). Started 2026-07-21 13:40Z.

## BS-scaling supplement (2026-07-22, post-close — §7.8 of KF_PROCESS_LOG.html)

compB across BS 1-1024 (real rows replicated, 75 rungs = REPORT §8b per-layer
envelope), all 5 arms local-paired on umbriel-b200-019 (kf_bs_scaling/):
BS=1 wins ALL arms (vs PR 1.874 / sglang_v2 1.182 / radix 1.754 / flashinfer
1.606); crossover already at BS=2; BS=1024 batched arms 45-125x faster ->
production port needs a BS==1 dispatch gate (legal: BS known at launch).
Cross-node PR-normalization was REJECTED: only the cuteDSL GVR arm drifts
locally (med 1.09, p95 2.4 at high BS; rivals reproduce 027 at med 1.000).
One-off spin-barrier LIVELOCK observed (compB v32_128k_L54 BS=1024, >40min,
solo retry clean 9ms/call) -> transient co-residency loss, not data-dependent;
new evidence for the fence-less-barrier ship constraint above.

Mechanism of the near-linear BS collapse (2026-07-22 analysis, confirmed
against harvest/r3_compB/kernel.cu): compB has NO batch dimension by contract
(`topk_launch(logits, n, k, out, stream)` is single-row per the SOLBench
problem), so BS>1 = host loop of BS sequential same-stream launches. Each
large-n launch sizes its grid to full-GPU co-residency
(cudaOccupancyMaxActiveBlocksPerMultiprocessor + sense-reversing grid
barrier), i.e. one row saturates the whole B200 -> total time is exactly
proportional to BS. Concurrent per-row streams are illegal: static global
g_scratch/g_gen (generation-token barrier words) would race. Batched rivals
launch once with batch as a grid axis and are latency-bound/underutilized at
BS=1, so extra rows are near-free until ~BS=128 saturation — hence the ratio
halves per BS doubling (1.874 -> 0.925 -> 0.470 -> 0.238 ...) and flattens at
~0.022-0.025 once rivals also go linear. Launch gaps are NOT the cause:
span/kernel tax measured <=1.04. Net: the collapse is the designed
whole-GPU-per-row latency trade, structural, not fixable in-harness ->
BS==1 dispatch gate is the correct (and only) production remedy.

## BS>1 extension design analysis (2026-07-22, unimplemented proposal)

Premise: BS=1 large-n is latency-bound (NCU DRAM <1%, issue ~15%, barrier
stall ~48% of runtime) -> huge free parallelism at low BS. High-BS limit is
the DRAM roofline (BS*n*4 B, compB is already single-pass-read) -> goal is
win at BS 2..~64 and TIE rivals at saturation, not beat them everywhere.
Hard constraint: topk_fast register-caches the whole row (float4+tail per
thread, 2048 elts/CTA), so each row needs EXACTLY ceil(n/2048) CTAs.

Layered plan:
- (A) small/mid-n tiers (n<=16896): single-CTA scratch-free kernels ->
  trivial batching grid.x=BS (row per blockIdx.x). Near-linear, zero risk;
  expected to beat all rivals immediately (148 SMs, one row per SM).
- (B) large-n single-wave: while BS*ceil(n/2048) <= co-residency cap
  (~active*148), launch all row-teams in ONE co-resident grid; per-row
  scratch slices (~52KB/row) + PER-TEAM barrier (fewer participants =>
  shorter stall, and teams hide each other's spin). Per-row efficiency may
  even exceed BS=1. E.g. n=131072: 64 CTA/row, BS<=9 fits one wave ->
  expect ~1.8x where today is 0.238x (BS=8).
- (B') multi-wave: persistent team array (grid<=cap) + atomic row work-queue,
  single launch consumes the whole batch. Also removes the back-to-back
  launch pattern that triggered the observed spin-barrier livelock.
- (C) CORRECTNESS RISK (the one new hazard): fence-less safety leg (2) "L1
  invalidated at launch boundary" DIES when a persistent team reuses its
  scratch slice for row i+1 within one launch (stale hist lines in L1).
  Remedies, cheap->dear: slice double-buffer rotation + sense token widened
  from gen*8+pass to (row_iter,pass); post-barrier hist/count reads via
  ld.global.cg (L2-only, expected << the measured -11% threadfence tax);
  fallback per-row __threadfence (multi-wave only).
- (D) ragged batch (production: per-row n = per-request KV len): host-side
  bucketing by existing dispatch tier, one batched launch per bucket;
  largest-first queue order in the large-n bucket to shrink the tail wave.
- (E) dispatch gate replaces BS==1 hard gate: BS==1 -> current compB
  (verdict preserved); small-n -> (A); large-n one-wave -> (B); multi-wave
  -> (B'); beyond measured crossover -> fall back to batched arm (fuse).

Validation order: (A) first (1 day); then fixed-grid single-wave (B) at
n=131072 BS in {2,4,8} local-paired vs rivals — this alone
falsifies/confirms the core "team parallelism is free" hypothesis before
investing in the persistent queue. Gates: 865-grid + BS-sweep dual
exactness, nsys cold-L2, per-batch p95 anchors, GVR arm local-retest rule
(cross-node drift med 1.09). Watch: L2 set-conflicts across BS hist slices;
spin threads stealing issue slots under full SM load (nanosleep backoff).

**MINIMAL VALIDATION DONE (2026-07-22, umbriel-b200-039, kf_bs_scaling/ext/):
both hypotheses CONFIRMED, 126/126 exact.** nsys cold-L2, 3 arms paired per
cell (gvr_pr local anchor / compB-seq / compB-ext), real rows, 6 cells.
(A) grid.y batching of the single-CTA tiers: flat to BS=64, beats gvr_pr
1.61-1.80x at EVERY BS 1..1024 (16/16 wins per BS row), up to 157x vs
sequential compB — small-n win region extends to the whole BS axis.
(B) row teams at N=131075 (team=65, measured cap=296 => active=2/SM
register-bound, rows_per_wave=4): BS=1 parity 1.006/1.016 (team-only grid ==
shipped bump-to-148 grid); single wave BS=4 costs +5% over BS=1 => extra
rows ~free as predicted; vs gvr_pr 2.49x/1.81x @BS4 (was 0.470x sequential);
multi-wave linear in waves, crossover vs PR arm at BS~8, batched arm wins
from BS=16. Span/kernel tax <=1.02 everywhere. Implied gate: ext for
n<=16896 any BS; large-n ext while BS <= rows_per_wave (~2 waves), else
batched arm. Next lever: register diet on topk_fast to raise active 2->4+
(smem allows 5), doubling rows_per_wave and pushing the crossover out.
Details + protocol: kf_bs_scaling/ext/RESULTS.md (ext_bs.csv canonical).

**Register-diet DONE (same day): active 2->4 achieved, crossover pushed
BS~8 -> BS~16.** Limiter was registers (56 regs), NOT smem/carveout.
`__launch_bounds__(512,4)` template forces 32 regs with ZERO local spill
-> cap 592, rows_per_wave 9 @ team=65. nsys verdict (48/48 exact,
diet_bs.csv): diet tax ~2% at BS<=4; BS=8 now single-wave 11.2us =
1.73x over active=2 ext, 2.20/1.61x vs gvr_pr; multi-wave region uniformly
+1.73-1.77x; pooled v4-vs-gvr gm: 2.29/2.06/1.88/0.98/0.54 at BS
1/4/8/16/32. Ship shape: single variant MINB=4 (dominates from BS=8, -2%
below) or v1@BS<=4 + v4@BS>=8 dispatch. active=5 would need <=25 regs —
diminishing; remaining headroom is the persistent queue (B').

**B' persistent queue DONE: FALSIFIED on uniform batches (no-ship).**
Implemented in full with the C-remedies (per-team slice reuse + __ldcg
post-barrier reads + gen*8192+iter*8+pass tokens + inter-row team barrier,
skipped on last row) — remedies CORRECT (42/42 nsys + all-row exactness to
BS=128), but pq loses to chunked ext_v4 everywhere: kernel-sum 1.05-1.31x,
fair span metric 1.03-1.31x slower, converging-not-reaching parity at high
BS. Economics: persistent can only reclaim the inter-wave launch gap
(measured <=2%, v4 span tax 1.01-1.02) while paying a 5-30% overhead floor
(per-row barrier + cg loads + loop codegen, 72-80B spill). Ship shape stays
chunked ext_v4. Residual niche = ragged batches (drain-limited waves), but
those break uniform team sizing => different design (per-bucket teams +
real atomic queue), not an upgrade of this kernel. pq_bs.csv +
ext/RESULTS.md.

### BS>1 extension — FINAL PICTURE (3 experiments closed, 2026-07-22)

One day, three verdicts on umbriel-b200-039, all nsys cold-L2 local-paired
vs gvr_pr on real rows, 216/216 records exact across the campaign:
A grid.y batching CONFIRMED (@3fc4a4e82b) -> register diet CONFIRMED
(@a569e02f96) -> B' persistent queue FALSIFIED (@4b3914a4b4).

Production dispatch shape for a compB-lineage batched port (supersedes the
plain BS==1 gate of the §7.8 supplement):

| regime | ship path | vs PR head |
|---|---|---|
| n <= 16896, ANY BS | grid.y batched single-CTA tiers | 1.61-1.80x, 16/16 BS rows won (BS to 1024) |
| large-n, BS <= rows_per_wave (~9 @131K) | chunked ext_v4 (MINB=4), single wave | 1.61-2.69x |
| large-n, BS ~ 2 waves (~16) | ext_v4 | ~parity (flash 1.13 / pro 0.84) |
| beyond | batched PR arm (fallback fuse) | PR arm wins |

Load-bearing facts: (1) team = ceil(n/2048) exactly — the register-resident
constraint is hard, and the shipped bump-to-148 is unnecessary (BS=1 parity
1.006); (2) extra rows inside one co-resident wave are ~free (+5% BS=1->4);
(3) occupancy was register-bound (56 regs), launch_bounds(512,4) gives 32
regs with ZERO spill and only ~2% single-row tax — cheapest lever of the
campaign; (4) multi-wave time is exactly waves x single-wave, and chunked
launch boundaries are effectively free (span tax <= 1.02), which is
precisely why B' had no room: any persistent design chases that <= 2% while
paying a >= 5% in-kernel sync floor; (5) the fence-less L1 hazard remedies
(ldcg post-barrier reads + gen*8192+iter*8+pass tokens + inter-row team
barrier) are proven correct and reusable wherever intra-launch slice reuse
appears — they just don't pay HERE.

Open items for a production port: ragged-batch design (per-bucket teams +
atomic queue — new kernel, only if serving traces show drain-limited
waves); v30 lineage (K=2048 mid-n 16896<n<=140000) still sequential in the
ext — needs the same team treatment before a full-envelope batched ship;
865-grid regression gate + BS-sweep dual exactness before any PR.

**D1 throughput arm — minimal validation DONE (2026-07-22): direction
CONFIRMED, BS 8-1024 target at 93%.** Barrier-free 3-kernel pipeline
(split-row hist / split-row collect / single-CTA finish, __ldcs streaming;
launch boundaries = all sync, fence-less legs stay valid — no ldcg).
Final verdict (tp_bs.csv, 36/36 exact, N=131K): tp/gvr pooled 1.97x @BS256,
2.60x @BS1024 (flash 3.09) — roofline math held; best-arm(tp,v4) gm over
BS 8-1024 = **1.489x vs PR head** (target 1.6, iteration-1 was 1.29).
Residual deficit localized at BS 32-64 (0.85/1.12) where gvr is still
latency-flat (~27us) while tp scales with work; tp @BS32 ~31us vs ~5us
2-pass DRAM floor => large headroom. In-experiment lessons: (i) valley was
underfeeding — C 2xSM/cap8 -> 4xSM/cap32 + dropping K2's per-element dual
ballot (hits ~ k << n => plain atomics win) took flash BS=8 30.6->17.1us;
(ii) RACE class note: re-zeroing a global hist in the SAME kernel that
plain-loads it needs a __syncthreads between the loads and the zero loop
(stochastic 1-row corruption at BS=1024, 122,880 row-checks clean after
fix). Next levers to close 1.6x: D2 sampled-estimate single-pass collect
(~40% byte cut), K2+K3 fusion for whole-bucket rows, C shape at BS 32-64.
ext/RESULTS.md final section.

**D2 sampled-estimate single-pass DONE (2026-07-22): 1.6x target
effectively met — best-arm gm BS 8-1024 = 1.597x vs PR head (D1 1.489).**
tp2 = whole-row uniform 1/16 sample -> budget-driven b_safe (deepest bucket
with expected cand <= CAP2/2; no delta hyperparameter) -> single full-read
candidate collect -> exact candidate-set finish; cand>=k superset invariant
+ count-check fallback (adversarial rows fall back, stay exact). 48/48
exact, fb=0 on real cells. tp2/gvr pooled 2.37x @BS256 / 2.67x @BS1024;
per-arm winner: v4 @8, tp @16, tp2 @32-256 (+fl1024=tp). Residual valley
BS 16-32 only (0.88-1.17 — gvr latency-flat denominator). Lessons: (i)
sampling MUST be spatially uniform — slice-based block sampling biased the
estimate on real positionally-structured rows (pro 100% fallback; host-sim
diagnosis); (ii) ballot-aggregation tax lesson now learned TWICE (1-4% hit
rate => plain atomics). Known issue: flash@1024 tp2_collect +50us on equal
bytes (bursty same-address atomics suspected) — best-arm dispatch covers.
ext/RESULTS.md D2 section; tp2_bs.csv.

One-line thesis: the collapse is a misallocated parallelism axis, not an
algorithm loss — reallocate the co-residency budget from "1 row x whole GPU"
to "BS rows x ceil(n/2048)-CTA teams"; the barrier demotes to team scope
(cheaper), and the only new correctness work is re-proving fence-less
safety under intra-launch scratch reuse (ld.global.cg + wider token).

## Decisions

- **D4 (skeleton adjudication, USER, 2026-07-22): Bar-first, loose-skeleton
  per campaign-1 precedent.** Constraint (a) preIdx-prior vacated by
  measurement evidence (12 June falsifications + campaign-1 r1 ×3 + R3
  5f3daaf8 WASH 1.0001); (b) retained as histogram-prefix threshold
  refinement; (c) fully retained. No cosmetic hint path. Ship object = compA
  lineage; report must state this explicitly with the evidence chain.

- **D1 (baseline packaging workaround).** `kf campaign prepare` with
  `--baseline-solution` failed: the platform baseline evaluator does not stage
  campaign assets (0/28 workloads — safetensors missing). Workaround: champion
  per-workload platform timings extracted from campaign-1 trace of kernel
  `c74fb3c0` (28/28 PASSED, geomean 10.76 µs) → `gvr-topk-r3/baselines.jsonl`;
  champion full source inlined into prompt.md v2 instead. c74fb3c0 timings ==
  c74f_sbx timings on this subset (no cell in the sbx graft rung 8448<n≤16896).
  Residual risk: the sbx graft raised `topk_small` `__launch_bounds__` 768→1024,
  which could shift register allocation on small-n rungs; local ab_sbx showed
  no give-back, accepted.
- **D2 (PR head moved).** PR#16457 head advanced e6fdbfac3d → `b14ec40e1b`
  (182 commits; GVR deltas: P4 bracket-window histogram + multi-level
  refinement, lane-parallel bin-search, redundant-warp sync reduction, parity-
  buffered DSMEM count exchange, float-domain bin clamp; only
  `gvr_topk_decode.py` changed among gvrpkg files, 210833→247384 B).
  `gvrpkg_head/` refreshed in place; old arm kept at `gvrpkg_e6fd/`.
- **D3 (foreign GPU load).** Intermittent short-lived foreign job bursts
  observed (67-77 GB, up to 83% util), hopping GPUs (0-3 then 7). Full grids
  run on 7 shards (GPUs 0-6) when GPU7 busy; contamination watched via
  per-rung pr_cold anchors + per-batch p95.

## Timeline

| ts (UTC) | event |
|---|---|
| 13:33 | prepare attempt #1 with baseline_solution → platform asset-staging gap (D1) |
| 13:40 | campaign started (baselines.jsonl path), monitor armed |
| 13:47 | gvrpkg_head refreshed to b14ec40e1b, import OK |
| 13:52 | 28-cell probe champh2 (GPU6): champion vs NEW head cold gm **1.7193**, 0/28 reg, 28/28 exact |
| 13:56 | full 865-cell grid champh2 launched, 7 shards GPUs 0-6 |

## Round log

- **Round 1** (13:40–~17:45Z, 6 agents, 17 kernels): best internal 0.9956 =
  verbatim champion resubmission (`821e5e5f topk_champion_final`, diff=0 vs
  c74f_sbx) → direct calibration of platform eval noise: identical code scores
  −0.4%; an agent also logged the same solution.json timing 17/23/20 vs
  23/28/26 µs across runs. `0d057e1e` = trivial rebase (0.9899).
  Only genuine variant: **`5f3daaf8` (0.9926)** — warm-hint min-threshold
  filter in coop pass-0 + final collect, gated n≥512K, with a provable
  ≥k-admission superset argument (min over logits[pre_idx] ⇒ pool ≥ k ⇒
  exact regardless of hint quality). Harvested to `harvest/r3_5f3daaf8/`;
  local probe pending GPU quiescence.
  Insights (40): grid.sync barrier stall ≈48% of large-n runtime (NCU);
  falsified: hand-rolled atomic barrier, block-count sweeps, TMA/cp.async
  hist prefetch, warp-agg (__match_any) hist accumulation, 15-bit 2-pass smem
  hist, champion+hier hybrid, 3 pre_idx threshold grafts (unGated), tight
  T_seed, single-1024-block collapse. Round 2 launched ~17:45Z.

- **Local-timing pause**: from ~17:40Z an 8-GPU foreign job occupies all GPUs
  (~118 GB resident, 1–17% util bursts). All local probes/grids paused until
  quiescence (monitor armed) per no-probes discipline.

## Barrier-ordering study (engineer variants of 09d13c81)

Measured cost of FORMAL memory ordering in the hand-rolled grid barrier
(28-cell probes GPU1, clean anchors):

| variant | ordering | cold gm vs PR | note |
|---|---|---|---|
| 09d13c81 as-harvested | none (relaxed intrinsics) | ~1.72-1.75 | fastest |
| + __threadfence pair | full membar.gl | 1.5657 | −11% |
| acq_rel asm barrier | scoped acq/rel | 1.6142 | −8% |
| relaxed asm (clobber only) | none | ~1.65 (6-cell 1.55) | clobber alone −5-7% |
| surgical (relaxed spin + trailing acquire) | scoped | 1.6146 | ordering cost is intrinsic: release-add must wait for the block's L2-pending writes on the critical path |

Conclusion: 09d13c81's win comes precisely from omitting barrier ordering +
avoiding the cooperative-launch premium. Safety argument for shipping
fence-less ON THIS PATTERN: (1) merged-hist lines are never plain-read before
the barrier within a launch (first plain touch is post-barrier), (2) L1 is
invalidated at kernel-launch boundaries, (3) pre-barrier writes are L2 atomics
⇒ post-barrier plain loads must miss L1 and fetch fresh from L2. Constraint
documented: any future edit that plain-reads merged hist pre-barrier, or an
L1-persistent-across-launch arch, breaks this. Flag for production port review.

## Verdicts

| tag | arms | cells | cold gm | regs | exact | notes |
|---|---|---|---|---|---|---|
| champh2_probe | c74f_sbx vs PR@b14ec40e1b | 28 | 1.7193 | 0 | 28/28 | GPU6 probe |
| r3a_5f3d | 5f3daaf8 vs PR@b14ec40e1b | 28 | 1.7158 | 0 | 28/28 | GPU6; vs champion: ALL 0.9985, n≥512K activation zone 1.0001 → **WASH, no displacement** (hint filter doesn't pay on radix-scan skeleton) |
| **r3grid09d1** | 09d13c81 vs PR@b14ec40e1b | 865 | **1.7553** | **0** (min 1.009) | **865/865** | 3-shard GPUs0/1/5; anchors med 1.002 all rungs ≤1.045; vs champion gm 1.0428 (coop rungs +6-11%, small-n ≈1.00) → **new composite** |
| r3c_fence | 09d1+threadfence | 28 | 1.5657 | 0 | 28/28 | REJECTED −11% |
| r3d_relacq | 09d1 acq_rel asm | 28 | 1.6142 | 0 | 28/28 | REJECTED −8% |
| r3f_surg | 09d1 surgical rel/acq | 28 | 1.6146 | 0 | 28/28 | REJECTED — ordering cost intrinsic |
| r3g_30e7 | 30e79029 vs PR | 28 | 1.8150 | 0 | 28/28 | GPU1, anchors med 0.991; **vs champion +6.2%** (contiguous-slice scan on top of 09d1 barrier); full grid running |
| **r3grid30e7** | 30e79029 vs PR@b14ec40e1b | 865 | **1.7714** | **0** (min 1.003) | **865/865** | 3-shard GPUs0/1/3; anchors med 1.004; vs champion +5.0%, vs 09d1 +0.67% (net; 118 cells ≤0.99 vs 09d1 = layout noise) → **new composite**. NB ≈ op35 UB reference 1.771 (zero-P3+P4blk relaxation) — at the estimated structural wall |
| **r3gridbecd** | becdc5c7 vs PR@b14ec40e1b | 865 | **1.7848** | **0** (min 1.010) | **865/865** | 6-shard; anchors med 1.000; vs champion +6.6%, vs 30e7 +1.5% net BUT v32 mid-n loses (64k 0.923/128k 0.939/32k 0.956) while pro/flash win big (pro_128k/1024k +13.5-13.8%) |
| r3i_compA | compA probe | 28 | 1.8557 | 0 | 28/28 | subset bias (v32_32k L03/L46 favor becd; 58-layer average favors v30) |
| **r3gridcompA** | compA (becd + k2048-mid-n→30e7) vs PR | 865 | **1.7873** | 1 borderline 0.999 | **865/865** | dispatch verified: v32 32k/64k/128k ≈ 30e7 (vs becd +5.4-8.8%); non-dispatched deltas = run noise ±2% on identical code; splice estimate ~1.81. Borderline pro_64k_L24 (N=16387, identical-code rung; 1.010 in becd grid) → 60-rep adjudication at final acceptance. **compA = ship candidate** |
| ~~r3gridaef3~~ | aef33fac vs PR | 865 | ~~1.9614~~ | — | 865/865 | **CONTAMINATED** (anchor p95 1.542, foreign job on GPUs 2-5 mid-run); structural signal retained: topk_mid<4> heals N=16387 (+18-19%), topk_mid<1> regresses n≈4099 (−11-12%) |
| **r3gridcompB** | compB (aef3−mid<1> ⊕ 30e7) vs PR | 865 | **1.8267** | **0** (min **1.140**) | **865/865** | 8-shard full-node quiet window; anchors med 1.006 p95 1.063; vs compA +1.3%, vs champion +8.1%; weak-rung healed (pro/flash_64k +19%, 32k +8-10% vs compA); NO borderline cells → **new composite / ship candidate** |
| ~~r3b_09d1~~ | 09d13c81 vs PR | 28 | ~~2.3698~~ | — | 28/28 | **INVALIDATED** — foreign job at 100% util GPUs 1-7 during run (pr arm inflated 19→26 µs); my quiet-check echo was unconditional (scripting bug, fixed to gated form). Exactness (load-independent) retained: 28/28. Re-probe pending quiescence |

- **09d13c81** (r2, internal 1.0351): replaces `cudaLaunchCooperativeKernel`
  with regular launch + hand-rolled sense-reversing global barrier (generation
  token ⇒ no per-launch reset), grid sized to co-residency. CAUTION for ship:
  barrier uses relaxed atomics with no __threadfence — memory-ordering risk on
  paper even if exact in practice; if it wins, add fence + re-measure.
- 09d13c81 partial probes (r3b2 GPU5 full-28, r3b3 GPU5 flash-only), per-cell
  anchor-gated (±6% vs champh2 refs): clean-anchor cells show a REAL win vs
  champion — v32_32k 1.19-1.23, v32_256k 1.17-1.18, v32_128k 1.06-1.10,
  pro_128k 1.13-1.18, pro_512k 1.05-1.13, pro_1024k 1.06-1.07,
  flash_128k 1.08-1.11, flash_32k/4k ≈1.00, v32_4k ≈1.00 (single-CTA path
  untouched, as expected). flash_512k/1024k anchors dirty in BOTH GPU5 runs
  (foreign bursts; cand times ≈ champion 0.98-1.00 but unverified).
  → genuine displacement candidate; full-865 verdict queued for quiescence,
  batched with any further round-2 winners.
| **champh2** | c74f_sbx vs PR@b14ec40e1b | 865 | **1.6770** | **0** (min 1.018) | **865/865** | 7-shard GPUs0-6; Bar-1/2/3 denominators; worst cells all N=16387 (graft-rung boundary, 1.02-1.08) |

## Anchor checks

- champh2 vs c74fsbx (old-head grid): per-cell `pr_cold(old)/pr_cold(new)`
  overall median **1.005**, p95 1.058; rung medians 0.995–1.048 (worst
  pro_4k 1.048 — small-n launch noise). No drifted rung. Conclusion: PR head
  b14ec40e1b ≈ e6fdbfac3d on this 865-cell envelope (the 07-20/07-21 P4
  bracket/kb512 commits do not materially move these cells); champion start
  vs current head = 1.6770 (vs 1.6828 on old head), consistent with the
  1.005 anchor shift.
