# op19 sandwich two-threshold GVR top-K — iteration log

**Goal (user 2026-07-02):** full GVR optimization bucket around the SANDWICH
two-threshold idea: from one M-ary ladder pass (op18 block_count_ge_multi),
pick thr0 (count M0 < K → those M0 elements are GUARANTEED top-K members,
direct-write them in P3) and thr1 (count M1 ∈ [K,kC], the accepted threshold);
P4 then refines ONLY the band [thr1, thr0) of M1−M0 candidates for the
remaining K−M0 slots. Maximize M0, minimize M1, keep P2 rounds low.
Variants: single-CTA (extend op18 src/gvr_mt_op.py) and multi-CTA cluster
(extend op17 portfolio). Target: ALL 720 B200 cells (K×dtype×N×BS grid of
report/report.html) beat single-CTA gvr_cutedsl; average speedup ≥1.5×.

## 📋 Resolved Plan (omni-kernel Phase 0/1)
- **Operator** — DSv4 indexer top-K decode, GVR family (P1 preIdx band → P2
  threshold search → P3 collect → P4 refine), memory-bound selection/reduction.
- **Shape grid** — K{512,1024,2048} × dtype{fp32,bf16,fp16} × N{4096..262144}
  × BS{1..2048} = 720 cells (K2048/N4096 excluded), synth bundles seed=42.
- **Reference** — torch.topk exactness gate: uniq=K, valdiff=0 (set-based;
  output order is free — P4 writeback order is atomic/arbitrary already).
- **Baseline** — harness/gvr_cutedsl_op.py single-CTA (report column), B200
  numbers in report/bs_data.csv (hw=B200).
- **Target GPU** — B200 sm_100, 148 SMs. **GPU 1 ONLY while op18's session
  drives nsys on GPU 0** (co-tenancy detected 2026-07-02 03:48 UTC).
- **SOL%/roofline frame** — single-CTA floor = 2 full-N passes (1 M-ary count
  + 1 collect) vs baseline ~2.5-3.5 passes + cand-linear P4; multi-CTA opens
  the small-N floor (op17: cluster 1.21-1.67× nsys).
- **Language priority** — CuTe DSL only (family constraint: extends vendored
  GvrTopKKernel; op16 falsified separate-kernel approaches).
- **Protocol** — cold-L2 512MB-evict + CUDA-graph + cudaEvent median
  (harness/sweep.py `_time_both`); nsys pure-kernel for positive claims;
  exactness gate before ANY timing. Per-iter commit `[op19 iterN] ...`
  (op19 paths only — branch shared with op18's live session).

## Strategy roster (Arch-Strategy for this op family)
- **Strategy-A** — single-CTA sandwich: op18 M-ary P2 + snapshot of the
  (thr0, col0) pair + fused P3 (direct-write ≥thr0 to gmem output at
  prefix positions; band → smem) + runtime-k band-snap P4 with histogram
  range seeded [thr1, thr0).
- **Strategy-B** — multi-CTA cluster sandwich: op17 portfolio; the CTA owning
  thr0 direct-writes its M0 elements CONCURRENTLY with the winner CTA's
  band refine (thr0 column lives in that CTA's own smem — zero DSMEM copies).
- **Strategy-C** — per-(K,N,BS) dispatch combining A, B, op18-auto, baseline;
  high-BS is A's home turf (aggregate working set ≫ L2 → baseline's 2nd
  pass is truly cold → pass-collapse pays full price; op18 only swept BS=1).

## Priors (falsification-aware, MUST respect)
- op16: two-threshold via SERIAL secant = tax-bound (extra full-N passes);
  Scheme X free-peel lost b/c band stayed wide (M0 far below K) + M0-wide
  2-pass partition cost. op19 differs: thr0/M0 come FREE off the same M-ary
  ladder pass; band collect replaces (not augments) normal collect; P4 gets
  runtime-k + seeded histogram range.
- op18 iter2: the L2 trap — at BS=1 baseline's 2nd P2 pass is L2-resident
  (~5K cyc); never count it as a full pass. CDF-aware fracs (place_mode=3)
  fixed it: iter3 avg 1.08-1.10×, max 1.35×.
- op18 iter2: P4 snap is placement-sensitive, NOT cand-linear alone: 15.3K
  cyc @cand1821 (loose thr) vs 12.6K @cand532 (refined thr) vs 7.2K @cand669
  (M8 uniform). Tight bracket [thr1,thr0) + tiny k_rem is the fix hypothesis.
- op12: P4 ~45-50% of kernel time at small N but partially barrier-floor-bound;
  op17 iter1b: P4 cand-linear with ~7500cyc floor.
- op17: single-CTA M-way compares in P2 cost ≤1.0× at M=16 (ALU tax) — keep
  M ≤ 8; M2 free, M4 ~×1.25, M8 ~×1.77 per pass (count_ge_multi bench).
- GVR falsification history (gvr_phase_timing/): 12 ruled-out paths; do not
  revisit Opt-L fuse, P1 self-loop, Opt-F, cluster-DSM-highBS.

## Honest ceiling note (to re-verify in iter0 sim)
±50% avg over gvr_cutedsl across ALL 720 cells was proven out of reach for
PURE single-CTA (op12/op17). The path to ≥1.5× avg: (a) sandwich P4-collapse
everywhere, (b) multi-CTA at BS≤16 (op17 already 1.18× gm alone), (c) high-BS
large-N pass-collapse (unmeasured, L2 trap absent), (d) dispatch. If the
ceiling math says the target is unreachable, report the gap honestly.

---
## Iter 0 — 2026-07-02 — offline sandwich ceiling sim: the pair is nearly free

`scripts/sim_sandwich.py` on the real fp32 bundles (results/sim_sandwich_fp32.jsonl),
replaying op18 M-ary rounds (CDF-aware f3 fracs) + tracking the sandwich pair
(thr1: tightest count>=K -> M1; thr0: max count<K -> M0) — zero extra passes:

| pass budget | typical band | typical M0/K | weak cells |
|---|---|---|---|
| 1 (M8R1f3) | 76-561 (K512), 105-2259 (K2048) | 0.83-0.98 where straddled | K1024 ALL N: M0=0 (fracs not straddle-aware); K512 4/8/65K, K2048 16/262K: M0=0 |
| 2 (M8R2_b64) | 10-777 | 0.55-1.00 | K1024 mid: band ~430-650 |
| 3 (R3_b64) | 10-97 | 0.95-1.00 | none |

Findings:
1. With 2-3 ladder rounds, 95-100% of top-K is DIRECT-WRITABLE and P4's
   working set collapses from 1.3-5.2K cand to a band of tens. k_rem tens.
2. op18's f3 fracs were optimized to pin M1 tight, NOT to straddle K ->
   R1 M0=0 at K1024 all-N + 5 more cells. New lever: straddle-fracs
   (re-run optimize_fracs targeting one frac each side of v_K). R1-straddle
   is the high-BS play (every pass cold there; minimize passes).
3. M must dispatch on (N, BS): M8 cold-pass tax x2.7 @262K forbids M8 at
   large-N BS=1 -> M2/M3 multi-round (refine rounds are L2-warm ~5K cyc);
   M8R2 at small N; M3/M4 R1-straddle at high BS.
4. High-BS is the make-or-break for the >=1.5x avg target (420/720 cells
   BS>=32, never measured in op18): aggregate working set >> L2 kills the
   L2 trap -> baseline's 2.5 P2 passes go truly cold vs sandwich's 1 M-pass
   + P3 + tiny P4 -> theoretical ~1.55-1.75x. Validate FIRST after exactness.

Next (iter1): Strategy-A kernel bring-up + exactness gate.
## Iter 2 — 2026-07-02 — first high-BS A/B (M4R2p3): WASH — the ACTIVE-SET L2 trap

results/ab_highbs_fp32.jsonl (M4R2b64p3): avg 0.96x (0.73-1.28). Root cause
of the miss vs the iter0 hypothesis: at high BS what must exceed L2 is the
ACTIVE working set (~148 SMs x min_bpm CTAs x rowbytes ~= 400 rows), not the
whole batch. N<=32K fp32: 400 x 128KB = 51MB < L2 -> baseline's extra passes
are STILL L2-warm; and R2 doubles the sandwich's COLD passes (round 2 is not
warm at high BS). Only K2048/262K won (1.22-1.28x). => high-BS config must be
R1 (one pass), and the pass-collapse pays only at N >= ~131K (fp32).

## Iter 3 — 2026-07-02 — straddle-fracs: R1 single-pass sandwich everywhere

`scripts/optimize_straddle_fracs.py` (5 seeds, 1024-pt lambda grid): fracs =
{0 anchor, l1 = max frac w/ count>=K all seeds, l0 = min frac w/ count<K all
seeds, + linspace(l1,l0) self-sorting inner points}. results/straddle_fracs.json:
d2=0 everywhere; worst-seed M4: band 129-946 M0/K 0.57-0.86; M8: band 38-510.
Kernel place_mode=4 loads it (same codegen as 3).

High-BS re-run with M4R1p4 (results/ab_highbs_fp32.jsonl): avg 1.22x, and the
262K row = **1.16-1.71x** (K512 1.16/1.32, K1024 1.17/1.33, K2048 1.55/1.71)
— the pass-collapse hypothesis CONFIRMED where the active set spills L2.
Remaining losses: K2048/32K 0.81-0.87 (M4 tax on a warm-extra-pass regime;
try M3R1p4) and K512/32K wash 1.01-1.02. All cells exact at every BS so far.

Running: BS=1 config sweep (R2 refine configs, results/cfg_bs1_fp32.jsonl) +
BS=2048 R1 config sweep (results/cfg_bs2048_fp32.jsonl) -> dispatch table.
## Iter 4 — 2026-07-02 — Strategy-B cluster sandwich: EXACT at G=4/8/16

`src/gvr_swc_op.py` (GvrClusterSandwichKernel <- GvrSandwichKernel): G slots at
straddle-aware fracs {0 anchor} + linspace(l1,l0,G-1); DSMEM count share; all
ranks compute pair (r1 tightest>=K, r0=r1+1 <K); winner DSM-copies r0's
per-thread count column into smem_ptcnt_up (barrier #2 covers smem lifetime)
then reuses Strategy-A phase3_sandwich + phase4_band_snap verbatim. op17 D0
fixed en passant: no-pair && M1>kC -> done=2 retry-shrink (no silent cap).
Exactness: 20/20 fp32 cells x G in {4,8,16} (GPU0, op18 idle-window).
Perf A/B queued behind the GPU1 config sweeps.
## Iter 7 — 2026-07-02 — P3 hot-loop restructure: the dual-branch tax, killed

Bisect (sw_enable flag, G16 BS1): swcOFF/op17 = 0.91-0.94 (push+release-fence
~6-9%) and swcON/swcOFF another 6-15% -> the sandwich P3's if/elif per-element
chain was breaking the 4-way LSU pipeline (op18's P3 has ONE rare outer
branch). BS=1 sweeps confirmed machinery-not-fracs (M2R2p3b64 == M2R2p4b64
both ~0.77 at 262K).

Fixes: (1) single rare outer `if vj>=thr1` with nested classify (all 3 scan
loops); (2) both prefix sums packed into ONE warp scan ((up<<16)|band, bounds
M0<=2048, band<=6144 < 2^16); (3) cluster sync selectable use_push (st+release
arrive) vs ld-copy (+relaxed barrier #2).

After fix (G16, BS1): swc/op17 0.94-1.01 (was 0.78-0.88); ld-copy slightly
ahead of push at 262K (1.010 vs 0.976). All exact 20/20 x 4 smokes.
Data points pre-fix: BS1 oracle gm 1.086 (M2R1p4 dominant; large-N <1 traced
to this tax); BS2048 oracle gm 1.227 (defer fixed K2048 small-N 0.75->0.91+).
Re-sweeping both BS regimes + full cluster A/B with the fixed kernel.
## Iter 8 — 2026-07-02 — iter7-kernel measurement round (fp32)

**BS=1 (results/cfg_bs1_fp32.jsonl, 80/80):** oracle gm 1.147, min 1.012 —
**zero cells below baseline** (pre-iter7: gm 1.086, several <1). M2R1p4
dominates (K512/262K 0.949 -> 1.171; K2048/262K 1.352).

**Cluster A/B (results/ab_cluster_fp32.jsonl, 60/60, ld-copy):** vs baseline
gm 1.189, min 0.993, max 1.467, ALL exact; vs op17 gm 1.010 (29/60 wins) —
sandwich cluster >= op17 net, and beats single-CTA at many BS<=16 cells
(K1024/32K 1.44 vs 1.28; K2048/262K 1.47 vs 1.35).

Running: BS=2048 + BS mid (16/64/256) fp32 re-sweeps (GPU1); bf16/fp16
transfer validation of fp32 straddle fracs (GPU0). build_dispatch extended to
ingest cluster rows as cfg=cluster<G>.
## Iter 13 — 2026-07-02 — nsys pure-kernel validation: all positives CONFIRMED

8 representative fp32 cells, cold-L2 inside cudaProfilerApi window, 20-60
iters, cuda_gpu_kern_sum median (results/nsys/, results/nsys_drive.log):

| cell | dispatch | event | nsys |
|---|---|---|---|
| K512/16K BS1 | cluster16 | 1.28 | 1.368 |
| K512/262K BS1 | cluster16 | 1.18 | 1.165 |
| K1024/32K BS1 | cluster16 | 1.44 | 1.517 |
| K1024/262K BS16 | cluster4 | 1.28 | 1.336 |
| K2048/262K BS16 | cluster4 | 1.47 | 1.531 |
| K512/262K BS2048 | M4R1p4 | 1.416 | 1.419 |
| K1024/32K BS2048 | M4R1p4 | 1.446 | 1.463 |
| K2048/262K BS2048 | M4R1p4 | 1.852 | 1.856 |

nsys >= event on 7/8 (cluster cells largest gap: launch overhead sits in the
event number, as in op17 §4.1); single-CTA high-BS cells match within 0.3%.
The fp32 fullgrid claims stand on pure-kernel evidence.
## Iter 14 — 2026-07-02 — K2048 high-BS rescue attempts: M1 falsified, union-buffer REVERTED

(a) M1R1p4 (zero-tax single pass at l1): 0.78-0.98 at K2048 midN highBS —
LOSES even with baseline-identical pass structure. The deficit there is not
config-tunable; baseline dispatch stands for those ~10 cells.
(b) smem union (didx aliasing hist, -8KB at K2048) REGRESSED 0.918 -> 0.745
at K2048/16K/BS2048. Bisect chain: iter7-worktree repro 0.916; HEAD 0.748;
direct union-vs-separate A/B: union 0.745 / separate 0.893. Root cause:
-8KB smem lifts residency 2 -> 3 CTAs/SM in that regime and 3-resident is
SLOWER (L1/LSU contention). REVERTED to separate buffers (kernel comment
records the measurement); post-revert re-check: 0.924 (matches iter7-era).
fp32 fullgrid numbers were measured pre-union -> remain valid.
Lesson: "free" smem savings are not free — occupancy changes are a first-
class perf variable in BOTH directions.
