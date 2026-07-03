# op20 iterations

Protocol: every iter = tier_bench run (in-run 3-way: gvr_x vs gvr_cutedsl base vs
radix_cutedsl rival, cold-L2 CUDA-graph median) + entry here + git commit.
Primary metric = rival/x per cell (≥1.0 = fastest); secondary x/base.
Priority: tier1 fp32 K512/1024 > tier2 fp32 K2048 > tier3 16-bit.

## Iter 0 — 2026-07-03 — baseline (op19 verbatim)

**Strategy**: `gvr_sw_auto` copied unchanged; harness bring-up.
**Implementation**: `src/gvr_x_op.py` (= op19), `scripts/tier_bench.py`.

Smoke (GPU0, solo, 4 cells): exact 4/4; x/base gm 1.125; rvl/x 0.751 (N4K) – 1.103 (N65K).
Full tier1 (84 cells): `results/iter0_tier1.jsonl` — running.
Known holes (report loss-map, fp32): N≤32K BS≤256 (need ~1.22–1.29×),
N=262K BS≤16 (need ~1.69×; in-run spot 0.63).

**Bottleneck**: P4 snap loop = per-distinct-value stepping (mean 3.7–5.5 full-band
scans w/ fmin/fmax reduces, max 15) after a 1/256 histogram start.

**Next**: iter1 = level-2 sub-histogram refinement in P4.

## Iter 1 — 2026-07-03 — P4 level-2 sub-histogram (D2 part 1)

**Strategy**: between the level-1 256-bin locate and the exact snap loop, add one
band pass that (a) counts candidates ≥ bin-hi (fresh, so approximate level-1
binning can't break invariants), (b) re-histograms the target bin into 256
sub-bins, (c) moves thr to the sub-bin edge → snap starts at 1/65536 resolution,
0–1 iterations instead of 4–15. Guard: band ≥ 512 (small bands: snap already cheap).
Snap loop unchanged = exactness authority.

Smoke (GPU1, concurrent w/ GPU0 baseline — ratios valid, abs µs noisy; 12 cells):
exact 12/12. x/base at the small-N holes: N4096 1.11–1.12 (iter0: ~1.00),
N16384 1.12–1.24, N65536 BS64 rvl/x 1.47. Still losing N4096 (rvl/x 0.77–0.85).

Full tier1 (GPU0, solo): pending baseline completion.

**Next**: full-tier1 A/B vs iter0; then D3 (P1 fold ~2.2–2.7µs) or D1
(ladder-interpolation → tighter thr1 → smaller band) for the remaining N4K gap.

### Iter 1 FULL RESULT (GPU0 solo, 84 cells) — FALSIFIED as always-on
exact 84/84, but kernel iter1/iter0 gm **0.963** (min 0.881, max 1.063);
fastest-vs-rival 59→54. Sporadic wins only (N65536 BS1-4: +2.7-6.3%).
**Root cause of the miss**: the "snap = 4-15 full-band scans" datum came from the
BASELINE kernel's P4 (report iters_data); op19's sandwich P4 starts from a tight
[thr1,thr0) band where level-1 alone converges snap in ~1-2 iters — level-2's
extra band pass + 2 barriers is a tax with nothing to save (op16-pattern).
GPU1 smoke "improvement" was concurrent-run noise → never trust cross-GPU smoke
for accept/reject; full solo A/B only.
**Action**: revert src to iter0 (level-2 kept in git history); iter2 = measure
op19's OWN phase budget at the losing cells before touching code again
(clock64/nsys on N4096/N8192 BS1-64 and N131K/262K BS1-4).

## Iter 2 — 2026-07-03 — attribution + GVR-internal mc routing (D5 step 1)

**Attribution (probe_variants.py + mc probe, GPU0 solo)**:
- N262K BS<=4: ALL existing variants lose big (best swcG4 33.4 vs radix 24.2);
  M-column cost is linear-in-N (M2 37 -> M6 50 @262K) => threshold-parallel
  cluster (swc) keeps O(N) per CTA. STRUCTURAL.
- mc (PR#15198, DATA-parallel cluster_size<=4): 131K BS1 20.9 vs radix 24.7
  (1.18x WIN); 262K 26.1-29.5 vs 24.7-25.1 (0.85-0.95). Confirms chunking is
  the lever; C=4 is not enough at 262K.
- N131K holes were mostly a dispatch mis-pick (cluster16 26.7 vs swcG4 24.6).
- Small-N (4-8K): config-INSENSITIVE ~13-15us floor across all M/G vs radix
  10.4-12.9 => fixed-cost wall (P1 + barrier chain + P4), needs kernel surgery.

**Change**: gvr_sw_auto learns cfg="mc" (routes to gvr_multicta_cutedsl, a GVR
kernel); dispatch_table_fp32 rekeys 18 entries (K512/1024 x {131K BS1-8,
262K BS1-16}) to mc. dispatch_table_fp32.json.pre_iter2 = backup.

Smoke (GPU0 solo, 8 hole cells): 131K 0.85-0.89 -> 0.95-1.10 (K512 flips to
win); 262K 0.62-0.63 -> 0.83-0.93. Full tier1: results/iter2_tier1.jsonl.

**Next**: iter3 = chunked-cluster sandwich kernel (extend data-parallel to
C=8/16 + sandwich band P3/P4) to push 262K past 1.0; iter4 = small-N fixed cost.

## Iter 3 — 2026-07-03 — mcC8 routing at 262K/131K (D5 step 2)

**Change**: cfg "mcC<G>" parsing in gvr_sw_auto; dispatch: 262K BS1-8 + K1024
131K BS1-8 -> mcC8 (probe: C8 = K512-262K parity 24.4 vs 24.5, C16 regresses
via DSM merge). **Lesson re-learned**: extended mcC8 to BS16 keys WITHOUT
probing -> 262K BS16 collapsed 0.62/0.79 (mc-auto picks smaller cluster at
high BS); reverted those 2 keys to mc-auto, confirmed 1.257/1.135. Never route
a dispatch key to a config not probed at that BS bucket.

**State after iter3 (composite)**: exact 84/84; ~62/84 fastest; losses:
16 small-N cells (N4-8K, 0.74-0.88, fixed-cost wall) + K1024 262K BS1/4
(0.88-0.91) + 3 parity-noise cells (0.983-0.996).

**Next**: iter4 = small-N P2+P3 fusion (ladder thresholds known up-front =>
collect-at-loosest-column in the SAME N-scan; valid where falsified Opt-L was
not, because secant needed converged thr first) + P1 gather trim.
K1024-262K residual -> chunked-sandwich (iter5).

## Iter 4 — 2026-07-03 — fused P2+P3 (found half-written after node loss; validated + dispatch re-tune)

**Recovery**: node interruption left iter4 code uncommitted in src/gvr_x_op.py
(+266 lines, no validation data). Smoke first: exact 18/18 cells x3 perturbed
inputs, 0 bad => code was complete and correct.

**Change**: block_count_collect_multi — during the (single) R==1 ladder pass,
also append every v >= thr[pred_col=l1] into per-thread smem slots
(slot_cap=max(8,kC/nt)); phase3_from_slots replaces the P3 full-N rescan with
a per-thread slot walk (prefix-sum via packed up|band counters). Overflow
(per-thread l1 count > slot_cap) or best-col < l1 => classic P3 fallback.
Auto-gate: R==1 & p4 & bs<=NUM_SMS. cfg suffix f/nf forces on/off (bare cfg
= auto-gate).

**HYPOTHESIS FALSIFIED, LEVER REDIRECTED**: small-N wall (N4-8K) is fuse-
INSENSITIVE (~14.8us flat) — the wall is P1+barrier fixed cost, NOT the P3
rescan. But at large N the fusion is decisive: the P3 cost stops scaling
with N (slot walk ~ candidates), so (a) iter4 auto-gate tier1: fastest 60->65,
big wins at 131K/262K BS16/64 R1p4 cells (1.22-1.58x); (b) iter4b re-tune of
all M*R1p4 keys BS<=64 ({M2,M4,M6}x{f,nf} per exact bucket): 15 keys ->
explicit fused cfgs, and fusion SHIFTS BEST M DOWN (4->2 on 4 keys: looser
thresholds are cheap when P3 ~ candidates, and M2's wider band no longer
hurts). Gains scale with N: 8K ~1.05x, 32K ~1.11x, 131K ~1.23x, 262K ~1.43x.

**Acceptance (iter4b full tier1)**: exact 84/84; kernel gm 1.034 vs iter3a
(max 1.590 @1024-262K-BS16); x/base gm 1.249; rival/x gm 1.323; fastest
64/84. Losses: 16 small-N wall + K1024-262K BS1/4 (0.909/0.934) + 4 near-
parity. Data: results/iter4_tier1.jsonl (auto-gate), iter4b_retune.log,
iter4b_tier1.jsonl; table backup .pre_iter4b.

## Iter 5 — 2026-07-03 — fusP4T4 routing at 131K/262K BS1-4 (the K1024-262K residual)

**Change**: cfg "fusP<P>T<T>" in gvr_sw_auto routes to op17-v2
gvr_portfolio_fusion (P partition-slices x T threshold-slots in one cluster —
the queued "chunked-cluster sandwich" already built and nsys-validated in
op17 D1). Probe at the exact buckets (red line): fusP4T4 beats mc/mcC8 on all
8 low-BS large-N keys, 1.05-1.15x; BS16 collapses (49-64us, bs*P over-
extension) => NOT routed; fusP8T4 cannot launch (P*T=32 > 16-CTA cluster cap).
Dispatch: K{512,1024} x N{131072,262144} x BS{1,4} -> fusP4T4
(backup .pre_iter5).

**Acceptance (full tier1)**: exact 84/84; all 8 keys improved, e.g.
1024-131072-1 23.0->20.8us (rival/x 1.079->1.169); the two hole cells
1024-262144 BS1/4: 0.934->0.991, 0.909->0.953 (kernel -8%/-5%; residual is
the in-run rival's run-to-run variance — probe showed both > 1.0). No
regression elsewhere (iter4b/iter5 gm 1.006). Composite: x/base gm 1.255,
rival/x gm 1.345, fastest 65/84.

**State after iter5**: losses = 15 small-N wall cells (N4-8K, 0.78-0.88,
P1+barrier fixed cost — needs kernel surgery, config-insensitive) + 4 near-
parity (0.95-1.00). Next: iter6 = small-N fixed-cost surgery (P1 gather trim
via sampled stats; barrier-chain shortening; or accept the wall and close
tier1 at ~65+15-parity/84).

## Iter 6a — 2026-07-03 — small-N wall attribution (scoping, no code change)

**Probe 1 (mc smem-cache at N4-8K)**: FALSIFIED — mcAuto/mcC1/mcC2 all
16.0-22.4us vs swM2f 14.2-16.4 (radix 11.8-14.4). The mc secant chain is
longer than sandwich's ladder; whole-row smem cache does not compensate at
small N.

**Analysis**: gap to radix is ~2.3us and the wall is CHAIN LATENCY, not
bandwidth: P1 gather is ONE parallel L2 round-trip (K loads spread over
threads) => subsampling preIdx (K/4 stats) shortens nothing. The serial
chain is P1(gather L2 + reduce + bcast) -> ladder(scan L2 + reduce +
leader) -> P4(slot walk + snap) with 6-8 block barriers; radix has fewer
serial stages.

**iter6 design (queued)**: smem-resident sandwich for N<=8192 — cooperative
row->smem bulk load FIRST (no dependency), then P1 gathers FROM SMEM
(~30cy vs ~200cy L2), ladder counts from smem, P4 collects from smem. Removes
every post-load L2 round-trip from the critical chain. smem budget at N8192
fp32: row 32KB + slots 2*kC*4B + band buffers — fits 1 CTA/SM. Implementation:
smem-path variants of phase1/block_count_collect_multi/phase3_from_slots that
take a smem row view (addrspace differs, _load_fp32 hard-codes gmem).

## Iter 6 — 2026-07-03 — smem-resident sandwich FALSIFIED; small-N wall ACCEPTED, tier1 CLOSED

**Decisive pre-probe (warm vs cold)**: N8192 warm gap vs radix = 0.01us (pure
cold-memory story) BUT N4096 warm gap = +2.05us and sw warm time is FLAT
10.25us across N4-8K — a phase-chain floor. Cold-warm delta-of-deltas is only
~0.5us => at most ~0.5-1.8us was ever recoverable from memory tiers.

**Change (kept, default OFF)**: smem_row_elems knob — whole row bulk-loaded
to smem CONCURRENTLY with P1's gather, ladder reads smem
(block_count_collect_multi_smem); cfg suffix s/ns, auto-gate default False.

**Result**: exact 12/12 cells x3 inputs, perf NO-OP (14.2-15.6us with or
without smem; radix 11.7-14.5). The wall is the serial phase chain
(P1 -> thr place -> ladder -> best-col -> P3-slots -> P4 snap, ~12+ barriers
+ leader-serial segments) vs radix's shorter pipeline — a structural GVR-
lineage floor at N<=8K, NOT a memory-tier or config problem (3rd falsified
lever after mc-smem-cache and P1-subsampling).

**TIER1 CLOSED** per red line: 65/84 fastest + 4 near-parity (0.95-1.00),
rival/x gm 1.345, x/base gm 1.255, exact 84/84. The 15 losing cells are all
N4096/8192 (0.78-0.88) = accepted structural wall. Next phase: tier2 (fp32
K2048), tier3 (16-bit) via the same probe->dispatch->fullgrid protocol.
