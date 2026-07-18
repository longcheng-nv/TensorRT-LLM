# Track A2 — distP4 (distributed Phase-4) design spec (iter6)

Target: P4blk (handoff2 + P4 + writeback) = 23-50% on the pr-routed region
(N>=65536, BS 32-128; op35 nsys_oracle_decomp.csv). Kill the handoff2
value-gather; distribute hist build + scatter/writeback across the cluster;
leader keeps only the scalar searches. op35 iter2b sketch, now fully pinned.

## Package
`op36_gvr_rival_7b/variant/gvrpkg36/` = copy of `op35_gvr_round2/variant/
gvrpkg35/` (drop __pycache__). All edits in `top_k/gvr_topk_decode.py`,
marked `# [op36-A2]`. New flag `dist_p4` (default False), requires
`enable_p4_rank_scatter_exact and cluster_size > 1`; asserts if enabled
otherwise. cs=1 and short-row-degrade (do_cluster_sync=False) paths are
UNTOUCHED (old leader path).

## New PTX helpers (next to ld_shared_cluster_f32, same @dsl_user_op style)
- `red_shared_cluster_add_i32(mapped_addr, val)`:
  `"red.relaxed.cluster.shared::cluster.add.u32 [$0], $1;"`, `"r,r"`,
  has_side_effects=True, no result.
- `atom_shared_cluster_add_i32(mapped_addr, val) -> old`:
  `"atom.relaxed.cluster.shared::cluster.add.u32 $0, [$1], $2;"`, `"=r,r,r"`.
Memory model: relaxed atomics ordered by the release `cute.arch.
cluster_arrive()` + acquire `cluster_wait()` — NEVER use
cluster_arrive_relaxed in this path (see memory: relaxed arrive has no
release; stale DSMEM reads cluster by CTA slice).

## SMEM (allocate at cs>1 only, same offset every CTA — mapa requirement)
`s_dp4` Int32[4]: [0]=own cand-min order-key, [1]=own cand-max order-key,
[2]/[3]=scratch (my_prefix/total_capped, then b_star/rank_above bcast, then
sb_star/ra_fine bcast — sequential reuse, each stage separated by a block
barrier after thread0 writes). Order key = f32_order_key(v) ^ 0x80000000
(signed-monotonic form) stored as Int32; inverse for min/max readback:
u = k ^ 0x80000000 (back to unsigned-monotonic), then invert f32_order_key:
s = u ^ (~(u >> 31) | 0x80000000)... SIMPLER: store min/max as raw f32 bits
via two slots each? NO — keep it simple: s_dp4 as Float32[4] is wrong for
atomics; min/max are NOT reduced via DSMEM atomics — each CTA WRITES its own
(min,max) to its OWN s_dp4[0]/[1] as f32 BITS (float_as_int32), and after
sync #1 every CTA mapa-LDs all cs peers' two slots, bitcasts back, and
reduces locally (cs<=8 → 16 lds by thread0). No order-key needed at all.

## Slot map (leader smem, peers access via mapa rank 0)
- counters (atom targets): leader `s_iscalars[4]`=cnt_above, `[0]`=cnt_mid,
  `[1]`=cnt_strad — SAME slots as the existing leader-local code (2301-2303).
- broadcast: leader `smem_hist[256]`=b_star, `[257]`=rank_above (published
  after coarse search, before fine zero of [0:256); kNumBins>=512 always);
  leader `smem_hist[2]`=sb_star, `[3]`=ra_fine (existing convention, fine
  bins dead after fine search).
- per-CTA local cand count: `s_iscalars[5]` (already stable post-P3).

## run_one_row rewiring (cs>1 branch, line ~3805 region of gvrpkg35 file)
```
if const_expr(dist_p4):            # NEW: replaces handoff2-arrive + gather
    (no handoff2 arrive/wait here — sequence has its own syncs)
    phase4_rank_scatter_dist(...)  # ALL CTAs enter (leader + peers)
else:
    <existing handoff2 + leader-only P4, byte-identical>
```
p4_oracle_skip / p4_fused_hist are const_expr-incompatible with dist_p4
(assert in __init__).

## phase4_rank_scatter_dist sequence (do_cluster_sync=True guaranteed here)
own_cnt = min(s_iscalars[5], kC); every CTA already holds cluster total in
s_iscalars[0] BUT the == kK path needs the CAPPED total: thread0 mapa-reads
all peers' s_iscalars[5], caps each at kC, accumulates total_capped and
my_prefix (ranks < own rank; rank = cute.arch.block_idx_in_cluster()); store
to s_dp4[2]/[3]; ALSO thread0 stages own (cmin,cmax) f32-bits to s_dp4[0]/[1]
after a block min/max reduction over smem_keys[0:own_cnt] (reuse the original
warp-staging pattern lines 2088-2110 but into smem_wmin/smem_wmax which are
dead post-P1). Leader additionally zeroes its full smem_hist[kBins].
block barrier.
SYNC1: cluster_arrive(); cluster_wait().
Every CTA thread0: mapa-ld all peers' s_dp4[0]/[1] → cluster bmin/bmax →
store to own s_thr[1]/s_thr[2] (f32 slots, dead post-P3) ; block barrier;
all threads read bmin_r/bmax_r (+1e-6 guard, identical FP math everywhere).
cand_total = s_dp4[3].
CASE cand_total == kK: strided copy own cands to
  output[my_prefix + i] (+values if return_output_values); done (fall through
  to the final cluster barrier OUTSIDE this method — peers/leader all exit).
CASE cand_total > kK:
  all: coarse-bin own cands (inv1 formula verbatim 2142) →
       leader: plain atomicAdd own smem_hist; peers: red_add via mapa.
  SYNC2.
  leader only: 3-step coarse search verbatim (2156-2208) → b_star,
       rank_above; publish to smem_hist[256]/[257]; zero smem_hist[0:256);
       (leader block barriers inside as original).
  SYNC3.
  all: thread0 mapa-ld b_star/rank_above → s_dp4[2]/[3]; block barrier;
       f_lo = bmin_r + b_star/inv1; finv verbatim (2219-2220);
       fine-bin own cands with coarse-bin==b_star test (verbatim math
       2227-2242) → red_add/atomicAdd into leader smem_hist[0:256).
  SYNC4.
  leader only: fine search verbatim (2245-2299) → sb_star→smem_hist[2],
       ra_fine→smem_hist[3]; reset counters s_iscalars[4]/[0]/[1]=0.
  SYNC5.
  all: thread0 mapa-ld sb_star/ra_fine → s_dp4[2]/[3]; block barrier;
       scatter own cands verbatim (2307-2350) EXCEPT the three atomicAdds
       target leader slots (leader: local atomicAdd; peer: atom_add via
       mapa); gmem writes direct from every CTA.
  SYNC6.
  leader only: read cnt_strad = s_iscalars[1]; pad verbatim (2352-2361);
       exact-tail ambiguity check need0 = kK - ra_fine;
       if cnt_strad > need0 > 0  (RARE):
           OLD handoff2 gather loop verbatim (3874-3906) appending peers'
           cands at [own_cnt ...] capped at kC → cand_count_g;
           then OLD exact-tail block verbatim (2375-…) over cand_count_g
           (bmin_r/inv1/b_star/f_lo/finv/sb_star/ra_fine all live).
  (peers fall to the final cluster barrier and keep SMEM alive for the
   rare gather — this is exactly the old-path liveness contract.)
```
DSL constraints: pre-init every var used across dynamic-if boundaries
(Int32(0)/Float32(0)); range_constexpr for cs loops; mapa per element.

## Exactness gates (must pass before any bench)
Unit battery `variant/battery_a2.py` vs torch.topk multiset, fp32, on
K∈{512,1024,2048} × N∈{65536,131072,262144} × BS∈{2,8,64}, cs from
pick_config (must be >1), preIdx from synth hit≈0.5, PLUS forced paths:
- cand_total == kK copy path (craft via K=N? no — accept statistical
  coverage: assert both branches were compiled; == kK is data-rare, covered
  by full-grid folded checks),
- ambiguous exact-tail: rows with >need duplicated values at the boundary
  (e.g. many exactly-equal values around rank kK) — verify vs pr arm output
  equality AND torch multiset,
- degrade path regression: short rows (N<65536 at cs>1 → do_cluster_sync
  False) unchanged vs pr arm.
Then harness folded exact across the sweep (every cell).

## Bench protocol (measurement discipline binding, PLAN.md)
Arm `gvr_a2` in ops_op36.py: gvrpkg36 GvrTopKKernel at the PRODUCTION launch
contract (launch/pick_config — clone _build_a0 shape WITHOUT A0 flags;
dist_p4=True only). Screening OPS="gvr_pr,sglang_v2,gvr_a2" on the 9 batches
covering the routed region first (flash/pro/v32 × 64k/128k/256k + 512k/1024k
flash/pro), 8-way; anchors at 1/3; then full grid if it wins; verdict ≤2-way.
Attribution axis: a2/pr per cell (nsys us). Ship rule: replace pr in the
Track-B dispatch table iff a2 ≥ pr everywhere routed (zero-regression rule).
