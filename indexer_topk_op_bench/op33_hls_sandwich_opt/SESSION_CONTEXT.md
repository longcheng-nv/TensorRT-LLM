# op33 SESSION CONTEXT — full carryover (2026-07-13)

Saved per user directive "保存所有上下文到本地". Everything a fresh session needs.

## Lineage of this campaign
User chain this session: query op26 → dissect count<K fast-write in op21-sandwich & sglang-v2 →
op32 (register-resident op26 short-row) CLOSED NO-SHIP → user pivots to op33: optimize HLS-op27
itself, borrow op29-HBE/sglang, target avg +30% at BS=1 fp32 K512/1024/2048.

## Incumbent = op27_hls (GvrSandwichKernel), what it does (gvr_ms_op.py / gvr_msc_op.py)
5-phase single-CTA (BS=1 short-mid N) pipeline:
1. P1 `phase1_stats_stash` / `phase1_preidx_stats`: gather K prev-topK values, stash to smem.
2. P1b `phase1b_rank_quantile`: 256-bin hist over the K stashed prev-topK VALUES (band [pmin,pmax]),
   extract **M=4 quantile rungs** = ORDER STATISTICS at qfracs. op27 K2048 tail = (0.75,0.45,0.048);
   K512 ship=(0.92,0.45,0.048); K1024 ship; stock=(0.75,0.5,0.25). All thresholds known BEFORE scan.
3. `block_count_collect_multi`: ONE fused full-N pass — count ≥ each of M thresholds AND append
   v≥thr[pred_col] into per-thread smem slots (slot cursor = per-thread count; NO online reserve →
   dodges the falsified Opt-L because threshold is pre-known).
4. `phase3_sandwich`: packed prefix-sum (direct<<16|band); v≥thr0 direct-writes idx to output[0:M0)
   (M0<K guaranteed winners), thr1≤v<thr0 → band smem. done=1 needs band≤kC (not M1≤kC).
5. `phase4_band_snap` / `phase4_band_rank_scatter`: refine ONLY band, select k_rem=K-M0.
Dispatch (gvr_ms_auto @ gvr_msc_op.py:1647): BS=1 fp32 K512/1024 short-mid N → single-CTA gvr_ms;
n≥131072(or ≥65536&K≥1024) → cluster gvr_msc C=8; K2048 n≥196608 → cluster.

## Measured facts (this session, nsys B200 sm_100, cold-L2, BS=1 fp32)
- op27_hls vs op26_r0auto short-N (report §1 seqlen BS=1): op27 SLOWER 1.0-1.65× (op26 faster) at
  N≤32K; op27 WINS only hugeN ≥512K (report data). So op27's headroom is largest at short-mid N.
- op26 base ~9.7µs at N8192 K512; NCU: single-CTA dram 0.06% (idle), issue 15%, warps 25% → the
  regime is LATENCY/ISSUE-bound, not memory-bound. Same regime applies to op27 sandwich at BS=1.
- count_ge_multi_bench/REPORT.html: M-thresh count is memory-bound cheap (M2 free, M4=1.15-1.46×,
  M6-8 up to 2.4×; small-N favors M6-8). So M=4 fused count is NOT the cost.

## Borrow (analysis only, NO copy)
- **op29 HBE-noB** (op29_gvr_hbe/, iter12 SHIPPED, "1st in-tree arm to beat sglang_v2" @N≥65536):
  64×64 coalesced sample → single column A (rank 2×rS_K) → 1-cmp/elem fused collect (smem bufA +
  global spillA) → cnt_a≥K → candidate mini-hist → miss fallback. iter11: the fused pass is
  ISSUE-BOUND (81-84% issue, ~16 inst/candidate); deleting the never-triggered tier-B insurance
  column (~6·K/row) was the whole "+188µs" K-proportional win. LESSON borrow: minimize inst/elem in
  the fused pass; delete columns that don't fire.
- **sglang_v2** (op28_ext_topk/ops/sglang_v2): register-resident short-row (row in registers, hist,
  scatter-from-registers, ZERO 2nd kernel); layered tie-select by candidate count (≤32 warp ballot
  = zero block barrier; ≤2048 = 4-round radix); coarse fp16 12-bit hist + fp32-boundary collect.
  LESSON borrow: warp/register band select (zero-barrier) for small bands; register residency IF it
  removes barriers (not for traffic — op32-F1).

## op32 falsification ledger (short-row op26, DO NOT re-propose in op33)
F1 register-resident-for-traffic DEAD (L2-trap, dram 0.06%). F2 threads-raise WASH. F3 warp-reduce
WASH. F4 barrier-cheapen-via-all-thread-redundancy SLOWER (+16%; redundant work > barriers saved;
rank-scatter's cut-barriers-wins does NOT transfer because it also cut WORK). F5 exponential/log
256-hist binning WORSE than linear (host ablation; linear already 1-3% count error, near-optimal;
rungs sit in prev-topK body not exp tail). CORRECTION: "M=4 = per-element tax" is WRONG — M-count is
memory-bound cheap; op26 ships M=2 due to single-round admission economics + iter-floor + hist-build
fixed cost, NOT count cost.

## Harness
- Invoke op27_hls: `sys.path harness/ + op22_temporal_fixed_hr_bench/`;
  `from sweep_nsys import build_call`; `build_call("gvr_ms_auto",K,dtype,N,BS,cr,logits,preidx)`
  with env `OP21_FB_LOGFALSI=1 OP27_K2048_TAIL=1`. Data: `bundle_data_rr.get_bundle(scen,K,fp32,N)`.
- Exactness: tie-aware value-multiset (sorted selected == sorted torch.topk, cardinality K).
- Bundles: best/worst regenerated, real=original op22; fp32 N present ≥8192 (N4096 missing for K2048).
- GPU: 8× B200 idle (34-39°C, healthy); use GPU1. cold-L2 = 256MB flush before each launch.
- nsys: `env -u GITHUB_TOKEN -u HF_TOKEN nsys profile -c cudaProfilerApi`; gitignore *.nsys-rep/*.sqlite.

## Env/ops hygiene (memory)
- L1 event timing USELESS at N≤16K BS=1 (launch floor ~10µs) — nsys mandatory for verdicts.
- pkill -f can hit own wrapper shell; kill by pid. Long runs: setsid. TaskStop doesn't kill trees.
- nsys sqlite embeds env tokens — env -u before nsys; gitignore before first commit.
- git commit --no-verify scoped to bucket dir (repo-wide pre-commit hook times out).
