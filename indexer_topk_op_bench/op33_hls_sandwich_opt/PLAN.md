# op33 — HLS-op27 sandwich optimization (borrow op29-HBE + sglang-v2, no copy)

## Objective triple (omni-kernel)
- **incumbent**: `op27_hls` = `gvr_ms_auto` @ op27 HEAD (GvrSandwichKernel, op18/msc lineage:
  P1 hint-gather+stash → P1b 256-hist rank-quantile M=4 thresholds → block_count_collect_multi
  ONE fused pass → phase3_sandwich direct-write M0<K winners + band → phase4 band-only snap/rs).
  Env: `OP21_FB_LOGFALSI=1 OP27_K2048_TAIL=1` (defaults). A/B ALWAYS vs THIS.
  Invoke: `harness/sweep_nsys.build_call("gvr_ms_auto", K, dtype, N, BS, cr, logits, preidx)`.
- **rivals (track, DON'T copy)**: op29 HBE-noB (1-pass sample-column, issue-bound, deleted tier-B),
  sglang_v2 (register-resident short-row + warp/register tie-select ≤32 zero-barrier).
- **envelope**: BS=1 · fp32 · K∈{512,1024,2048} · all N (seqlen sweep 4K→1M). At BS=1 fp32
  gvr_ms_auto dispatches: K512/1024 short-mid N → single-CTA sandwich (gvr_ms); large N → cluster
  (gvr_msc C=8); K2048 n≥196608 → cluster.
- **verdict_axes**: [worst, real, best].
- **TARGET (user)**: beat op27_hls by **average 30%** on this envelope. ship_rule: geomean ≥1.30
  vs op27_hls on the envelope AND exactness green (tie-aware vdiff=0, 3 tracks) AND dispatch ≤3.

## Hard constraints
1. Stay within HLS-op27's GENERAL framework (256-hist rank-quantile placement + fused-collect +
   sandwich fast-write). BORROW insights from op29-HBE/sglang, do NOT copy their code/skeleton.
2. New behavior via flag/subclass; op27_hls baseline byte-identical, one-revert recoverable.
3. Measure: cold-L2 + CUDA-graph (L1 screen) → nsys ×3-median (ship arbiter). L1 event is NOISE
   at N≤16K BS=1 (op32 lesson) — nsys mandatory for verdicts.

## Candidate directions (to probe, ledger-checked)
- D1 warp/register band tie-select (sglang INSIGHTS-P3): sandwich band is small at BS=1; replace
  phase4_band snap/rank-scatter barriers with ≤32 warp-ballot / ≤128 register ranking (zero block
  barrier). P4 is barrier-bound (op#7 rank-scatter precedent). HIGH promise.
- D2 reduce fused-pass M columns where they don't earn keep (op29 tier-B deletion analogue) —
  but M=4 is cheap (count_ge_multi_bench: M4=1.15-1.46×) so headroom limited; check per-N.
- D3 shrink P1b 256-hist fixed cost at short-N (the thing that makes op27 lose to op26 there):
  coarser bins / sampled placement. GATED by op32-F5 (log-binning worse; linear near-optimal) —
  so NOT a binning change; a COST change (fewer bins / cheaper extraction).
- D4 register-resident fused pass at short-N (sglang) — GATED by op32-F1 (register-for-traffic
  dead, dram 0.06% at BS=1). Only revive if it enables barrier removal, not traffic saving.

## Red lines (falsification ledgers)
- op32/FALSIFIED.md: F1 register-resident-for-traffic (L2-trap), F4 barrier-cheapen-via-redundancy
  (SLOWER, redundant work > barriers saved), F5 exponential binning (worse than linear).
- gvr_topk_falsification_history: Opt-L (fuse collect w/ online slot-reserve), P4-internal reseed,
  smem-resident (op15), M≥4 explicit... BUT op27's fused-collect is NOT Opt-L (threshold pre-known).
- op27_hls_allge_probe/: the K2048 tail-ladder history + all_ge mode analysis.
