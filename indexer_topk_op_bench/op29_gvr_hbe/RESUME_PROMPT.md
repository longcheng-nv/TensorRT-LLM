# op29 RESUME — GVR-HBE campaign (updated 2026-07-13, iter10 done @2a5931fca2)

## 1-minute context
Goal: beat sglang_v2 across the op22 fp32 grid, op27 = production incumbent.
WINNING DESIGN (iter9, SHIP-CANDIDATE in its domain): hint-free HBE =
64x64-chunk coalesced row sample -> cand-targeted columns (sample-rank
2*rS_K / 8*rS_K) -> histogram-FREE fused single pass (2 cmps/elem, smem bufs
+ global spill) -> validity by count (cnt_a>=K) -> b* from candidate-only
smem mini-hist -> tiered resolve; miss falls back to stock 2-pass.

## Proven verdicts (nsys same-batch, gate 216/216, fork parity 1.000)
- ENGAGED domain (guard: K<=1024 && N>=131072, streaming regime i.e. BS>512):
  262144x1024 **1.46-1.50x**, 262144x2048 **1.36-1.40x**, 131072x1024
  **1.06-1.12x** over sglang_v2, ALL scenarios incl worst (scenario-invariant).
- Outside guard: parity 0.99-1.01 (zero regression).

## Falsification ledger highlights (do NOT re-propose; see FALSIFIED.md)
min-hint predictor; hint x sample max(); strided single-element sampling
(DRAM-burst ~half a pass); INLINE FULL HISTOGRAM in fused pass (issue-bound
wall — NCU 545MB@1.4TB/s vs rival 1.02GB@4.2TB/s); N<=65536 engagement
(fixed overheads); K2048 w/ capA=2K (+188us unattributed K-cost).

## Next steps (priority order)
1. NCU-attribute the K2048 HBE cell (262144x1024): suspects = universal spill
   (cand target 2K=8192 > capA 4096), resolve/tie scaling. Then either
   capA=4K@occ1 A/B or tighter col (rank 1.2*rS_K) + capB catch.
2. 131072 residual (1.09 vs 1.47): shrink fixed phases — merge the two
   find_thresholds (one pass over sample hist can yield both ranks), or
   sample 2048 instead of 4096.
3. Cluster-path HBE (BS<=512, N>cluster_floor): same trick inside
   TopKCluster::forward (skip Phase1+DSMEM all-reduce on count-valid hit).
4. Short rows N<=16K: register-path already 1-read; win only via P5 (occ/vec).
5. Full-grid sweep: add gvr29 to op28-style harness (3 arms incl anchors),
   all idle GPUs, K512/1024 first; then REPORT.html arm via update_report
   pattern (must extend the op28 last-writer chain).
6. Production shape: the guard means HBE is a THIRD dispatch tier of
   gvr_ms_auto (>=131K high-BS) — integration decision needs the user.

## Preflight
- git log -1 >= 2a5931fca2; branch omni/op21-gvr-prod.
- GPU blacklist recheck (was: 0 co-tenant, 4/5 co-tenant 158GB; 1/2/3/6/7 free).
- nsys/ncu: env -u GITHUB_TOKEN -u HF_TOKEN. Builds in ../_build/gvr29.
- Pilot relaunch: see launch block below (byte-exact).

```bash
cd indexer_topk_op_bench/op29_gvr_hbe
setsid env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=<g> \
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o results/pilot/pilot_<scen> -f true \
  python3 scripts/pilot_op29.py --scenario <scen> \
  --out results/pilot/pilot_<scen>.jsonl > results/pilot/pilot_<scen>.log 2>&1 &
```

## Gotchas
- enable_smem_spilling illegal with dyn smem; TieValue.idx uint32;
  find_threshold(rank, TOTAL_IN_HIST, smem); pilot jsonl archived per-iter
  under results/pilot/iter<N>/.
