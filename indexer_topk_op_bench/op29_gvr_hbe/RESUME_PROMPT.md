# op29 RESUME — GVR-HBE campaign (updated 2026-07-13, iter12 SHIPPED)

## 1-minute context
Goal: beat sglang_v2 across the op22 fp32 grid, op27 = production incumbent.
WINNING DESIGN (iter12, SHIPPED in-op): hint-free HBE-noB = 64x64-chunk
coalesced row sample -> ONE cand-targeted column A (sample-rank 2*rS_K) ->
1-cmp/elem fused single pass (smem bufA + global spillA) -> validity by count
(cnt_a>=K) -> b* from candidate-only smem mini-hist -> resolve; miss falls
back to stock 2-pass (never observed). Tier-B insurance column REMOVED
(iter11 NCU: ~16 inst per band element in an issue-bound pass, fired 0/18;
kColB compile-key + col_b flag retained for A/B, default OFF).

## Proven verdicts (nsys same-batch, gate 324/324, fork parity ~1.000)
Guard: !cluster && streaming && N>=65536 — ALL K (K cap removed).
- 262144x1024: 1.71-1.73 (K512/K1024), 1.62-1.64 (K2048)
- 262144x2048: 1.60-1.75 · 131072x1024: 1.46/1.48/1.33-1.36 (K512/1024/2048)
- 65536 cells (BS 64-2048): 1.03-1.13 ALL K — falsified domain REVIVED
- N=32768: guard excludes (would be 0.85-1.00); no-force parity 0.99-1.02.
- Scenario-invariant (real/best/worst within ~1%).
- iter11 K1024 guard hole confirmed (B-on 0.84 @131072x1024!) and FIXED (1.48).

## Falsification ledger highlights (do NOT re-propose; see FALSIFIED.md)
min-hint predictor; hint x sample max(); strided single-element sampling;
INLINE FULL HISTOGRAM in fused pass (issue-bound wall); tier-B insurance
column (0/18 fires, ~6*K*16 inst/row — revival needs real per-row-variance
miss-rate data); N<=32768 engagement (fixed overheads, re-scoped iter12).

## Next steps (priority order)
1. Cluster-path HBE (BS<=512, N>cluster_floor): same trick inside
   TopKCluster::forward (skip Phase1+DSMEM all-reduce on count-valid hit).
2. Short rows N<=16K: register-path already 1-read; win only via P5 (occ/vec).
3. Full-grid sweep: add gvr29(noB) to op28-style harness (3 arms incl
   anchors), all idle GPUs, K512/1024 first; then REPORT.html arm via
   update_report pattern (must extend the op28 last-writer chain).
4. Production shape: guard means HBE is a dispatch tier of gvr_ms_auto
   (N>=65536 streaming) — integration decision needs the user.
5. (opt) real-world tier-A miss-rate study on varied-row captures (bench rows
   are identical per cell; B-off insurance argument rests on cnt_a>=1.33*K
   margins seen in replay).

## Preflight
- git log -1 >= iter12 commit; branch omni/op21-gvr-prod.
- GPU blacklist recheck (this node: all 8 B200 cool+free at iter12 time).
- nsys/ncu: env -u GITHUB_TOKEN -u HF_TOKEN. Builds in ../_build/gvr29.
- GVR29_FORCE_HBE=1 = diagnostic guard bypass (N<65536 probes).
- Pilot relaunch (byte-exact):

```bash
cd indexer_topk_op_bench/op29_gvr_hbe
setsid env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=<g> GVR29_FORCE_HBE=1 \
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o results/pilot/pilot_<scen> -f true \
  python3 scripts/pilot_op29.py --scenario <scen> \
  --out results/pilot/pilot_<scen>.jsonl > results/pilot/pilot_<scen>.log 2>&1 &
```

## Gotchas
- enable_smem_spilling illegal with dyn smem; TieValue.idx uint32;
  find_threshold(rank, TOTAL_IN_HIST, smem); pilot jsonl archived per-iter
  under results/pilot/iter<N>/ (iter12 also has force_pilot_table.txt).
- parse_pilot.py --force-export re-exports sqlite from the CURRENT rep:
  archive jsonl+table BEFORE relaunching nsys onto the same -o path.
- 4-arm pilot arms: sglang_v2 | gvr29_hbe (B-on) | gvr29_hbe_nob | gvr29_off.
