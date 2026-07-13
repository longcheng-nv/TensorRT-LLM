# op29 RESUME — GVR-HBE campaign (updated 2026-07-13, iter2 gate green)

## 1-minute context
Goal: beat sglang_v2 (op28 arm, new fastest in op22 REPORT) across the full
fp32 grid, from op27 as production incumbent. Core design = HBE: hint-quantile
dual-column speculative collect fused with the full histogram in ONE DRAM pass
(rival needs 2); miss -> in-kernel redo = rival parity. See PLAN.md + crux
verdict in ITERATIONS.md iter1.

## State
- iter1 CRUX GO @9286086177 (scripts/crux_hint_{bin,quantile}.py).
- iter2 kernel DONE @eb59922b46: src/gvr29 (fork of ops/sglang_v2 + HBE
  streaming path, flag use_hbe, dyn-smem bufs); wrapper scripts/gvr29_op.py
  (build dir ../_build/gvr29). randn smoke 27/27; bundle gate
  scripts/gate_op29.py 216/216 (real hints, HBE on+off).
- iter3 pilot IN FLIGHT: scripts/pilot_op29.py — 3 arms (sglang_v2 rival /
  gvr29_hbe / gvr29_off parity) x {K512,K2048} x 9 HBE-engaged (N,BS) cells,
  nsys same-batch, scenarios real/best/worst on GPU2/6/7.
  Artifacts: results/pilot/pilot_<scen>.{nsys-rep,jsonl,log}.
  Parse: scripts/parse_pilot.py -> ratio tables.

## Preflight checklist
- git log -1 must show iter >= 2; branch omni/op21-gvr-prod.
- GPU blacklist (2026-07-13): GPU0 (co-tenant 100%), GPU4/5 (co-tenant
  158GB) — RECHECK nvidia-smi, occupancy changes hourly.
- nsys must run with `env -u GITHUB_TOKEN -u HF_TOKEN`.
- Builds cached in ../_build/{gvr29,sglang_v2}; if lock-stale, rm the lock.

## Next steps (disjoint)
1. Parse pilot -> iter3 verdict (target: hbe/rival >= 1.2 on engaged cells,
   gvr29_off/rival ~= 1.0 fork parity; miss telemetry via K2048 cells).
2. If GO: extend HBE to level-3 streaming rows + cluster path (iter4);
   short-row register hint trim (iter5); dispatch-floor tuning (iter6).
3. Full-grid sweep via op28 harness pattern (add gvr29 ops to ops_ext-style
   build_call), all idle GPUs, then REPORT arm + anchor transfer.

## Launch commands (byte-exact)
```bash
cd indexer_topk_op_bench/op29_gvr_hbe
setsid env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=<g> \
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o results/pilot/pilot_<scen> -f true \
  python3 scripts/pilot_op29.py --scenario <scen> \
  --out results/pilot/pilot_<scen>.jsonl > results/pilot/pilot_<scen>.log 2>&1 &
```

## Gotchas
- enable_smem_spilling pragma is ILLEGAL in dyn-smem kernels (ptxas fatal) —
  HBE kernel omits it; other kernels keep it.
- TieValue.idx is uint32 (not int32).
- find_threshold(rank, TOTAL, smem): 2nd arg must equal the histogram's total
  count (topk for the hint mini-hist).
- HBE engages ONLY when !cluster_eligible && maxseq > 16384: BS>512 all N, or
  N <= cluster_floor (32768@BS<=15 / 65536@BS<=512). Cluster cells run the
  baseline fork path.
