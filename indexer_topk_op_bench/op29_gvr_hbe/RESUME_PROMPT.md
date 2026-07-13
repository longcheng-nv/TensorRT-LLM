# op29 RESUME — GVR-HBE campaign (updated 2026-07-13, NODE-MIGRATION handoff)

## 1-minute context
Campaign goal: beat sglang_v2 (op28's new fastest arm) across the op22 fp32
grid; op27 = production incumbent; omni-kernel v2 protocol.
STATE: iter12 SHIPPED + REPORT ARM DONE + ALGORITHM DOC DONE + HBE-C DESIGN
COMMITTED. Everything is committed; no in-flight runs; safe to resume on any
B200 node (NFS-shared checkout).

- iter12 ship @8135a4de55: HBE-noB (tier-B column removed, col_b=False
  default), guard = !cluster && streaming && N>=65536, ALL K. Engaged
  1.03-1.75x over sglang_v2; gate 324/324.
- Report arm @97edc7edc2: 906 cells, 3 arms same-batch, anchor drift med
  1.0020/p90 1.0156 (node 073 -> orig 037/044 scale); ship-rule 9/9 slices
  geomean >= sglang_v2; ZERO cells lose >5%. In
  op22_temporal_fixed_hr_bench/REPORT.html (updater = update_report_op29.py,
  append-on-top last-writer — re-run it after any older updater).
- ALGORITHM.html @9a0be348e2: bilingual CSS-only algorithm doc.
- DESIGN_HBEC_HINT_LADDER.md @cd3c7e5cbd: next campaign design (hint-ladder
  cluster single-pass, BS<=512 domain = 41% of report grid; preIdx revival;
  HLS Step-3 lineage; ledger-checked; probe plan inside).
- NAMING: op30 is TAKEN (parallel 10-arm re-test campaign) — HBE-C = op31.

## Priority queue
1. HBE-C rung-0 CRUX (host, ~1h): hint-ladder placement replay on 27 op22rr
   bundles (+ real captures): tightest-rung cand counts / bracket rate /
   miss rate per (scenario,K,N). GO line: E[passes] <= ~1.2, miss <= ~10%.
   Extend scripts/replay_hbe_cand.py; spec = DESIGN_HBEC_HINT_LADDER.md §6.
2. HBE-C rung-2 microbench: DSMEM M×8-scalar reduce vs 4096-bin all-reduce
   latency at BS=1 (decides small-N BS=1 engagement).
3. HBE-C rung-3 kernel (tier-5 behind flag in gvr29), gate 3-track, pilot
   (131072..1048576) × BS {1,16,64,256,512}, nsys same-batch 3 arms.
4. Short rows N<=16K (P5); sub-65536 hint tier (P3, small stakes).
5. Production integration decision for tiers 4+5 (USER).

## Preflight on a NEW node (do all before any measurement)
- cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM
  (NFS — same checkout everywhere); branch omni/op21-gvr-prod;
  git log --oneline -3 must include cd3c7e5cbd (HBE-C design).
- Concurrent-session check: another session works op26/op30 in THIS repo —
  do NOT touch their dirs; check `git log -5` for their fresh commits and
  `find indexer_topk_op_bench -newermt "-15 minutes" -name "*.log"` for live
  runs before claiming GPUs (nvidia-smi is namespace-blind; poll file growth).
- GPU health: nvidia-smi temps — idle >50C = blacklist that GPU (known bad:
  b200-019 GPU0, b200-035 GPU0, b200-036 GPU1).
- Build: JIT auto-rebuilds from NFS sources on first import
  (indexer_topk_op_bench/_build/gvr29). Smoke:
  `cd indexer_topk_op_bench/op29_gvr_hbe && CUDA_VISIBLE_DEVICES=<g> \
   GVR29_FORCE_HBE=1 python3 scripts/gvr29_op.py` -> "SMOKE EXACT".
- ANCHOR: absolute us from node 073 do NOT transfer. Any new measurement =
  same-batch arms (gvr29 + sglang_v2 + gvr_cutedsl anchor); quote ratios only.
- nsys/ncu always `env -u GITHUB_TOKEN -u HF_TOKEN`; never commit
  *.sqlite/*.nsys-rep; long runs via single `setsid ... > name.log &` line;
  stop = pkill triple + 30s respawn recheck.

## Byte-exact launch templates
```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op29_gvr_hbe
# pilot (iter-style, 4 arms):
setsid env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=<g> GVR29_FORCE_HBE=1 \
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o results/pilot/pilot_<scen> -f true \
  python3 scripts/pilot_op29.py --scenario <scen> \
  --out results/pilot/pilot_<scen>.jsonl > results/pilot/pilot_<scen>.log 2>&1 &
# report-grade full-grid shard (3 arms, batch-resumable via .done_*):
setsid env OUT=results_b200_op29 GPU=<g> SCENARIOS="<scen>" KS="<ks>" \
  bash scripts/drive_nsys_op29.sh > op29_gpu<g>.log 2>&1 &
```

## Gotchas
- enable_smem_spilling illegal with dyn smem; TieValue.idx uint32;
  find_threshold(rank, TOTAL_IN_HIST, smem).
- parse_pilot --force-export re-exports sqlite from the CURRENT rep: archive
  jsonl+tables BEFORE re-launching nsys onto the same -o path (per-iter
  archive under results/pilot/iter<N>/).
- update_report chain: any OLDER updater re-run erases the gvr29 arm —
  re-run update_report_op29.py afterwards.
- Bundles: op22_temporal_fixed_hr_bench/bundle_data_rr (NFS, no re-gen
  needed). Raw sweep root results_b200_op29/ is untracked-on-purpose.
