# op22 ENV (latest-skill 9-arm envelope) — 8-GPU RESUME / HANDOFF PROMPT

> Migration handoff: the 2-GPU dev box (umbriel-b200-047) built + smoke-validated
> the harness; the heavy 18-batch nsys sweep must run on an 8-GPU B200 node.
> The checkout is **NFS-shared**, so every file below is already visible on the
> target node — nothing to copy. Paste the block under the line into a fresh
> Claude Code session ON the 8-GPU node.

---- PASTE BELOW THIS LINE ----

Run the op22 ENV benchmark: a unified **9-arm** nsys pure-kernel sweep on
**latest-skill** GVR performance-envelope synthetic data, then add a new
section to the op22 REPORT.html. Work dir:
`indexer_topk_op_bench/op22_temporal_fixed_hr_bench/` (TensorRT-LLM checkout,
branch `omni/op21-gvr-prod`). All harness files already exist on this NFS
checkout (built + smoke-validated on umbriel-b200-047). Do NOT rebuild them.

## What this measures
9 arms, all reading BYTE-IDENTICAL inputs per cell, all timed in ONE process
on ONE node (no cross-node anchor transfer):
  gvr_cutedsl (BASELINE) · radix_cutedsl · gvr_multicta_cutedsl (PR#15198) ·
  radix_single_cuda · radix_multi_cuda · op27_hls (=gvr_ms_auto HLS-op27,
  OP21_FB_LOGFALSI=1) · op26_r0auto · sglang_v2 (main 2026-07) ·
  flashinfer_topk (0.6.11). fp32 only (external arms are fp32-only); K
  512/1024/2048 = V4-Flash/V4-Pro/V3.2.

Data = LATEST indexer-topk-temporal-synth skill, exact envelope prompt
(bundle_data_env.py SCEN_ENV):
  BEST  = 逐 K 顺风 cfg + target_hr 0.55: flash=aggregate, pro=beta_moderate,
          v32=beta_moderate.
  WORST = beta_shallow + target_hr 0.05 (K-flat adversarial pole).
  SYNTH_POSITIONAL=1, seed 42, steps 1, BS=1 rows (harness replicates to BS).
(NOTE: generation assets calib_/posz_ are byte-identical to GitHub, so the
data is deterministic; the new content vs op22rr best/worst is v32-best cfg =
beta_moderate + a single-node unified 9-arm sweep.)

Test conditions IDENTICAL to the rest of REPORT.html: B200, nsys pure-kernel
GPU time, cold-L2 canonical (512MB evict before each timed call) + warm-L2,
20 cold / 50 warm reps, eager+sync inside NVTX range, cudaProfilerApi window.

## Preflight (30 s)
1. `cd indexer_topk_op_bench/op22_temporal_fixed_hr_bench`
2. `git log --oneline -3` should show the `[op22 env]` harness commit.
3. `python3 -c "import torch, cutlass, flashinfer; print(flashinfer.__version__)"`
   must print 0.6.11 (external arms need it).
4. `cd ../harness && python3 -c "import sglang_v2_op"` must import; `cd -`.
5. GPU preflight: `nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv`
   — idle >50C ⇒ don't time on that GPU (019/035 GPU0, 036 GPU1 are broken-cooling
   nodes). Check co-tenancy by output-file growth, not nvidia-smi.

## Launch (one command — 8-GPU balanced, setsid, resumable)
```
./launch_op22env_8gpu.sh
```
This pre-generates bundles_env/ once, then fans 18 batches across GPU0-7:
GPU0-2 = best {bs,seqlen} K512/1024/2048; GPU3-5 = worst {bs,seqlen}
K512/1024/2048; GPU6/7 = best/worst bs_hugeN (all K). Logs `envrun_gpu*.log`;
per-batch `.done_*` markers under `results_b200_op22env/{best,worst}/`.

Monitor:  `tail -f envrun_gpu*.log` ·
          `ls results_b200_op22env/*/.done_* 2>/dev/null | wc -l`  (target 18)
Est. wall-clock ~1-2 h (bs batches dominate: 84 cells × 9 arms).

### Gotchas (from prior op22 nodes — obey)
- nsys `-c cudaProfilerApi` exits 143 on success; driver has no `set -e`.
- NEVER commit `*.sqlite` / `*.nsys-rep` (they embed env tokens); the driver
  already runs under `env -u GITHUB_TOKEN -u HF_TOKEN`.
- Kill = `pkill -f sweep_op22env; pkill -f drive_nsys_op22env; pkill -f "nsys profile"`
  then re-check ps for respawns (`kill -9 -<PGID>`). `TaskStop` does NOT kill the
  setsid driver tree.
- If a node reclaims mid-batch, the in-flight batch has no `.done` marker →
  just re-run `./launch_op22env_8gpu.sh` (batch-granular resume, re-measures
  the whole unfinished batch in one nsys run).

## Parse + report (after all 18 `.done_*` present)
```
python3 ../op28_ext_topk/parse_op28.py "$PWD/results_b200_op22env"   # reused as-is
python3 update_report_op22env.py                                     # adds §new section
```
- parse_op28.py writes `results.jsonl` per (scenario,sweep) with canonical
  cold-L2 `us` (kernel-sum) + `us_span` (nvtx projection; honest for the
  sglang_v2 2-kernel PDL overlap).
- update_report_op22env.py is DONE + validated (fake-data smoke: 6 seqlen SVG
  charts, 2 geomean tables, 8 KPI cards, idempotent marker-replace, `<script>`
  stays 3). It injects a new REPORT.html §env section: per-K seqlen cold-L2
  latency line charts (static inline SVG — NO plotly/JS), best/worst 9-arm
  geomean-speedup tables, KPI headline cards, en/zh discussion; writes
  op22env_{seqlen,bs}_data.csv. Skeleton-safe: with no results.jsonl it prints
  the run steps and exits 0. Run it AS-IS after parse (no edits needed).
  If you want richer BS-scaling visuals, extend build_section() (the SVG helper
  svg_lines() is reusable).

## QA gates before shipping the section
- exactness: all 9 arms' index sets == torch.topk at BS=1 (kernels already
  414/414 in op22rr/op28; spot-check a few cells with `_smoke_env.py`).
- per-batch anchor sanity: gvr_cutedsl cold µs should track the op22rr real
  column order-of-magnitude for the same (K,N) (absolute µs are node-local;
  ratios are the canonical metric).
- no arm produced all-errors (grep `"error"` in results.jsonl; sglang_v2 +
  flashinfer must have >0 rows at every K).

## State (2026-07-14, umbriel-b200-047, pre-migration)
- Harness DONE + smoke-validated (8/9 arms confirmed build+run on best/worst
  env data; flashinfer JIT was still warming at handoff — verify in preflight).
- bundles_env/ generation validated (best K512 N8192 hr=0.551, Npad==N).
- NOT YET RUN: the 18-batch nsys sweep (needs 8 GPUs). Everything downstream
  (parse reuse + update_report_op22env.py) is DONE + validated — just run.
- Files (all committed): bundle_data_env.py, sweep_op22env.py,
  drive_nsys_op22env.sh, launch_op22env_8gpu.sh, update_report_op22env.py,
  _smoke_env.py.
