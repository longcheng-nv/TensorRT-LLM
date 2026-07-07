# op22 RESUME PROMPT — paste this into a fresh Claude Code session (B200)

> Keep this file CURRENT: update State/Next at every commit. It is the
> disaster-recovery handoff for the op22 campaign.

---- PASTE BELOW THIS LINE ----

Continue the op22 GVR-vs-rivals benchmark campaign on temporal-synth
fixed-hit-rate data, in
`indexer_topk_op_bench/op22_temporal_fixed_hr_bench/` (TensorRT-LLM
checkout, branch `omni/op21-gvr-prod`, start HEAD f51f50f4da). Read, in
order:
1. `op22_temporal_fixed_hr_bench/PLAN.md` — goal, 3 scenarios (best/worst/
   real with fixed hit rates 0.90/0.05/sampled), grid, work items W1–W7,
   protocol gotchas, decisions already made. Follow it; do not re-litigate
   §7 decisions.
2. `.claude/skills/indexer-topk-temporal-synth/SKILL.md` — the data
   generator (validated, 5 gates PASS ×3 models); its motivating study is
   `synth_vs_real_validation/SYNTH_VS_REAL_VALIDATION.html`.
3. `indexer_topk_op_bench/report/report.html` + `report/gen_report.py` —
   the report style/data-method to mirror (§1 seqlen BS=1, §2 BS-scaling).

## State (2026-07-07, campaign not yet started)
- PLAN.md authored; NO code, NO bundles, NO measurements yet.
- op21 kernel state: iter12 @f51f50f4da — bench op `gvr_ms_auto`
  (op21_gvr_prod/src/gvr_msc_op.py) is the object under test. The upstream
  PR-1 integration is PAUSED by user decision — do NOT work on
  tensorrt_llm/ integration in this campaign.
- Harness to reuse: harness/sweep_nsys.py + drive_nsys_full.sh (nsys
  cudaProfilerApi, warm `w|` + cold `c|` NVTX ranges, resumable batches),
  sweep_op21._build_op21_call, report/parse_nsys_full.py.

## Next (= PLAN.md §4, in order)
(a) W1 gen_bundles.py — generate 1-row bundles, scenarios best(beta_deep,
    hr .90)/worst(beta_shallow, hr .05)/real(aggregate, sampled), models
    v4flash/v4pro/v32 ↔ K512/1024/2048, dtypes fp32/bf16/fp16,
    N 4K..1M, --seed 42; gitignore bundles/ FIRST.
(b) W2 bundle_data.py adapter (same dict shape as harness/synth_data
    .get_bundle; seq_lens = BENCH convention N*cr).
(c) W3 sweep_op22.py + drive_nsys_op22.sh (5 ops: gvr_ms_auto,
    gvr_cutedsl, gvr_multicta_cutedsl, radix_cutedsl, sglang_streaming;
    N_SEQ + {524288,1048576}; OUT=results_b200_op22/<scenario>).
(d) W4 exactness pre-gate BEFORE timing (sorted-set criterion for GVR —
    output order is atomicAdd-nondeterministic, op21 iter12 LEARNINGS).
(e) W5 run scenario-serial real → best → worst; parse; sanity `real` vs
    report.html ratios.
(f) W6 REPORT.html (CSS-only toggles, zero <script>; record the §2 CLI +
    natural-language generation prompts verbatim for reproducibility).
(g) W7 commit per iteration `[op22 iterN]`, `git commit -s`, trailers
    Made-with + Co-Authored-By Claude; update PLAN/RESUME each commit.

## Environment / recovery (NFS-shared; a node timeout loses only /tmp)
1. `cd` this checkout; `git log --oneline -1` should show f51f50f4da or a
   later `[op22 …]` commit.
2. `python3 -c "import torch, cutlass"` must pass (plain python3 on
   umbriel B200 nodes; op-bench needs torch+cutlass only, NOT trtllm).
3. GPU preflight: `nvidia-smi --query-gpu=index,temperature.gpu
   --format=csv` — idle >50C ⇒ don't time on that GPU (019/035 GPU0 are
   broken-cooling nodes → use GPU1 there).
4. Co-tenancy: check by output-file growth, not nvidia-smi (namespace-
   blind sandbox), before any cold-L2 batch.
5. nsys: `env -u GITHUB_TOKEN -u HF_TOKEN`; never commit *.sqlite/
   *.nsys-rep; `nsys -c cudaProfilerApi` exits 143 on success (no set -e).
6. Absolute µs don't transfer across nodes — per-cell rival RATIOS are the
   canonical metric; `real`-scenario ratios should roughly reproduce
   report.html's aggregate columns (sanity anchor).
