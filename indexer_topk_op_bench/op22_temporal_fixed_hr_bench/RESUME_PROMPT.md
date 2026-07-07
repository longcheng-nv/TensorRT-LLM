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

## State (2026-07-07, iter1: W1–W4 DONE on umbriel-b200-040, sweeps launching)
- W1 DONE: gen_bundles.py → 234/234 bundles (152 MB, gitignored), 0 fails,
  hr verified (best .900 / worst .050±.001 / real sampled per layer).
  SEED AMENDMENT (PLAN §2): per-cell seed = 42 + crc32("{K}|{N}") % 1e6
  (constant 42 would collapse the aggregate layer mixture — synthesize()
  draws the layer as the FIRST rng call); exact seed + CLI in each
  bundle's meta.json ("gen_cmd").
- W2 DONE: bundle_data.py (legacy dict shape). NEXT_N=1 ⇒ skill seq_lens
  == bench N*cr EXACTLY — no convention gap (asserted per load).
- W3 DONE: sweep_op22.py + drive_nsys_op22.sh + parse_op22.py; headless
  smoke 45/45 recs 0 errors (incl. N=512K/1M cells).
- W4 DONE: gate_op22.py GATE exact=456 mismatches=0 errors=0 (all 5 ops ×
  3 scenarios × 9 (dtype,K) × N {65536,1048576} × BS {1,16}).
- op21 kernel state: iter12 @f51f50f4da — bench op `gvr_ms_auto`
  (op21_gvr_prod/src/gvr_msc_op.py) is the object under test. The upstream
  PR-1 integration is PAUSED by user decision — do NOT work on
  tensorrt_llm/ integration in this campaign.

## Next
(e) W5 IN PROGRESS: `cd op22_temporal_fixed_hr_bench && OUT=results_b200_op22
    GPU=0 ./drive_nsys_op22.sh 2>&1 | tee ../results_b200_op22/drive.log`
    (scenario-serial real → best → worst; batch-resumable via
    results_b200_op22/<scen>/.done_* markers — just relaunch the same
    command after a node loss). Then `python3 parse_op22.py`; sanity
    `real` ratios vs report/report.html aggregates. Stretch after main:
    SWEEPS=bs_hugeN ./drive_nsys_op22.sh (N {512K,1M} × BS 2–64).
(f) W6 REPORT.html (CSS-only toggles, ZERO <script> — legacy report.html
    has 3, do NOT copy them; inline-SVG charts; record the §2 CLI +
    natural-language generation prompts + per-cell seed policy verbatim).
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
