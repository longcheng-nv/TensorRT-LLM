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

## State (2026-07-07 ~06:40Z, iter1.5 MIGRATION CHECKPOINT off umbriel-b200-040)
- W1–W4 DONE + committed iter1 @2dac88d7f7 (bundles 234/234 152 MB
  gitignored; hr verified .900/.050/sampled; SEED AMENDMENT per-cell
  seed = 42 + crc32("{K}|{N}") % 1e6, see PLAN §2; bundle_data NEXT_N=1
  ⇒ seq_lens == N*cr asserted; gate 456/456 exact).
- W5 MEASUREMENT PARTIAL (all on NFS results_b200_op22/, safe):
  * real 18/18 batches DONE + PARSED. SANITY PASS: op21 vs gvr_cutedsl
    fp32 1.198 / bf16 1.105 / fp16 1.078 vs op21-campaign anchor
    1.249/1.091/1.055 (−4.1%/+1.3%/+2.2%). vs radix fp32 1.089,
    bf16 0.892, fp16 0.839 (16-bit weakness as known).
  * best 15/18 batches done at checkpoint (in-flight batch is auto-redone
    on resume: driver deletes any un-.done batch's jsonl+rep). worst 0/18.
  * bs_hugeN NOT started (the queued waiter died with the node —
    relaunch it manually after main grid, see Next).
- ⚠️ HEADLINE SIGNAL (pending verification): `best` (beta_deep, hr .90)
  makes the WHOLE GVR family much SLOWER, not faster — partial geomeans
  vs radix fp32 0.805 / bf16 0.634 / fp16 0.579; K2048 fp32 N=1M: op21
  236 µs vs real-scenario 28.6 µs, radix flat ~20 µs. Same-layer (L22)
  best-vs-real also ~2× slower ⇒ high hr / deep marginal → tie-dense
  boundary → GVR count-convergence refine blowup (consistent with known
  GVR undershoot non-convergence). VERIFY before reporting as fact:
  (1) does `worst` (hr .05) come out FAST? (2) count refine iterations
  on best vs worst bundles via harness/count_gvr_iters.py (check its op
  compatibility first) — artifact-vs-mechanism discriminator.
- W6 generator READY: gen_report_op22.py smoke-built 3× on partial data
  (0 <script>, bilingual CSS-only, inline SVG; auto TL;DR + sanity note,
  §6 extreme-cell findings, bs_hugeN folded into §2 view).
- op21 kernel object unchanged: gvr_ms_auto @f51f50f4da; upstream PR-1
  integration remains PAUSED — bench-only.

## Next (on the NEW B200 node, same NFS)
1. Preflight: temps <50C idle (019/035 GPU0 broken-cooling → GPU1 there);
   co-tenancy by output-file growth; python3 -c "import torch, cutlass".
2. Resume main grid (auto-skips done batches, redoes the in-flight one):
   `cd indexer_topk_op_bench/op22_temporal_fixed_hr_bench && OUT=results_b200_op22
   GPU=0 ./drive_nsys_op22.sh 2>&1 | tee -a ../results_b200_op22/drive.log`
3. Then stretch grid: `OUT=results_b200_op22 GPU=0 SWEEPS=bs_hugeN
   ./drive_nsys_op22.sh 2>&1 | tee ../results_b200_op22/drive_hugeN.log`
4. Mechanism check for the best-scenario signal (see ⚠️ above).
5. `python3 parse_op22.py` → `python3 gen_report_op22.py` (writes
   REPORT.html; verify 0 <script>) → review findings text.
6. W7 commit `[op22 iter2]` (`git commit -s`, trailers Made-with +
   Co-Authored-By Claude Fable 5); update PLAN/RESUME. NOTE absolute µs
   differ across nodes — if resuming mid-scenario mixes nodes, per-cell
   ratios remain valid but note the node change in the report; `best`
   was 15/18 on b200-040, the remaining 3 best batches + all worst will
   be from the new node. Record new hostname in the report meta line.

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
