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

## State (2026-07-07 ~13:4xZ, iter2-pre MIGRATION CHECKPOINT off umbriel-b200-049)
- W1–W4 DONE (iter1 @2dac88d7f7); iter1.5 checkpoint @78e40c6945.
- W5 MEASUREMENT: MAIN GRID 54/54 DONE + PARSED (results.jsonl fresh for
  all 3 scenarios). Node split: b200-040 = real 18 + best 16;
  b200-049 = best bs_K2048_{bf16,fp16} + worst 18 (+hugeN below).
  bs_hugeN: real 9/9, best 9/9 DONE; worst 5/9 at checkpoint (driver
  auto-redoes the in-flight batch on resume).
- ✅ MECHANISM RESOLVED (was the ⚠️ headline signal; full detail
  MECH_FINDINGS.md, artifacts mech_check_iters.py/.jsonl,
  mech_crossover.py): tie-density/non-convergence FALSIFIED (replay 162/162
  converged). Cost = extra full-row re-scans, a PURE function of preIdx
  (crossover: swap preIdx ⇒ 40↔240 µs regardless of logits). hr→GVR-speed
  NON-monotone: hr .90 poisons init (pmean ≈ top-K median → undershoot →
  1-3 re-scans); boundary misses (real/worst) seed ≈ K-th value → ev1.
  op21 msc has TWO fallback triggers, both observed in isolation:
  refine-driven (best K2048: 236 vs 28.6 µs) and candidate-band slot
  overflow (worst K2048 N=1M: 105 µs at ev1, cand=4318≈kC; K512 real
  crossover 186 vs 124 µs). Each fallback = leader 1-CTA full-row recount
  (~95 µs @N=1M fp32, gvr_msc_op.py:1096).
- MAIN-GRID VERDICT (cold geomeans, rival/op21, >1 ⇒ op21 faster):
  real: cutedsl 1.244/1.157/1.130 (anchor ✓), radix 1.089/0.895/0.844,
  sglang 1.186, multicta ≈1.0. best: op21 loses to ALL (radix
  0.766/0.584/0.557; even cutedsl 0.844). worst: radix 1.021/0.818/0.778.
  hr-sensitivity t_worst/t_best: GVR family 0.72-0.82 (worst FASTER),
  radix/sglang flat 1.00-1.01. op21 wins ONLY on real; each stress
  scenario trips one msc trigger.
- W6 generator UPDATED: gen_report_op22.py now has §7 mechanism section
  (crossover table + replay summary + build-time worst-prediction check),
  corrected §1/§6 prose (tie-density claim removed), two-node meta line.
  Smoke-built on partial data: 4.3 MB, 0 <script>.
- op21 kernel object unchanged: gvr_ms_auto @f51f50f4da; upstream PR-1
  integration remains PAUSED — bench-only.

## Next (on the NEW B200 node, same NFS — nothing to copy, /tmp loses only
## disposable task logs)
1. Preflight: temps <50C idle (019/035 GPU0 broken-cooling → GPU1 there);
   co-tenancy by output-file growth; python3 -c "import torch, cutlass".
2. Finish stretch grid (auto-skips 23 done batches, redoes in-flight):
   `cd indexer_topk_op_bench/op22_temporal_fixed_hr_bench && OUT=results_b200_op22
   GPU=0 SWEEPS=bs_hugeN ./drive_nsys_op22.sh 2>&1 | tee -a
   ../results_b200_op22/drive_hugeN.log`  (~4 batches ≈ 30-40 min)
3. `python3 parse_op22.py` → `python3 gen_report_op22.py` (REPORT.html;
   verify prints 0 <script>) → review TL;DR/§6/§7 text, check the §7
   build-time worst-prediction line reads sensibly.
4. If worst hugeN absolute µs matter for a figure: worst is all b200-049 +
   the 4 remaining hugeN batches from the NEWEST node — add that hostname
   to the report meta line (edit gen_report_op22.py meta string).
5. W7 commit `[op22 iter2]` (`git commit -s`, trailers Made-with +
   Co-Authored-By Claude Fable 5); update PLAN/RESUME State to CLOSED.

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
