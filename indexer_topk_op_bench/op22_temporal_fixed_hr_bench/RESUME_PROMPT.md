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

## State: CLOSED (2026-07-07 ~14:2xZ, iter2 final on umbriel-b200-019)
- CAMPAIGN CLOSED at iter2. Full grid 81/81 batches DONE (main 54 +
  bs_hugeN 27), parsed (3834 recs × 3 scenarios, 0 errors), REPORT.html
  final (iter2.1 restyled to report/report.html conventions: dark theme,
  header cards, latency+speedup pairs, §3 Full-data CSV+table; still
  CSS-only 0 <script>, 8.4 MB). No measurement ran on b200-019 — the
  b200-049 session finished all worst-hugeN batches at 13:49Z before
  handoff (the "worst 5/9" note below was stale at write time).
- W1–W4 DONE (iter1 @2dac88d7f7); iter1.5 checkpoint @78e40c6945.
- W5 MEASUREMENT: ALL DONE. Node split: b200-040 = real 18 + best 16;
  b200-049 = best bs_K2048_{bf16,fp16} + worst 18 + ALL bs_hugeN 27/27.
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

## Next: NONE — campaign closed. Deliverable = REPORT.html (bilingual,
## CSS-only). Final TL;DR (cold geomean, rival/op21, >1 ⇒ op21 faster,
## now INCLUDES hugeN cells): real cutedsl 1.244/1.151/1.128 + radix
## 1.089/0.873/0.827 (fp32/bf16/fp16); best radix 0.721/0.531/0.508;
## worst radix 0.977/0.757/0.724. op21 wins only on real-scenario data;
## §7 mechanism (msc refine + slot-overflow fallbacks) explains both
## stress losses; build-time worst-prediction confirmed (t_worst/t_real:
## 1-CTA 0.769 / multi-CTA 0.791 <1, ms_auto 1.142 >1). Follow-on levers
## live in op13 (cheaper-P2) and op21 RESUME (PR-1 integration, PAUSED).

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

## ADDENDUM 2026-07-08: GVR-mCTA arm backfill — DONE (see final UPDATE below)
- Goal: add "GVR multi-CTA (cuteDSL, PR#15198)" (`gvr_multicta_cutedsl`) to
  the re-tested §1/§2 dataset (all op22rr cases, same conditions).
- sweep_op22rr.py now has ARMS_EXTRA + OP22RR_ARMS env filter (committed?
  check git log). Runs = 2 arms (mc + co-located gvr_cutedsl ANCHOR).
- Launch (both resumable via .done markers, OUT=results_b200_op22rr_mc074):
  GPU0: DTYPES=fp32 all scen, then DTYPES=fp16 SCENARIOS=worst  (36 batches)
  GPU1: DTYPES=bf16 all scen, then DTYPES=fp16 SCENARIOS="real best" (45)
  env: OP22RR_ARMS="gvr_cutedsl,gvr_multicta_cutedsl"; logs mc074_gpu{0,1}.log
- After: python3 parse_op22_cached.py ../results_b200_op22rr_mc074
  then python3 update_report_mc.py  (anchor-transfer onto orig baseline scale,
  patches REPORT.html D blob/COL/SHORT/checkboxes/note, rewrites op22rr_*.csv
  + new op22rr_mc_raw074.csv). Label GVR-mCTA, color #2ec4b6 (matches §3).
- Smoke-validated end-to-end (K512 fp32 real seqlen): exact 18/18 ok,
  cs dispatch 1/4 per PR heuristic, mc 1M BS=1 cold 43 vs 120 µs 1CTA.

### UPDATE 2026-07-08 ~14:20Z — two-node split (A expires ~16:50Z)
- Machine A b200-074: GPU0 fp32 / GPU1 bf16 (setsid drivers, logs
  mc074_gpu{0,1}.log). Node B: ALL fp16 + A-leftover takeover — paste
  NODEB_MC_PROMPT.md into node B's Claude Code.
- INCIDENT: TaskStop didn't kill driver children → old+new drivers
  overlapped 14:07:59-14:09:30 on both GPUs; best/seqlen_K2048 fp32+bf16
  batches were polluted → markers+jsonl+reps deleted, re-running. All
  real-scenario + best seqlen K512/K1024 markers (≤14:06:38) are CLEAN.
- Kill recipe: pkill -f drive_nsys_op22rr; pkill -f sweep_op22rr;
  pkill -f "nsys profile"; then re-check ps for respawns (kill -9 -PGID).

### UPDATE 2026-07-08 ~16:05Z — BACKFILL COMPLETE (074 single-node; no node B)
- NODEB_MC_PROMPT.md was pasted on 074 itself; fp16 was folded into this
  node: GPU0 chained fp32→fp16(best), GPU1 waiter bf16-exit→fp16(real,worst).
  All 81 batches done 15:55Z, before the 16:50Z node expiry.
- SECOND INCIDENT (14:12Z): the ~14:10 recovery relaunched TWICE (14:12:07 +
  14:12:37) → two fp32 drivers co-ran on GPU0 again, cross-deleting jsonl and
  clobbering the same .nsys-rep. Both trees killed; best/{seqlen_K2048,
  bs_K512}_fp32 purged + re-run. bf16/GPU1 stayed single-tenant (K2048 rerun
  75s ≈ clean 71s reference) — bf16 kept. NOTE: .done marker mtime = LAST
  toucher (all drivers touch the same file); it cannot identify the first
  finisher — judge validity from log batch headers, not marker mtimes.
- Finalize shipped @ 2c783d7d15: parse 81/81 reps (first pass had 4
  transient ranges=0 parses; uncached by design, clean on re-run),
  anchor drift base074/baseorig median 1.0053 p10 0.998 p90 1.033 (n=2718),
  REPORT D=14220 rows (2718 mc), exactness 0 FAIL. Headline gm
  t(1CTA)/t(mc): seqlen BS=1 1.36-1.41, bs grids 1.15-1.18.

## ADDENDUM 2026-07-09: op25_hls arm backfill — DONE
- Goal: add "op25 HLS" (`op25_hls` = gvr_ms_auto @ HEAD ship default: w3a
  ladder K512/K1024 + slot×2 N<65536 + fp32 C8 bs≤8) to the re-tested §1/§2
  dataset, co-located gvr_cutedsl anchor, anchor-transfer onto orig scale.
- Run history: 028 GPU0/1 (30/81, node reclaimed 01:26:33Z mid-hugeN-batch;
  the two in-flight batches had no markers → clean rerun) → 036 GPU0 takeover
  01:46Z (GPU1 82C broken cooling, banned) → 02:17Z three-card split:
  036 GPU0 = fp32 all + bf16 real/best (done 02:37Z), 038 GPU0 = fp16
  real/best, 038 GPU1 = worst bf16+fp16 (both done 03:44:46Z). 81/81, 0 FAILED.
  Logs op25hls028_gpu{0,1}.log (028), _gpu0_node036.log, _gpu[AB]_split.log.
  Split recipe = NODEC_OP25_SPLIT_PROMPT.md (shard-aware co-tenancy check).
- QA gates: exactness ok=414 FAIL=0; anchor drift op25local/baseorig
  median=1.0001 p10=0.9823 p90=1.0219 (n=2718); REPORT.html D=16938 rows;
  <script count = 2. CSVs: op22rr_{seqlen,bs}_data.csv + op22rr_op25_raw028.csv.
- Headline (cold-L2, gm over cells, t(arm)/t(op25) unless noted):
  seqlen BS=1: base/op25 1.80/1.81/1.29 (real/best/worst);
    radix/op25 0.82/0.82/0.76 — radix still wins seqlen but the old ~2.2×
    HLS gap narrows to ~1.2-1.3×; op21_hls/op25 1.10/0.97/1.25.
  bs grid: base/op25 1.40/1.37/1.01; radix/op25 1.16/1.15/1.10 — op25 now
    beats radix on the whole bs grid incl. worst; op21_hls/op25 1.05/0.95/1.13.
  vs op21 HLS: op25 wins real+worst axes, ~3-5% behind on best (ladder tax).
- Gotcha (new): pkill -f matches heredoc text embedded in a wrapper shell's
  cmdline — write kill/relaunch scripts to a file first, then exec the file.

## ADDENDUM 2026-07-09: Radix-CUDA arms backfill — DONE (b200-027, 8-GPU)
- Goal: add report.html's "Radix single-CTA (CUDA)" + "Radix multi-CTA
  (CUDA)" (`radix_single_cuda` / `radix_multi_cuda`, standalone harness
  ops, cr-aware, NOT trtllm) to the re-tested §1/§2 dataset — same
  bundles byte-for-byte, co-located gvr_cutedsl anchor, anchor-transfer.
- Run: umbriel-b200-027, ALL 8 GPUs idle → 8-chain shard (fp32 chains
  alone on GPU0/3/6; worst-fp16 K-sharded onto the lighter bf16/fp16
  GPUs). 81/81 batches in 41 min (02:42–03:23Z), 0 FAILED, 0 errors.
  Launcher = launch_radix027.sh (setsid chains, .done-marker idempotent);
  logs radix027_gpu{0..7}.log; results ../results_b200_op22rr_radix027.
- QA gates: exactness ok=414+414 FAIL=0; anchor drift radix027/baseorig
  median=1.0013 p10=0.9919 p90=1.0178 (n=5436); REPORT D=22374 rows
  (2718 × 8 arms + 630 sglang); <script> count = 2; notes/checkboxes ×1.
- Finalizer = update_report_radix.py — SELF-CONTAINED & LAST-WRITER-WINS:
  re-derives mc + op25 + radix rows from their own roots, rewrites full
  COL/SHORT consts and inserts any missing checkbox/note, so it must be
  re-run after ANY other update_report_*.py touches REPORT.html.
- Headline (cold-L2 gm, t(GVR-1CTA)/t(arm), >1 ⇒ arm faster):
  radix_single_cuda: seqlen BS=1 0.53/0.53/0.42, bs 0.61/0.60/0.47;
  radix_multi_cuda:  seqlen BS=1 0.62/0.63/0.49, bs 0.48/0.47/0.37
  (real/best/worst) — GVR-1CTA baseline beats BOTH CUDA radix ops
  everywhere (1.6–2.7×), gaps widest on worst (GVR gets faster, radix
  hr-flat). radix_cutedsl remains the only competitive radix arm
  (2–4× faster than its CUDA siblings on real). bs grid
  t(best-radix-CUDA)/t(op25_hls) = 1.82/1.80/1.70.
