# NODE-B LAUNCH PROMPT — op22 GVR-mCTA backfill, fp16 slice (+ A-leftover takeover)

Paste everything below the line into a fresh Claude Code session on the
SECOND B200 machine (dual-GPU, same NFS). Machine A (umbriel-b200-074,
expires ~16:50 UTC 2026-07-08) owns fp32+bf16; this node owns ALL fp16.

---- PASTE BELOW THIS LINE ----

Continue the op22 "GVR multi-CTA (cuteDSL, PR#15198)" benchmark backfill.
Working dir (NFS-shared checkout, branch omni/op21-gvr-prod):
`/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench/`
Read `RESUME_PROMPT.md` section "ADDENDUM 2026-07-08" for full context.

GOAL: measure arms (gvr_cutedsl anchor + gvr_multicta_cutedsl) over the
op22rr grid. Machine A (umbriel-b200-074) is running fp32 (GPU0) and bf16
(GPU1) into the SHARED output root `../results_b200_op22rr_mc074`
(batch-level `.done` markers make everything resumable/idempotent).
THIS NODE owns the fp16 slice (27 batches), split across its 2 GPUs.

## 1. Preflight (all from the working dir above)
1. `git log --oneline -1` — expect `8bb22d1daf` or later (op22rr checkpoint).
2. `python3 -c "import torch, cutlass; print(torch.cuda.is_available())"`
   must print True (plain python3; needs torch+cutlass only, NOT trtllm).
3. `nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used --format=csv`
   — GPUs must be B200, memory.used ≈ 0, idle temp ≤ 50 C (a hotter GPU
   has broken cooling — do NOT time on it; put both slices on the good one
   sequentially instead).
4. Claim the slice (prevents double-run if A tries to pick fp16 up later):
   `touch ../results_b200_op22rr_mc074/.nodeB_claimed_fp16`

## 2. Launch (fp16, disjoint by scenario across the two GPUs)
```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench
D=$PWD
setsid bash -c "cd $D && OP22RR_ARMS='gvr_cutedsl,gvr_multicta_cutedsl' OUT=results_b200_op22rr_mc074 GPU=0 DTYPES=fp16 SCENARIOS='real best' bash drive_nsys_op22rr.sh >> mc074_nodeB_gpu0.log 2>&1" &
setsid bash -c "cd $D && OP22RR_ARMS='gvr_cutedsl,gvr_multicta_cutedsl' OUT=results_b200_op22rr_mc074 GPU=1 DTYPES=fp16 SCENARIOS='worst' bash drive_nsys_op22rr.sh >> mc074_nodeB_gpu1.log 2>&1" &
```
GPU0 = 18 batches (~1.5 h), GPU1 = 9 (~45 min), at A's observed pace
(seqlen/hugeN ~1-2 min, bs ~6 min per batch). setsid = survives session.

## 3. Monitor
- `grep -c "BATCH DONE" mc074_nodeB_gpu0.log mc074_nodeB_gpu1.log`
- fp16 done target: `ls ../results_b200_op22rr_mc074/*/.done_*_fp16 | wc -l` == 27
- Zero `FAILED` lines expected; a FAILED batch is left unmarked — rerun the
  same driver command (it resumes).
- GOTCHAS: Claude's TaskStop does NOT kill driver children — to stop, use
  `pkill -f drive_nsys_op22rr; pkill -f sweep_op22rr; pkill -f "nsys profile"`,
  then verify with `ps -eo pid,ppid,cmd | grep -E "drive_nsys|sweep_op22|nsys prof"`
  and re-check for respawns after 30 s (kill by PGID if needed). Never run
  two drivers whose (scenario,sweep,K,dtype) sets overlap on any node.
  If `rm` is permission-blocked, delete via `python3 -c "import os; os.remove(...)"`.

## 4. After GPU1 (worst-fp16) finishes: A-leftover takeover check
Machine A dies at ~16:50 UTC. Once BOTH conditions hold:
  (a) current time ≥ 16:55 UTC, and
  (b) `stat -c %y ../results_b200_op22rr_mc074/*/.done_* | sort | tail -1`
      shows no new marker for ≥ 10 min,
check what A left unfinished:
`for dt in fp32 bf16; do echo $dt $(ls ../results_b200_op22rr_mc074/*/.done_*_$dt 2>/dev/null | wc -l)/27; done`
If short of 27, run A's slices here on the free GPU(s) — same commands as
§2 but with `DTYPES=fp32` (and/or `bf16`), no SCENARIOS filter, logs
`mc074_nodeB_gpu0_takeover.log` etc. `.done` markers skip everything A
completed; only missing batches re-run.

## 5. Finalize (only when ALL 81 batches are done: 27 per dtype)
```bash
python3 parse_op22_cached.py ../results_b200_op22rr_mc074
python3 update_report_mc.py          # anchor-transfer + REPORT.html patch + CSVs
```
- update_report_mc.py prints anchor-drift stats (base074/baseorig median
  should be ~1.00 ± few %; investigate if p10/p90 outside 0.9-1.1) and is
  idempotent (safe to re-run).
- Verify: REPORT.html has `gvr_multicta_cutedsl` in COL/SHORT consts, mc
  checkboxes in §1/§2, the teal (#2ec4b6) mc-note card; op22rr_*.csv have
  gvr_multicta_cutedsl columns; op22rr_mc_raw074.csv exists.
- Exactness sanity: every mc BS=1 rec has "exact":"ok" —
  `grep -h '"exact"' ../results_b200_op22rr_mc074/*/*/results_K*.jsonl | grep -c FAIL`
  must print 0.
- Commit (repo convention: `git commit -s`, trailers `Made-with: Claude Code`
  + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`); stage ONLY
  REPORT.html, op22rr_*.csv, op22rr_mc_raw074.csv, RESUME_PROMPT.md,
  NODEB_MC_PROMPT.md, update_report_mc.py — NEVER any *.sqlite/*.nsys-rep
  or results_b200_op22rr_mc074/ content (nsys embeds env tokens).
- If machine A's session already committed the final report (check
  `git log --oneline -3` for "[op22 mc]"), skip §5 entirely.
