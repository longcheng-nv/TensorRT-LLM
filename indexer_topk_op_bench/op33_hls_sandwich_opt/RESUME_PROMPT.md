# op33 RESUME — cross-server migration (2026-07-14)

## 0. HOW TO RESUME ON THE NEW 8×B200 SERVER
The workspace is on **NFS** (`dc2-cdot33-scr01-lif4a:/vol/scratch54/scratch.loncheng_gpu`), so the
new server sees the SAME files at the SAME path + the SAME git HEAD. No push/pull/copy needed — just
paste the block in section 1 into Claude Code on the new node.
- Previous node: **umbriel-b200-027** (session interrupted). Its grid procs are orphaned there; ignore.
- git branch = `omni/op21-gvr-prod`. Bucket = `indexer_topk_op_bench/op33_hls_sandwich_opt/`.

## 1. PASTE-BLOCK for the new server's Claude Code
```
Resume the op33 HLS-op27 sandwich campaign on this 8×B200 node (migrated from umbriel-b200-027).
Workspace is NFS-shared at /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM
(branch omni/op21-gvr-prod). Read first, in order:
  indexer_topk_op_bench/op33_hls_sandwich_opt/RESUME_PROMPT.md   (this file — full state)
  indexer_topk_op_bench/op33_hls_sandwich_opt/SESSION_CONTEXT.md (full carryover)
  indexer_topk_op_bench/op33_hls_sandwich_opt/ITERATIONS.md      (iter0-6 verdicts)

VERDICT SO FAR (committed): op33 = NO-SHIP. M=3 (qfracs 0.85,0.35, D2) REGRESSES the WORST
scenario (clean single-GPU paired A/B: K1024 N32768 worst 0.787, K512 N262144 worst 0.727,
K512 N8192 worst 0.884). The iter5 "+9% ship" was a real-only/N<=65536 + 8-GPU-contention
measurement artifact. dispatch = op27_hls default (src/gvr_ms_op33.py marked DO-NOT-SHIP).

PENDING TASK (finish this): a CLEAN full-3-axis grid to document the NO-SHIP comprehensively and
confirm M=3 has no clean safe region. RESUMABLE (partial in results/reliable_grid.csv, ~8/54 done).

PREFLIGHT (before launching):
  1. cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM
  2. git rev-parse --abbrev-ref HEAD   (confirm omni/op21-gvr-prod)
  3. nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu --format=csv,noheader
     -> need GPU1 & GPU2 IDLE (<50C, ~0%). If busy (external jobs land on shared nodes), pick idle
        GPUs and edit the shard() GPU ids in scripts/run_reliable_grid.sh.
  4. pgrep -f 'run_reliable_grid|paired_one'  -> must be EMPTY on this node (kill any stale).

LAUNCH (resumable — skips cells already in results/reliable_grid.csv):
  cd indexer_topk_op_bench/op33_hls_sandwich_opt
  setsid bash scripts/run_reliable_grid.sh > results/reliable_grid.log 2>&1 &
  # ~35-40 min for remaining cells. Watch: grep -c RELIABLE_DONE results/reliable_grid.log ;
  #   wc -l results/reliable_grid.csv  (target 54 data rows + header).

ON COMPLETION (RELIABLE_DONE in log):
  1. Analyze results/reliable_grid.csv: per (N × scenario) geomean of ratio (base_ns/m3_ns, >1 =
     M=3 faster) over K512/1024. Find any N where M=3 wins ALL 3 axes (best AND real AND worst >=1).
     Expected: NONE (worst regresses broadly) -> dispatch STAYS default, verdict STAYS NO-SHIP.
  2. Refresh REPORT.html: replace the misleading real-only knob table with this clean 3-axis grid
     (edit scripts/gen_report.py or add a small table); keep the NO-SHIP verdict.
  3. If (unexpectedly) a clean safe N-region exists, refine gvr_ms_op33.py dispatch to
     "M=3 iff K<2048 AND N in <safe region>", re-gate exactness, re-measure that region paired.
  4. Record iter7 in ITERATIONS.md, update memory project_op33_hls_sandwich_opt.md, commit
     (git commit --no-verify -s scoped to the op33 bucket; add Claude Code Co-Authored-By +
     Signed-off-by trailers).

GOTCHAS (learned this campaign — do NOT repeat):
  - ALWAYS report all 3 verdict axes [worst,real,best]. real-only hid the worst regression.
  - Ship verdicts need single-GPU PAIRED back-to-back A/B. >=8 concurrent nsys corrupts ratios
    (contention) AND flakes capture (empty base_ns). The reliable grid runs <=2 concurrent nsys.
  - nsys sqlite leaks env tokens: env -u GITHUB_TOKEN -u HF_TOKEN before nsys; *.nsys-rep/*.sqlite
    gitignored in results/.gitignore; the grid deletes them after parsing.
  - git commit --no-verify (repo-wide pre-commit hook times out); scope adds to the op33 bucket.
```

## 2. Exact reproduction facts
- Incumbent op27_hls = `build_call("gvr_ms_auto",K,fp32,N,1,cr,logits,preidx)` (from harness/
  sweep_nsys.py) with env OP21_FB_LOGFALSI=1 OP27_K2048_TAIL=1. M=3 variant = same + env
  OP25_QFRACS=0.85,0.35 (K512/1024 only; K2048 keeps M=4 tail ladder).
- Harness: scripts/paired_one.py (one cfg per nsys window), scripts/run_reliable_grid.sh (resumable).
- Bundles: bundle_data_rr.get_bundle(scen,K,torch.float32,N); N4096 missing for K2048 (skip).
- Clean 5-cell paired reference (already trustworthy): results/paired_final.txt.
- Node this was authored on: umbriel-b200-027 (8×B200, cap 10.0, 148 SM).

## 3. Deliverables (all committed, NFS-visible on the new node)
PLAN.md, SESSION_CONTEXT.md, ITERATIONS.md (iter0-6), FALSIFIED.md, AUTONOMY.md, REPORT.html (temp),
src/gvr_ms_op33.py (DO-NOT-SHIP), scripts/{harness,paired_one,gate_m3,val_dispatch,gen_report,
run_reliable_grid,...}, results/{baseline.log,knobs.csv,d3.csv,dispatch.csv,paired_final.txt,
reliable_grid.csv(partial)}.
