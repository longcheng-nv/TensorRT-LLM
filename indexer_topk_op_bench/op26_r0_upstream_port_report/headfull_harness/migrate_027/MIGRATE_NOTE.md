Completed batches from b200-027 (2026-07-20). Incomplete-batch jsonls NOT
staged (partial jsonl + fresh rep = missing NVTX ranges; redo whole batch).
Restore on target node:
  mkdir -p /tmp/gvrheadfull/refresh_results/nsys_reps
  cp <thisdir>/*.jsonl /tmp/gvrheadfull/refresh_results/
  cp <thisdir>/.done_* /tmp/gvrheadfull/refresh_results/
  cp <thisdir>/nsys_reps/* /tmp/gvrheadfull/refresh_results/nsys_reps/
nsys reps NEVER go into git (env tokens; dir gitignored). 019: GPU0 broken
cooling — run 7-way shards, W in 0..6 -> CUDA_VISIBLE_DEVICES=W+1.
