# op37 RESUME PROMPT — paste into a fresh Claude Code session

Continue the op37 pure-GVR-vs-sglang campaign in
`indexer_topk_op_bench/op37_gvr_sgl_n32k/` (TensorRT-LLM checkout, branch
omni/op21-gvr-prod). Read in order: PLAN.md (goal, baseline 0.8664 arithmetic,
loss map, mechanism map, falsification ledger), DP4V2_DESIGN.md (sync-
reduction chain), then this file's State.

## Hard constraints (user /goal)
- Axis = op26 REPORT §7b real decode-capture, fp32, K{512,1024,2048},
  **N≥32K only** (12 rungs × 11 BS = 132 cells vs sglang_v2 us_span).
- Pure GVR algorithm improvement ONLY: no dispatch to radix/sgl_bx/etc.
  (gvr29 wholesale port ruled OUT — it's a sglang CUDA fork; HBE *ideas*
  re-implemented inside GvrTopKKernel are IN scope.)
- Checkpoint everything to NFS + git commit at every milestone (user order
  2026-07-20). Final deliverable = bilingual REPORT.html (CSS-only toggle,
  zero <script>) in this dir.

## Environment (new node)
```
WD=/tmp/gvrlayers; mkdir -p $WD/cutlass450
ln -sfn /home/scratch.loncheng_gpu/python-userbase/lib/python3.12/site-packages/nvidia_cutlass_dsl $WD/cutlass450/nvidia_cutlass_dsl
ln -sfn /home/scratch.loncheng_gpu/python-userbase/lib/python3.12/site-packages/flashinfer $WD/cutlass450/flashinfer
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WD/cutlass450/nvidia_cutlass_dsl/python_packages:$WD/cutlass450
# cutlass must print 4.5.0
```
All nsys via `env -u GITHUB_TOKEN -u HF_TOKEN`; never commit *.nsys-rep /
*.sqlite; ship verdicts ≤2-way concurrent nsys, arms paired same-run
same-GPU; cold-L2 protocol is inside the harness.

## State (2026-07-20, umbriel-b200-028)
- T0 baseline @e6fdbfac3d DONE: composite gm 0.8664 (results/baseline,
  132 cells, 0 inexact). Loss: BS1-8 0.68-0.71, BS16 0.82, BS≥256 ~0.89
  (flash-512k 0.48-0.53); wins BS32-128 1.09-1.17.
- Mechanism (results/phase_bs.csv): warm cells P4-bound 44-58% leader-only;
  cold cell (flash-512k hit .057) P2-bound 45-51%; flash-512k bigBS =
  BW multi-pass wall (pr 0.40 vs sglang 0.19 µs/row @BS1024; pr 2-3× faster
  than base there, R0 not the cause).
- FALSIFIED: forced cs2/cs4 at 32K rung (results/l1probe — loses all BS).
- gvrpkg37 = current head + dist_p4 splice (variant/, kernel md5
  fd5a675bc3624113a87d76b0c5b8dcbc): battery 42/42, PTX byte-identical
  default-off, real-data 3/3 exact (logs/).
- IN FLIGHT at last checkpoint: dp4-v1 verdict sweep results/dp4
  (12 rungs × {gvr_pr, sglang_v2, gvr_dp4}, drive_op37.sh workers on GPU0/1).
  If interrupted: batches are cell-resumable BUT never resume a partially
  written batch (nsys rep overwrite) — delete that batch's jsonl + rerun it.
  Analyze: `python3 analysis/analyze.py results/dp4 --ref sglang_v2` and
  `--ref gvr_pr --arms gvr_dp4`.

## Next-step decision tree
1. dp4-v1 verdict: expect win N≥262K / sync-tax loss 65-131K (op36 A2 shape).
   Then implement DP4V2 levers 1+2 (6→4 sync rounds; see DP4V2_DESIGN.md)
   in gvrpkg37 as flags, battery re-run, nsys A/B same cells. Target: win
   boundary down to N≈65K, BS≤8 warm cells → ≥0.95 vs sglang.
2. Cold-hit P2 wall (flash-512k + big-BS): design admission/guess lever
   (HBE-lite sampled guess for fallback threshold is the in-scope shape;
   qfracs retuning was silicon-wash at BS=1 but big-BS is BW-bound so
   pass-count reduction DOES pay there).
3. Composite accounting after each lever vs results/baseline; protect the
   BS32-128 win block (zero-regression rule).
4. Close: full 132-cell verdict sweep with final flags, REPORT.html
   (bilingual, CSS-only), memory update, commit.
