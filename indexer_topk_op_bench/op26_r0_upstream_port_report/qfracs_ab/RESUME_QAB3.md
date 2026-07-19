# RESUME — bundle-v2 4-arm A/B (qab3)

**Goal**: validate op35 bundle-v2 (skip_h1 + kb512@K2048) ON TOP of the shipped
PR#16457 head + K2048 rung swap (head 0d6fc4f1f2), full 77-cell fp32 grid,
nsys cold-L2, 4 arms paired per cell: gvr_ship / gvr_h1 / gvr_kb / gvr_full.

**State (update as it progresses)**
- [x] patched bench kernel: /tmp/gvrqab/gvrpkg/top_k/gvr_topk_decode.py
      = PR head 0d6fc4f1f2 file + 2 bench flags (skip_h1, k_num_bins);
      canonical checkpoint copy = `bundlev2/gvr_topk_decode_bundle.py` (NFS).
- [x] smoke: 4 flag combos compile + exact @K2048 N=262144 (b200-027 GPU0)
- [ ] batches: 9 = synth best/worst x 3K + real x 3 models (fp32, BS=1 seqlen)
- [ ] verdict: aggregate_qab3 → qab3.csv; ship rule = no cell < 0.95, real gm
      and K2048-domain gm are the headline; compare vs op35's standalone
      +3.9%/K2048 +13.3% for interaction with the new rung pair.

**Machine layout (umbriel-b200-027)**
- /tmp/gvrqab: gvrpkg (PATCHED), cutlass450 symlink → /tmp/gvrlayers/cutlass450,
  qab3_results/ (jsonl + .done + nsys_reps).
- GPU0: batches_qab3a (synth best x3) · GPU6: qab3b (worst x3) · GPU7: qab3c (real x3)

**Resume after interruption (any host state)**
1. If /tmp/gvrqab lost: rebuild —
   `mkdir -p /tmp/gvrqab/gvrpkg/top_k`; copy gvrpkg skeleton
   (`gvrpkg_snapshot/gvrpkg/{__init__,utils}.py`, `top_k/__init__.py`),
   worktree `block_scan.py` + `single_pass_multi_cta_radix_topk_cluster.py`
   from TensorRT-LLM-gvr-r0 .../top_k/, then
   `cp qfracs_ab/bundlev2/gvr_topk_decode_bundle.py /tmp/gvrqab/gvrpkg/top_k/gvr_topk_decode.py`;
   `ln -sfn /tmp/gvrlayers/cutlass450 /tmp/gvrqab/cutlass450`
   (if gvrlayers also lost: cutlass450 recipe in rival_harness/drive_rival_shard.sh).
2. Restore partial results (if /tmp lost): `cp qab3_ckpt/* /tmp/gvrqab/qab3_results/`
   — BUT delete any batch whose .done marker is absent (jsonl-resume + rep
   overwrite silently drops timings; rerun those batches whole).
3. Relaunch (idempotent, .done-guarded):
   `BATCHFILE=batches_qab3a.py setsid bash drive_qab3.sh 0 > /tmp/gvrqab/qab3a.log 2>&1 &`
   (same for b→GPU6, c→GPU7; pick any idle GPUs — arms are paired within-batch).
4. Aggregate: `python3 aggregate_qab.py` won't work (3→4 arms); use
   `python3 aggregate_qab3.py` (4-arm variant).
5. Checkpoint sync loop (jsonl+markers → NFS qab3_ckpt/, NEVER nsys reps —
   they embed env tokens): `bash ckpt_sync_qab3.sh` (runs until QAB3 done).

**Verdict destination**: §9c addendum via update_report_rungrecal.py pattern;
port target = SEPARATE follow-up PR (user directive), NOT #16457.
