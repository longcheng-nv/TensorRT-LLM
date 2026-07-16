# §8 rival full-ISL real BS backfill — note (2026-07-16, umbriel-b200-081)

## Why
User observed that in REPORT §8's real(decode-capture) BS view, the external
arms (SGLang v2 / FlashInfer / Radix-cuteDSL) only had data at ISL=128k. Root
cause: the 07-15 rival sweep's real BS grid ran a single representative rung
(`REAL_BS_ISL=128k`, sweep_rival.py), while the 07-16 GVR refresh extended the
GVR arms to full-ISL BS grids — two sweeps, two scopes.

## What was run
- Node: umbriel-b200-081 (8× B200, idle), same container generation as the
  07-15 rival run (torch 2.12.0a0 nv26.05 / cuda 13.2) + same env recipe:
  `PYTHONNOUSERSITE=1` + machine-local cutlass 4.5.0 + fi_clean flashinfer
  0.6.11 (BS_SCALING_ENV_FIXES.md / RIVAL_SWEEP_NOTES.md).
- Arms: `radix_cutedsl` / `sglang_v2` (fp32-only) / `flashinfer_topk` +
  `op26_r0auto` as the cross-node ANCHOR (gate only, rows NOT merged).
- Grid: real BS sweep, ALL ISL rungs (flash/pro 4k–1024k = 9, v32 4k–256k = 7)
  × dtype {fp32,fp16,bf16} × BS {1..1024} — 75 nsys batches (one per
  model×dtype×isl, op28 batch protocol), 2750 cells, **0 errors**,
  rival exactness **1925/1925** vs torch.topk. ~25 min on 8 GPUs.
- Harness: `rival_harness/batches_backfill.py` + `drive_backfill_shard.sh`;
  `sweep_rival.py` gained `--isl` (per-ISL batches) + `--ops` (arm filter);
  `parse_rival.py` now prefers a rep named exactly like the jsonl stem.

## Gates (aggregate_backfill.py, per-batch p95 per the fin2 lesson)
- Anchor drift backfill(081)/refresh(094), n=825 overlapping cells:
  **median 1.000, p95(sym) 1.055** (gate ≤1.15).
- Rival 128k overlap backfill(081)/rival(044): median 0.998–1.001,
  p95(sym) ≤1.053 per arm.
- 1 failing batch: `v32/bf16/16k` p95 1.174 (median 0.993). Re-run
  independently → identical pattern (±15% per-cell scatter at BS≤32, BOTH
  directions; the 094 rows are equally non-monotonic there). Verdict:
  noise-limited ~15µs short-kernel cell, NOT node bias / contamination →
  merged with `--force`, disclosed in REPORT §8 provenance.

## Merge
`rival_long.csv` 10934 → 12628 rows: dropped the 231 old single-rung (128k)
rival rows (044), added 1925 full-ISL rival rows (081) — the real BS rival
grid is now single-node consistent. GVR rows (094 refresh) untouched.
Re-run recipe: `rival_harness/drive_backfill_shard.sh <W> <GPU> <NW>` (resumable),
then `parse_rival.py /tmp/gvrval1/rival_results_bf` + `aggregate_backfill.py`
+ `gen_report.py`.

## New finding (only visible with the full grid)
The BS story is strongly ISL-dependent (fp32, t(op26)/t(rival), >1 = rival
faster): SGLang v2 keeps both ENDS (BS=1: 1.21–1.77×; BS=1024: 1.14–1.74×)
but collapses in the MIDDLE at large ISL — BS 8–64 × ISL ≥128k falls to
0.45–0.76 (GVR op26 up to ≈2.2× faster at 1M/BS32). Radix's large-BS collapse
is ISL-graded (BS1024: 0.92 @4k → 0.33–0.47 @≥64k). FlashInfer stays within
0.67–1.31 everywhere. §8 trend bullets updated (EN+ZH).

nsys-reps/sqlite live only in /tmp/gvrval1/rival_results_bf* (env-token rule —
never commit).
