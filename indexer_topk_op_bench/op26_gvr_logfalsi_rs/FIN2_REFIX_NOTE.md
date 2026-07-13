# fin2 §4 anchor-drift refix note (2026-07-13, umbriel-b200-047)

## What happened
fin2 reached 81/81 (36 from 027 handoff + 45 re-run on 047). §4 step-3 anchor
drift (gvr_cutedsl BASE, fin2 vs ORIG_ROOT=results_b200_op22rr) **aggregate**
median was clean (1.0015), but the **per-batch** targeted check of the 027-tail
markers (13:45-13:55, right before the external 8-GPU job landed on 027) found
**two contaminated batches**:

| K512 batch      | med   | p95   | max   | >1.1 cells | verdict       |
|-----------------|-------|-------|-------|-----------|---------------|
| real/bs         | 1.003 | 1.485 | 1.721 | 26        | CONTAMINATED  |
| best/seqlen     | 1.023 | 1.321 | 1.745 | 5         | CONTAMINATED  |
| (clean control) worst/bs K512 | 1.001 | 1.040 | 1.132 | 1 | clean |
| (clean control) best/bs  K512 | 1.000 | 1.023 | 1.073 | 0 | clean |

Signature was **systematic**, not cold-L2 jitter: absolute cold-us inflated
~1.68× across many (N,BS,dtype) cells (e.g. 28.9us vs 16.8us orig at N=32768).
Random small-cell noise (present in every batch) sits at p95<=1.04; contamination
sits at p95>1.15. Decision rule used: **p95 > 1.15 on the base anchor ⇒ contaminated**.

## Fix
Moved 6 markers aside (`/tmp/op26_contaminated_markers/`, `rm` is harness-blocked
so used `mv`), re-ran real/bs + best/seqlen K512 × {fp32,bf16,fp16} on the
verified-empty 047 (all GPUs 0 MiB). Driver idempotently SKIPs the clean
real/seqlen + best/bs cells. Logs: `fin2fix_gpu{0,1,2}.log`.

## Tools
- `check_027tail_drift.py` — per-cell drift on the 3 nominal 027-tail batches.
- `/tmp/drift_by_batch.py`, `/tmp/scope.py` — per-batch small-N vs large-N split
  and p95 contamination classifier (rewrite into op26 dir if needed again).

## After refix
Re-parse fin2, re-run update_report_op26_iter6.py, re-confirm drift p95<=1.15 on
ALL K512 batches (esp. the two refixed), exactness 414/414, then COST/commit.
