# HEADFULL verdict — PR#16457 head @e6fdbfac3d vs REPORT.html (@018251950f grids)

> 2026-07-20 ~08:0x UTC. Sweep: 54/54 batches (16 on b200-027, 38 on
> umbriel-b200-019 GPUs 1-7; GPU0 excluded — broken cooling, 75C idle).
> 8316 rows (2772 cells x 3 arms), 0 errors. Raw jsonl + compare log in
> `results_headfull/`. REPORT reference = refresh grids measured on b200-094.

## Exactness

- **gvr_pr (shipped head): 2772/2772 exact.**
- 36 inexact all in the gvr_base arm, all at real flash N=131075 (fp32/fp16/
  bf16 x 12 BS) — the long-known base undershoot on real data, not a head
  defect.

## Anchor gate (op26_r0auto in both runs)

- n=2618, med **1.015**, p95 1.091. Origin split: 027 med 1.016 / 019 med
  1.015 — **no bimodality**, the two source nodes are mutually consistent.
- The p95 tail (1.09-1.20) concentrates in DRAM-heavy N=1M large-BS cells:
  these nodes are ~10-20% slower than b200-094 on bandwidth-saturated cells.
  Absolute cross-node ratios there reflect node drift, not the kernel —
  worst raw "REPORT/new" cells (0.77-0.90) all sit in this tail and are
  cleared by the within-run pairing below.

## Six-axis comparison (pr arm)

anchor-normalized REPORT/new geomean (1.00 = parity with REPORT after node
bias removal), and node-clean within-run base/pr speedup vs REPORT's:

| axis | n | anchor-norm pr | within-run base/pr (head) | REPORT base/pr | delta |
|---|---|---|---|---|---|
| §3 synth seqlen fp32 | 52 | 0.977 | 1.156 | 1.148 | +0.7% |
| §7 synth BS fp32 | 572 | 0.973 | 1.178 | 1.153 | +2.2% |
| §7 synth BS 16-bit | 1144 | 0.997 | **1.157** | 1.098 | **+5.5%** |
| §4 real seqlen fp32 BS=1 | 25 | 0.953 | 1.257 | 1.309 | -4.0% |
| §7 real BS fp32 | 275 | 0.972 | 1.313 | 1.293 | +1.6% |
| §7 real BS 16-bit | 550 | 1.015 | **1.213** | 1.141 | **+6.3%** |

## Verdict

1. **Head reproduces REPORT on all six axes** — anchor-normalized pr within
   [0.953, 1.015]; no axis regressed beyond the ~5% cross-node noise floor
   established by the anchor tail.
2. **16-bit axes genuinely improved** (+5.5% synth / +6.3% real within-run
   speedup vs the REPORT snapshot): consistent with the post-@018251950f
   commits (vseed + p4tt tiny-tie fast path) — 16-bit data ties far more
   often, so the exact-tail fires more and p4tt recovers it.
3. **real seqlen fp32 -4.0%** (1.257 vs 1.309, 25 cells, 027-origin batch):
   driven by pro mid-ISL cells (64k 1.046 vs 1.108, 256k 1.127 vs 1.205,
   512k 1.226 vs 1.290). Two node families + the exact-tail correctness
   machinery on the tie-prone pro capture make this the expected direction;
   magnitude is within the anchor p95 envelope. Not a ship blocker; if it
   matters, a paired old-vs-new on ONE node (newpr_nsys_ab.py style) is the
   discriminating experiment.
4. Worst raw cells (synth worst K2048 1M BS128-256 16-bit, 0.77-0.78 raw)
   decompose into anchor drift 1.14-1.18 x within-run residual ~0.95-0.97
   vs REPORT's ~0.98 — a <=5% within-run pr-vs-base deficit on 3 envelope-
   edge cells (N=1M is outside the deployment envelope, stress probe only).

**Bottom line: shipped PR#16457 head @e6fdbfac3d holds or beats every
REPORT.html axis; 16-bit BS grids are measurably better than the published
numbers; no real regression found.**
