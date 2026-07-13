# COST — per-phase accounting

Budget (from PLAN.md): ____ GPU-h · $____ tokens.
Calibration anchors from the GVR record: mid-size campaign ≈ 15 GPU-h + ~$108
(op26); flagship productionization ≈ $797 (op21, 16 iterations, 4 days).
Report burn rate to the human at 80% of budget (AUTONOMY must-stop).

| Phase / iter range | GPU-h | Tokens ($, per /cost) | Notes |
|---|---|---|---|
| setup + probes | | | |
| iters — | | | |
| verdicts (nsys) | | | |
| **total** | | | |

## rmsnorm_campaign ledger
Budget: 5 iterations / ~2 h wall-clock, single B200 (GPU2).

| Phase / iter range | GPU-h | Tokens ($, per /cost) | Notes |
|---|---|---|---|
| setup + iter0 characterization | ~0.25 | n/a (session) | smoke, L1 grid, ncu x1 cell, nsys incumbent+copy grids (start 01:43 UTC) |
| node migration + re-anchor (027→035) | ~0.05 | n/a | anchor re-run x3 on 035/GPU1 → 21.17 µs |
| iter 1 (gate + full nsys grid) | ~0.6 | n/a | verify_exact x5 cells, nsys_verdict x5 cells (T=16384 rerun after 10-min timeout) |
| iter 2 (NCU + config screen + 2 nsys) | ~0.4 | n/a | ncu x1, L1 x14 configs, nsys x2 escalations — FALSIFIED |
| iter 3 (dispatch gate + final nsys grid) | ~0.5 | n/a | verify_exact x5, nsys_verdict x5 on final artifact — SHIP |
| **total** | **~1.8 GPU-h** | 1 session | 4 iters used of 5; ~2 h wall; well under GVR mid-size anchor (15 GPU-h) |
