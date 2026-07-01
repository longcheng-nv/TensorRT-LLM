# op17 GVR threshold-portfolio — deliverable snapshot (2026-07-01, B200 sm_100)

Open **REPORT.html** (bilingual EN/中文, CSS-only toggle, no JS) for the full write-up.

- `gvr_portfolio_cluster_op.py` — the operator (current best). Call:
    `gvr_portfolio_cluster(logits, pre_idx, seq_lens, K, compress_ratio, G="auto")`
    (auto-G dispatch; falls back to single-CTA baseline at high BS).
- `RESULTS_SUMMARY.md` — all perf tables (CSV-in-markdown).
- `ITERATIONS.md` / `LEARNINGS.md` — full process log + knowledge base.
- Live sources & nsys reps: ../src, ../scripts, ../results/nsys/v2_*.nsys-rep

## Headline (all EXACT, vs single-CTA gvr_cutedsl baseline)
- BS=1 nsys pure-kernel: 1.21-1.67x (fp32); event x3-median avg ~1.22x, all dtypes, no regression.
- vs PR#15198 multicta cluster: wins N<=65K, loses N>=262K -> per-(N,BS) dispatch.
- NOT the original +40%-universal (physically unreachable); a real broad low-BS/decode win.

## Next (see REPORT §6): D1 partition+portfolio fusion (reclaim N>=262K), D2 adaptive band,
## D3 G=2 stability / G>16, D4 productionization + GSM8K gate.
