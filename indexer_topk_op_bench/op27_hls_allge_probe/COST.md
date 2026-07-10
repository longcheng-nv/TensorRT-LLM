# op27 — token & cost accounting

Model: Claude Fable 5 (`claude-fable-5`). Pricing (per 1M tokens): input $10,
output $50; prompt-cache read ~0.1x input ($1/M), cache write 1.25x ($12.50/M).

Method: per-phase estimates from session activity (tool-call volume x typical
context size, cache-adjusted). These are ESTIMATES — the authoritative number
is the CLI `/cost` command run by the operator; record its value in the last
column at each phase boundary. GPU cost tracked separately in GPU-hours.

| # | Phase (2026-07-10) | Est. output tok | Est. billed input (cache-adj) | Est. cost | /cost (auth.) |
|---|---|---|---|---|---|
| 0 | REPORT.html audit + fixes (pre-op27, same session) | ~35K | ~1.5M eff. | ~$5-8 | |
| 1 | op27 recon + PLAN + bucket (kernel read, loss-cell mapping) | ~12K | ~0.8M eff. | ~$2-3 | |
| 2 | host screen build + run (screen_mprobe.py; probe hypothesis FALSIFIED for K512/1024) | ~10K | ~0.6M eff. | ~$2-3 | |
| 3 | iter1 decomposition A/B authoring + launch (ab_decomp.py, driver; GPU0 b200-027) | ~10K | ~0.7M eff. | ~$2-3 | |
| 4 | iter1 parse + verdict | | | | |
| 5 | kernel change (if screen justifies) + gates | | | | |
| 6 | nsys ship A/B + report | | | | |

GPU usage log:
- 2026-07-10 ~10:0xZ: iter1 ab_decomp launched, GPU0 umbriel-b200-027,
  3 scenarios x 15 cells x ~4 arms, est. 1-1.5 GPU-h.

Running estimated total (phases 0-3): ~$11-17 agent + ~0 GPU-h billed-idle.
Update at every phase boundary; reconcile against `/cost` when the user runs it.
