# COST.md — op40_omni_gvr (final, 2026-07-23)

Budget: uncapped (user ruling). Actuals (estimates, single day on
umb-b200-239):

| Phase | GPU-h (approx) | Notes |
|---|---|---|
| Phase 0/harness bring-up + gates | ~6 | compiles + 3 gate rounds |
| bl0 baseline grid | ~8 | 8-way, 25 batches |
| ab_v1 / ab_iter2 / ab_iter3 / ab_iter6 / ab_iter8 grids | ~70 | 2-4 arms × 865 × paired nsys |
| probes (floor/csopt/tsweep/icache/smemcache/NCU/phases) | ~10 | |
| **total** | **~95 GPU-h** | one working day, 8× B200 |

Deliverables per cost: 1 ship arm (gm 1.1250, 0 reg), 2 upstream defect
fixes + repros, 6 falsifications + 3 walls with mechanisms (ledger value for
future campaigns), double-locked stretch-goal attribution.
