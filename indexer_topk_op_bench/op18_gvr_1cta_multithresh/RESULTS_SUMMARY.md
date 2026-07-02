# op18 single-CTA multi-threshold GVR top-K — results summary (B200 sm_100)

Operator: `src/gvr_mt_op.py` (`gvr_mt_auto(...)` — per-(K,N) dispatch, CDF-aware
placement). Baseline: single-CTA `gvr_cutedsl`. Synth = report bundles (seed 42).
All cells EXACT (vdiff=0, uniq=K).

## x3-median cold-L2 event, full grid (results/validate_x3.jsonl)
dtype,min,avg,max,exact
fp32,1.010,1.144,1.344,60/60 (20 cells)
bf16,1.000,1.114,1.244,exact
fp16,1.012,1.143,1.364,exact
Zero cells < 0.99 in any dtype.

## nsys pure-kernel (BS=1 fp32, cold-L2, 100 iters, median)  [canonical]
cell,base_us,mt_us,nsys_speedup,event_speedup
K512_N16384,15.31,10.59,1.446,1.289
K512_N65536,18.82,16.77,1.122,1.113
K512_N262144,38.46,35.10,1.096,1.080
K1024_N32768,21.92,16.19,1.354,1.297
K1024_N262144,41.47,36.38,1.140,1.130
K2048_N65536,25.06,18.37,1.364,1.313
K2048_N262144,51.07,36.93,1.383,1.344

## BS sweep (K512 fp32 N65536, M3R1)
BS,speedup: 1,1.086 / 4,1.076 / 8,1.082 / 16,1.097 / 32,1.102 / 64,1.150 / 128,1.164
No high-BS guard needed (win grows with BS).

## Dispatch (fp32-fit, generalizes to bf16/fp16)
K512:  4K M4R1 | 8K M3R2a1.3 | 16-65K M3R1 | >=131K M2R2a2.0
K1024: 4K M6R1 | 8K M4R1     | 16-65K M3R1 | >=131K M2R2a2.0
K2048: 8K M3R2a1.3 | 16K M2R2a2.0 | 32K M4R1 | 65K M2R2a2.0 | 131K M4R1 | 262K M2R2a2.0

## Summary
- Exact, no-regression, all-dtype, all-BS win over the single-CTA baseline;
  avg ~1.13-1.14x (event), nsys 1.10-1.45x on spot cells.
- Decisive lever: CDF-aware round-1 threshold placement (offline 5-seed fit)
  — uniform/dyadic placement was 0.93x avg (falsified).
- The L2 trap bounds the ceiling: baseline's extra secant passes are L2-warm
  (~5x cheaper than cold), so only the tighter-threshold P4-shrink survives.
- vs op17 cluster portfolio: lower BS=1 peak (1.10-1.45 vs 1.21-1.67 nsys) but
  single-CTA, stable, wins at ALL BS (op17 degenerates BS>=32).
