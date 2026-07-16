# ITERATIONS — op35 APEX top-K

## iter 0 — 2026-07-16 — GO (rung-0.1 floor calibration, b200-038 GPU0/1)
Hypothesis (ledger: fresh): frontier sits 3.17x above info floor; a 1-pass
filter kernel can approach the floor incl. at BS=1 (where op32/op34 walls were
GVR-skeleton-specific, not physical).
Probe: rung0 microbench floor_probe.cu (k_empty / k_stream_reduce / k_filter_append,
warp-agg append + last-CTA ticket). Event L1 + nsys L2 (30 reps, cold-L2, GPU1).
Result (nsys median span us; frontier = per-cell 6-arm best, us_span):
  shape                R      F    frontier  F/R-tax  frontier/F
  BS1    N131072     3.62   4.51    11.6      1.25      2.57x
  BS1    N262144     3.65   5.22    15.7      1.43      3.01x
  BS1    N1048576    4.66   7.84    20.6      1.68      2.63x
  BS32   N262144    10.11  18.26    24.0      1.81      1.31x
  BS256  N262144    54.91 100.40   105.7      1.83      1.05x
  BS1024 N65536     54.08  74.67    93.0      1.38      1.25x
  empty-kernel span ~1.0us; event floor 8.3-8.5us (host launch, L1 axis only).
Diagnosis: BS=1 wall falsified for lean 1-pass design (GO). Filter inner loop is
the bottleneck at throughput shapes: dynamic-index register buffer spills to
local + warp-prefix coordination runs ~every group. Pure-read ceiling measured
4.9-6.3 TB/s (big BS), 3.3 TB/s @BS32 (wave ineff., fixable), ~0.9 TB/s @BS1 1M.
Ledger write-back: WALLS candidate removed (BS1 launch floor NOT a wall down to
~4.5us span); Opt-L revival PARTIALLY validated (append at 3% admit is cheap at
BS1, coordination too heavy at big BS -> v2 = SMEM-atomic staging).
Next: iter 1 filter v2 (SMEM staging, target F/R <= 1.10) + rung0.2 FR band math
on real captures + rung0.3 rival source study.
