# ITERATIONS — rmsnorm_campaign

## iter 0 — 2026-07-13 — CHARACTERIZATION (no kernel; SKILL Phase 1 + Phase 3 rung 0)
Hypothesis (ledger check: grep FALSIFIED.md/WALLS.md/../*/FALSIFIED.md → NO HITS, fresh
dense-class ledger — miss noted per protocol): where does the incumbent leave headroom?

Probes:
- Smoke: flashinfer.norm.rmsnorm vs fp32-upcast eager reference, maxdiff 0.0078 < 1e-2 → semantics confirmed.
- L1 bench_cold grid (incumbent vs eager rival): incumbent 2.1-15.9x faster than eager
  everywhere (eager rival eliminated as a threat; T=16384: 79.6 vs 1265.9 µs cold).
  L1 artifact caught: T=16 cold (21.3 µs) > T=256 (19.0) — graph-launch noise; nsys shows
  monotone 2.96/4.04 µs (M1 escalation validated).
- L3 ncu_attrib incumbent @T=16384 (KERNEL_REGEX=regex:RMSNormKernel; without the regex: prefix
  ncu -k matches nothing — gotcha logged): dram_read/input ratio 1.00 → single HBM pass,
  traffic levers VOID. grid 16384x(128 thr), 52 regs, occ 52%, mem throughput ~70% pk elapsed.
- Rung-0 crux P1/P2 (src/probe_copy.py, same-traffic torch elementwise; copy_ lowers to
  DtoD memcpy — gotcha, use torch.mul out=): nsys ceiling vs incumbent:
  | T | copy ceiling µs | incumbent µs | headroom |
  |---|---|---|---|
  | 1 | 1.64 | 2.83 | 1.73x |
  | 16 | 2.25 | 2.96 | 1.31x |
  | 256 | 3.64 | 4.04 | 1.11x |
  | 4096 | 21.54 | 21.82 | 1.01x (saturated) |
  | 16384 | 75.09 | 71.79 | incumbent beats torch copy (6.54 TB/s) |

Result: GO for a Triton candidate. Win region = small T (latency/occupancy regime:
incumbent uses only 128 threads/row); T>=4096 is bandwidth-saturated margin defense
(candidate must stay >=0.98, target = match).
Diagnosis: RMSNorm@7168 is 1-read-1-write; all remaining space is launch/latency/ILP at
small T, ~0-5% BW margin at large T.
Ledger write-back: WALLS.md + FALSIFIED.md #1 (traffic levers void — NCU-locked).
Anchor set: incumbent T=4096 nsys = 21.82 µs ± 3%.
Next: iter 1 = single-pass Triton kernel, 1 CTA/row, fp32 accum, autotune num_warps by T.
