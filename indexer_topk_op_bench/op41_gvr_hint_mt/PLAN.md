# op41: GVR hint-multithreshold campaign (2026-07-23, umb-b200-045)

## Goal (user brief, 2026-07-23)
Keep the GVR skeleton — mine preIdx heuristics for MULTI-threshold estimation
-> ITERATE to high-quality thresholds -> refine to exact top-K — equivalent
per-phase algorithm changes allowed, skeleton fixed. Continue optimizing the
op39 envelope (75 cells x BS2-1024 vs bs_real_layers pr anchors; combined
dispatch record = op39 e6 gm 1.3179 / mean 1.3564, 0/750 inexact).

## Prior hard lines (FALSIFIED ledger — do NOT re-cross)
- row-sample quantile at sample rank r<64: undershoot storms (op39 a2/a5).
- min-hint-only threshold: candidate blowup (31/75 cells overflow).
- dispatch on hit-rate: hit unknowable at inference (feedback memory).
- CDP2 / -rdc in this kernel family: 15-20% global tax (op39 iter14).
- cp.async smem double-buffer collect: smem round-trip loss (op39 iter13).

## Levers this campaign opens (not falsified)
- hint ORDER STATISTICS (full ladder, not just min) — values known in-kernel.
- exact-count feedback iterate (collect pass computes TRUE count in ovf[row])
  — bypasses the r>=64 sampling-variance wall entirely.
- L2-resident domain (npad*BS*4B << L2 126MB): re-reads are cheap -> in-kernel
  multi-round threshold iterate is nearly free there; DRAM domain stays
  1-pass (sampling) discipline.

## Phase 0 (crux): offline study on real captures
scripts/hint_study.py: per cell — hit-rate h, counts at hint ranks
m in {K/8..K}, viability of fixed-rank vs ladder+iterate, L2-domain split.
Decides the kernel design (fused single-launch iterate path vs K0 upgrade).
