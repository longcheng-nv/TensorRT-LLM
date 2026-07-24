# ITERATIONS.md — op42 GVR BS=1-1024 (base 28dc11f6, incumbent gvrpkg_04a0)

## iter 0 — 2026-07-24 — SETUP + node anchor + measurement artifact
Hypothesis: n/a (P0). Chain bring-up on umbriel-b200-073 (4x B200 idle).
Result: base 28dc11f6 copy compiles + exact on real cells; BS=1 event-axis
anchor flash_128k_L36 bsx/head = 1.62-1.66 (record: gm 1.65) — direction ok.
ARTIFACT FOUND: head (cuteDSL) event-axis numbers include HOST issue latency
— pro_1024k BS<=64 events read ~1.2ms while nsys pure-kernel shows 24us
(matches op37 b200-026 anchor 23.5us). Reproduced on GPU0+GPU1, all layers.
Mechanism: mCTA cluster variant host-side launch cost; BS>=128 variant ~92us
events. VERDICT: event axis BANNED for any ratio involving the head arm;
smoke mode only for bsx-vs-bsx iteration. nsys = only A/B axis (matches
production CUDA-graph deployment where host cost is hidden).
Ledger write-back: FALSIFIED.md M-entry added.
Next: iter 1 nsys screen.

## iter 1 — 2026-07-24 — IN PROGRESS (nsys screen running)
Hypothesis (ledger: FALSIFIED #1 revival = kernel-side row parallelism):
minimal grid.y batching — one cluster per row along grid.y, single launch,
per-row pointer offsets; per-tier CS/TB/MAXV unchanged.
Probe (rung 2, event axis, bsx arm only — valid intra-arm):
  flash_128k_L36 (CS8 tier): BS1-8 flat ~14.4us (seq was 73us @BS8);
  BS64 35.6us; BS1024 383us (seq was 8347us — 22x better, still wave-bound:
  4096 CTAs / 148 SMs = 28 waves x ~13us).
  flash_4k_L36 (direct, 1 CTA/row): wins at every BS (events; nsys pending).
Exactness: 0 inexact on all smokes so far (value-multiset vs torch.topk).
Structure identified for BS>=64: __launch_bounds__(TB,1) + CS CTAs/row =>
waves = BS*CS/148; per-wave ~= BS1 latency. Lever for iter2: BS-aware tier
re-dispatch (minimize CTAs/row at large BS: CS=1 dense tiers w/ TB=1024 or
larger MAXV; K-aware). nsys 12-cell x 11-BS screen sharded on GPU0-3.

## iter 2 — 2026-07-24 — PARTIAL (kept for BS 8-31 band)
Hypothesis: dense min-CS tiers amortize per-wave fixed cost. Result (nsys, 7
cells): BS>=32 lifted (0.23->0.40 @1024) but SATURATES — wall = per-row WORK
(8-rung ALU x secant iters vs head ~1-pass). BS8 dense wins only CS16 tiers.
Ledger: W2 added (per-row work wall). 96/96 exact.

## iter 4 — 2026-07-24 — SHIP-CANDIDATE DIRECTION (superseded by iter5 line)
Hypothesis: throughput form — CS=1, 2 CTA/SM, fused P2-count+P3-collect at
pivot=hmin (count>=K by construction), atomic push, P4 in-CTA. nsys 9 cells:
BS1024 gm 0.86 (0.40 prior); pro_1024k 1.19, flash_4k 1.83; laggards
flash_1024k 0.65 (pivot overflow -> re-stream on low-hit rows), pro_128k 0.63,
v32 0.64-0.75. 60/60 exact. Skeleton intact (P1/P2/P3/P4 + plateau).

## iter 5 — 2026-07-24 — MIXED (bug found, fix queued)
Hypothesis: sampled 8-rung pre-pass calibrates pivot + fused pass R=8->2.
Result (nsys): flash_128k 0.84->1.07-1.13, v32_256k 0.84->1.50 (fat rows
fixed); BUT v32_64k 0.64->0.33, v32_16k 0.75->0.62 (+~1 full-row stream).
ROOT CAUSE (code review): sample stride `i += T*SS` samples the FIRST 512
contiguous float4s of the row (+ tail), not every-SS-th — positional bias
onto the attention-sink region -> est overshoots -> pivot too high ->
count<K -> hmin re-collect stream. Fix: idx = v0 + (j*T + tid)*SS uniform.
Ledger: (positional-biased row sampling fabricates threshold estimates on
real decode data, domain: any sampled-pivot scheme, evidence: nsys iter5
v32 cells + code) — measurement/algorithm artifact. 24/24 exact so far.
