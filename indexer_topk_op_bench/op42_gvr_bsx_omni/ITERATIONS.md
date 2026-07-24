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
