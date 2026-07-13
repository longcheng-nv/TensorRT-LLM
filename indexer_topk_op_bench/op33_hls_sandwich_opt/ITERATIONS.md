# op33 iterations — HLS-op27 sandwich optimization (target: beat op27_hls avg +30%, BS=1 fp32)
Env: B200 sm_100 GPU1, cold-L2 flush (256MB) + CUDA-graph (L1 screen); nsys = ship arbiter.
Incumbent = op27_hls (gvr_ms_auto @ op27 HEAD, OP21_FB_LOGFALSI=1 OP27_K2048_TAIL=1).

## iter0 — 2026-07-13 — SETUP: bucket + full-context save + harness validated
- Created op33 bucket; saved PLAN.md + SESSION_CONTEXT.md (full carryover) per user "保存所有上下文".
- Harness (scripts/harness.py) reuses harness/sweep_nsys.build_call("gvr_ms_auto",...) + cold-L2 +
  tie-aware exactness. VALIDATED: op27_hls K512 N8192/16384 BS=1 fp32 build+exact OK, ms_path=ms_1cta.
- L1 baseline (NOISY): K512 N8192 ~18µs / N16384 ~20µs. nsys baseline pending (the A/B floor).
- Next: (1) nsys baseline over K512/1024/2048 × N seqlen at BS=1 fp32 (the +30% floor);
  (2) D1 probe = warp/register band tie-select (sglang INSIGHTS-P3, P4 barrier-bound) — highest promise.

## iter1-4 — 2026-07-13 — D1-D4 knob A/B (nsys, BS=1 fp32, 9 cells K512/1024/2048 × N8192/32768/65536)
Baseline op27_hls (nsys cold-L2): K512 N8192 11.5µs / K1024 11.8 / K2048 12.7µs (grows with N,K).
speedup = t(op27_hls)/t(cfg); env A/B on the SAME incumbent kernel (no new kernel — sandwich already
has the borrow ideas built in).

- **D1 warp/register band tie-select — ALREADY DEFAULT (p4_smallbin=True)**. A/B p4_smallbin OFF
  = geomean 0.865 (−14%) → D1 already contributes ~14%; it IS the sglang INSIGHTS-P3 warp-ballot
  (cnt≤32) / register-rank (≤128). NO new headroom; FALSIFIED as a new lever (incumbent has it).
- **D2 fewer thresholds (qfracs M) — the ONLY positive lever**. d2_qm2 (OP25_QFRACS=0.85,0.35 → M=3)
  = geomean **~1.10 (+10%)**; d2_qstock (0.75,0.5,0.25 M=4) = 0.87 (WORSE — the tail ladder beats
  stock). Mechanism: M=3 fused count is cheaper than M=4 (count_ge_multi_bench M4=1.15-1.46×) and at
  short-N the 4th column doesn't earn keep. CAVEAT: overrides the K2048 tail ladder for ALL K —
  exactness must be verified per-K (esp K2048 all_ge holes) before any ship.
- **D3 256-hist cost (qbins) — NEGATIVE**. qbins128 = 0.96, qbins64 = 0.94 (both slower). Coarser
  bins lose rung precision (op32-F5: linear 256 already near-optimal) and the hist scan is NOT the
  BS=1 bottleneck; fewer bins = wash-to-loss. FALSIFIED.
- **D4 p4/slot — incumbent defaults optimal**. p4_rs OFF (snap) = 0.94 (rank-scatter default wins);
  slot_scale=1 = 0.99 (slot_scale=2 default ~flat-to-better). No headroom.

## VERDICT — 2026-07-13 — +30% NOT MET by knob-tuning; incumbent well-tuned; D2 ~+10% only positive
No single knob reaches the +30% target. Every default-OFF A/B is SLOWER → op27_hls is already at its
tuned optimum. The one positive lever = D2 M-reduction (M=4→M=3) ~+10%, but it trades the K2048 tail-
ladder robustness for speed (exactness-gate pending per-K). Borrow ideas (D1 warp/reg select) are
ALREADY in the incumbent. +30% would require a STRUCTURAL change at BS=1 single-CTA, which op32
established is walled (dram 0.06% / issue 15% / near-optimal barrier chain; F1-F5). Pre-authorized
negative: op27_hls (+ optional D2 M=3 for a conditional ~10%) remains the best option; +30% infeasible
by tuning within the HLS-op27 framework. REPORT.html = the temporary deliverable.

## iter5 — 2026-07-13 — D2 DEEP-DIVE → conditional dispatch SHIPPED (exact, +6% overall)
Full D2 nsys (BS=1 fp32, base vs M=3 qfracs=0.85,0.35):
  K512  N8192 1.112 / N32768 1.143 / N65536 1.051   (M=3 wins)
  K1024 N8192 1.118 / N32768 1.104 / N65536 1.026   (M=3 wins)
  K2048 N8192 0.996 / N32768 0.981 / N65536 0.892   (M=3 LOSES — needs the tail ladder)
Exactness gate M=3 (scripts/gate_m3.py): **48/48 PASS** incl adversarial hr=0/hr=1 beta rows
  (sandwich M0==0 fallback keeps it correct even with the deep column removed).
Dispatch = `gvr_ms_op33.gvr_ms_auto_op33`: **M=3 (0.85,0.35) iff K<2048, else op27_hls default**
  (ONE rule; K2048 byte-identical). VALIDATED exact 12/12 (val_dispatch.py). Result vs op27_hls:
  **geomean 1.060 over all K / ~1.093 over K512/1024** (K2048 unchanged). NOT the +30% target, but
  the ONLY exact positive lever within the HLS-op27 framework.
Mechanism: M=3 fused count cheaper than M=4 (count_ge_multi_bench M4=1.15-1.46×) + at short-N the
  ship qfracs' deep 0.048 column doesn't earn keep for K512/1024; K2048's tail column DOES (all_ge).
FINAL: +30% infeasible by any lever (borrow ideas already in incumbent; structural wall = op32).
  Shippable deliverable = op33 conditional dispatch, +6% overall / +9% K512/1024, exact 48/48.
