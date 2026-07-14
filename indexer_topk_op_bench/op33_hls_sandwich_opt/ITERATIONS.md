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

## iter6 — 2026-07-14 — RETRACTION: M=3 dispatch is NO-SHIP (worst regresses); iter5 was WRONG
User asked if op33 was tested on the FULL op22rr synth grid. It was NOT — iter1-5 used ONLY
scenario=real at N≤65536 (9 cells). Two measurement failures exposed:
1. **real-only subset** hid the worst-axis regression (the biggest error — violated "report all 3
   verdict axes [worst,real,best]").
2. **8-GPU-saturated full-grid** (156 cells) CORRUPTED the base/m3 ratios (base & m3 ran on
   different GPUs under different load) — produced FAKE outliers (K1024 N32768 real 0.227 that the
   clean paired A/B shows is actually 1.108).
CLEAN single-idle-GPU PAIRED A/B (base vs m3 back-to-back, the only trustworthy verdict):
   K512  N8192   worst 0.884 | K512  N32768 worst 1.171 | K1024 N32768 worst 0.787
   K1024 N32768  real  1.108 | K512  N262144 worst 0.727
VERDICT: M=3 WINS real (~1.1) but REGRESSES worst on 3/4 cells (−12% to −27%, worst at large-N).
   FAILS the ship rule (worst must not regress). The deep 0.048 column M=3 removes IS earning its
   keep on worst (op27 tail-ladder design). **op33 D2/M=3 = NO-SHIP. iter5 "+9% ship" RETRACTED.**
   op27_hls (M=4 tail ladder) remains the best safe default.
MEASUREMENT LESSON (both omni-kernel violations): (a) NEVER headline one verdict axis — real-only
   hid a worst catastrophe; (b) 8-GPU-saturated sweeps corrupt A/B ratios — use single-GPU PAIRED
   back-to-back A/B for any ship verdict (cross-run/cross-GPU ratios are noise until same-GPU-paired).

## FINAL VERDICT — 2026-07-14 — op33 CLOSED, NO-SHIP (+30% infeasible, no valid lever)
No borrow (D1 already default) and no knob (D2 M=3 fails worst; D3/D4 negative) beats op27_hls on
the full 3-axis envelope. +30% infeasible. op27_hls remains best. Pre-authorized negative delivered.

## iter7 — 2026-07-14 — DEFINITIVE clean full grid: NO-SHIP confirmed, safe region NONE
Re-ran the full op22rr fp32 BS=1 grid CLEANLY on umbriel-b200-027 (2-way sequential, base+m3
back-to-back same GPU — the reliable method after 8-GPU contention + 8-way nsys-flakiness were both
falsified as measurement-corrupters). 54/54 cells, K512/1024 × 9N × best/real/worst.
M=3/base speedup (>1 = M=3 faster), geomean over K512/1024:
  N       best   real   worst
  4096    1.124  1.103  0.947
  8192    1.132  1.115  0.808
  16384   1.197  1.157  0.992
  32768   1.125  1.123  0.958
  65536   1.067  1.032  0.717
  131072  1.047  1.043  0.711
  262144  1.061  1.065  0.725
  524288  1.059  1.062  0.685
  1048576 1.099  1.099  0.667
M=3 wins best/real at EVERY N, LOSES worst at EVERY N (0.67-0.99). SAFE region (all 3 axes >=1) =
NONE. Overall geomean 0.982 (net negative incl worst). CONFIRMS iter6 NO-SHIP definitively.
dispatch STAYS default op27_hls (no M=3). Report REPORT.html refreshed with this clean grid as the
authoritative table. Data: results/reliable_grid.csv. op33 CLOSED.
