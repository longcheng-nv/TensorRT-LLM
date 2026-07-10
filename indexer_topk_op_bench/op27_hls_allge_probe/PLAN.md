# op27 — HLS all_ge M-ary bracket probe (algorithm-first robustness campaign)

Date: 2026-07-10. Node: umbriel-b200-027 (8×B200 idle). Branch: omni/op21-gvr-prod.
Baseline = op25 ship config (w3a ladder + slot2@n<64K + fp32 C8, OP21_FB_LOGFALSI=1,
kernel src `op21_gvr_prod/src/gvr_ms_op.py` / `gvr_msc_op.py`).

## Problem (op22rr re-test, sections 1-2)

WORST (op24 adversarial pole, beta_shallow hr 0.05) losing cells vs GVR (cuteDSL)
baseline, seqlen BS=1 fp32 — ALL on the paths without cheap fallback:

| cells | path | base/op25 |
|---|---|---|
| K512  N4K-32K   | ms_1cta | 0.68-0.91 |
| K1024 N8K       | ms_1cta | 0.80 |
| K2048 N8K-32K   | ms_1cta | 0.83-0.99 |
| K2048 N64K-131K | msc_C4  | 0.91-0.94 |

Mechanism (op24 + MECH_FINDINGS + count_ge_multi_bench): worst rows are all_ge
(every ladder rung count > kC) -> fallback fires per row; below the dist gate the
fallback is SERIAL full-row scans (entry + hi-check + geometric expansion <=12 +
falsi), single-CTA priced. The plain baseline wins because its mean-stash seed
lands in [K, kC] first pass (ev1).

## Fix (algorithm, NOT dispatch)

When the R0 bracket is missing an end (all_ge / all-low), replace the serial
expansion with ONE M-column probe using the EXISTING `block_count_ge_multi`
primitive: probe thresholds extrapolated in value space from the ladder's own
(thr, count) points via the log-linear CCDF slope (log-count falsi already relies
on this ~exponential tail), targets spanning mstar..mstar/4^(M-1). One probe scan
(tau(M): fp32 x1.25@M4; ~free at N<=32K latency-bound per count_ge_multi_bench)
replaces up to 12+ serial scans; the probe counts hand log-falsi a proper bracket
-> <=1-2 landing passes.

Constraints (user directive 2026-07-10):
- NO data-dependent dispatch tables; improve the algorithm in-path.
- Zero regression on already-won cells (best/real all dtypes x K; P0 grid).
- Grow the winning envelope (scenario x shape x dtype x K) incrementally.

## ITER0 VERDICT (2026-07-10, host replay — screen_mprobe.py)

The M-ary probe hypothesis is FALSIFIED for K512/K1024: the op25 w3a 0.048
column already brackets the worst pole — worst rows are mode `fast` or
`overflow`, and every fallback converges in 1 log-falsi pass. K2048 worst is
all_ge at every N (stock ladder) but the host model also lands in 1 pass.
Silicon evidence relocates the loss:
  - op25 loses on BEST small-N too (K512 4-16K: base/op25 0.87-0.97 while
    base/op21_hls is 1.04-1.15) => a DATA-INDEPENDENT w3a+slot2 machinery tax
    of 10-17%.
  - worst-vs-best delta for the whole HLS family (~1.7us at 16K) = miss-path
    cost: overflow/all_ge forfeit fused-collect -> classic P3 rescan + wide
    acceptance band [K,kC] -> barrier-heavy P4; base is FASTER on worst (ev1,
    small band).
=> iter1 pivots to a silicon DECOMPOSITION A/B (ab_decomp.py, b200-027 GPU0):
   plain / stock_s1(pre-op25) / w3a_s1 / ship (+ tail_s1 for K2048) on
   best/worst/real x fp32 BS=1 small-N cells. The fix will target whichever
   knob owns the tax (ladder placement vs slot gating vs K2048 tail column) —
   still algorithm/parameter work, no data-dependent dispatch.

## Steps

1. RECON (done): losing cells mapped to ms_1cta/msc_C4; fallback code paths read
   (gvr_ms_op.py phase3 override L1250-1394; ladder finalize L1857-1895).
2. HOST SCREEN `screen_mprobe.py`: fb_mprobe vs fb_logfalsi (ship) on op22rr
   worst/best/real bundles, pass-count + model-us accounting.
   Gate: worst all_ge rows converge in <= tau(M)+2 pass-equivalents; fast/real
   rows unchanged (probe only fires on missing-end fallback).
3. KERNEL: wire probe into gvr_ms_op.py (ms + msc share the phase3 override);
   knob OP27_FB_MPROBE (default off until gates green) for paired A/B.
4. GATES + nsys A/B on 027: exactness (all arms, BS=1 per (K,dt,N)); paired
   cold-L2 worst+real+best, fp32 first then bf16/fp16; ship rule:
   worst ms_1cta cells >= plain baseline, best/real gm >= 0.99x ship, P0 grid
   unchanged. Envelope expansion order: fp32 seqlen -> 16-bit -> bs grids.

## Non-goals

- P2 M-ary refine (falsified: Opt-F, op8; log-falsi already 1.00 mean passes).
- Dispatch-table routing on data statistics.
- K2048 (ladder,kC,cap) geometry redesign — separate follow-up if probe doesn't
  close the msc_C4 65K-131K cells.
