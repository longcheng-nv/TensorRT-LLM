# pro/512k (K1024, N=131075) +45% regression — root cause & fix proposal

Campaign: 2026-07-20, b200-027. Regression: pre-vseed 16.1µs -> shipped
23.2µs (nsys cold, x3 median), the single deepest cell on the PR head.

## Bisect (4 builds, same-process paired, nsys x3 median)

| build | pro/512k | pro/256k | pro/1024k | flash/512k |
|---|---|---|---|---|
| snap @018251950f | 16.09 | 13.51 | 19.41 | 20.41 |
| vseed @88a563b145 | 15.99 (INEXACT 3/3) | 14.31 | 19.66 | 20.35 |
| ptail @eae374554c | **23.19 (+45%)** | 14.34 | 19.72 | 20.37 |
| head @0d6fc4f1f2 | 22.04 | 14.36 | 19.68 | 20.33 |

- vseed is FREE here (0.994) — the "q.35 removal / ladder" hypothesis is
  REFUTED (restoring r0_qfracs=(0.85,0.35) on head: 20.4µs, no help).
- The vseed build is INEXACT on this cell (the boundary-tie defect the
  p4_exact_tail commit itself cites: "real Pro 512k-ISL captures").
- The ENTIRE +45% is p4_exact_tail's FIRE-PATH cost: this real cell has a
  genuine 2-element boundary tie every step -> the ambiguity gate fires on
  every launch.

## Why the fire path costs ~5.3µs for a 2-element tie

`p4_exact_tail` resolves the tie class with an UNCONDITIONAL 4-level
MSB-first 8-bit radix select: each level = 256-bin zero + a FULL candidate
scan (membership recompute) + barriers. 4 candidate passes ≈ 5.3µs warm at
this config regardless of tie multiplicity. Early-exit-on-divergence would
NOT rescue this cell: the tie pair (gap 3.04e-6 at |v|≈0.289) shares order-
key digits through level 2 and diverges only at level 3 — radix needs all
4 passes by construction.

## Fix proposal: tiny-tie collect+select fast path (validated mechanism)

Inside the exact-tail fire branch, gate on tie-class size:
  cnt_strad <= CAP (e.g. 128): ONE candidate pass collects the (b*, sb*)
  class (value_bits, cand_idx) into SMEM scratch; thread0 selects top-need
  by full 32-bit order-key compare and rewrites [ra_fine, kK).
  cnt_strad > CAP: keep the existing 4-level radix select (bounded backstop).
This is exactly the v4 mechanism implemented and validated in this dir
(battery 164/164 incl. CAP/CAP+1 boundary; collect+select fire cost ≈ one
candidate pass ≈ 1.3µs vs 5.3µs). Expected pro/512k: 23.2 -> ~16.5-17µs
(recovering ~80% of the regression) with exactness unchanged; all
non-firing cells byte-identical.

## Notes

- Composite exposure today: pro/512k is the only bench cell that fires
  (25-cell A/B: all other ptail/vseed ratios ~1.00); the 9 old per-layer
  fixture cells fire at head too. Envelope-relevant (N≈131K).
- The K2048 rung recalib (@0d6fc4f1f2) already recovers 5% here (0.950)
  incidentally.
