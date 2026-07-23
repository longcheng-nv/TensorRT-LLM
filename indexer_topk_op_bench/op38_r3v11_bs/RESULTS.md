# op38 close-out: r3_v11 BS=2-1024 batching campaign (2026-07-23, umb-b200-045)

Goal (as set): vs PR head over the §7b fp32 envelope (75 cells x BS 2-1024),
**mean >= 1.8x AND every case >= 1.0x**.

## VERDICT: bars DOUBLE-LOCKED INFEASIBLE; v3 harvest shipped in-tree (this dir)

| arm | BS>1 gm | mean | min | <1.0 | exact |
|-----|---------|------|-----|------|-------|
| v2 (probe-tuned 3-regime dispatch) | 1.2831 | 1.3262 | 0.652 | 137/750 | 750/750 |
| **v3 (15 all-layer-confirmed key switches)** | **1.2928** | **1.3324** | 0.652 | **115/750** | 750/750 |
| oracle bound (per-case ladder best) | 1.288* | 1.330 | 0.778 | 133 | — |
| oracle + clamp losers to 1.0 (hypothetical) | 1.306 | **1.342** | 1.0 | 0 | — |

*oracle computed against v2; v3 realizes the key-level-safe subset of it.

- **Lock 1**: 750-case nsys sweep — mean 1.33 vs bar 1.8.
- **Lock 2**: 52-variant (TB,CS,MAXV,AR,HS) ladder on all 137 v2 losses —
  133/137 have NO variant reaching 1.0x (best gm 0.815, min 0.61); deepest
  losses already run their best variant. Even the impossible "clamp all
  losers to 1.0" projection gives mean 1.342 << 1.8.

## Where r3_v11 batching stands (per-BS gm, v3)

BS2 1.555 / BS4 1.538 / BS8 1.497 / BS16 1.307 / BS32 1.303 / BS64 1.286 /
BS128 1.240 / BS256 1.074 / BS512 1.100 / BS1024 1.136.
BS=1 reference unchanged: gm 1.574, min 1.05 (op37 verdict carries over).

- BS 2-8: solved (single 0.95-0.99 noise-edge case per BS).
- BS 16-32: v3 fixed the band (losers 14+17 -> 7+4; min 0.73 -> 0.88/0.95)
  via all-layer-confirmed switches: 4-CTA cluster-reg (1024,4,9,6,x) at
  npad<=131136, 4-CTA streaming (1024,4,0,8,1) at 163776-262144.
- BS >= 256: structural arm ceiling. PR head's row-parallel throughput shape
  wins on low-hit-rate layers (pro L46/L14, v32 L54); no ladder variant
  closes it. Consistent with op37 (BS=1 only deep win region) and compB
  (large-n crossover BS≈8).

## Residual-weakness ledger

- pro_1024k_L46 BS128-1024: 0.65-0.74 (worst family; low-hit-rate deep layer).
- pro_512k_L46, v32_64k_L34/L54, flash_256k_L22 at BS256-1024: 0.84-0.88.
- pro_64k_L30 BS256/512: 0.84/0.87 (its BS2-128 losses were fixed by v3).
- A zero-loss dispatch would need the PR head itself as a fallback arm
  (parity, not a win) — pointless for a beat-the-rival campaign.

## Measurement notes

- Judge = nsys-vs-nsys (report pr from bs_real_layers.csv). Paired local
  CUDA-event PR timings drift 1.365 median (cuteDSL host-launch overhead in
  the event window; p95 69x on short kernels) — event-based pr is NOT a
  valid anchor; long-kernel drift is only 1.02-1.04 so the report anchor is
  sound on this machine.
- v3 confirm-probe gate: a (model,npad,BS) key switches only if the new cfg
  >= prod on EVERY envelope layer sharing the key (hit-rate is not
  dispatchable — hard rule).

Repro: `bash drive_v2_sweep.sh v3` -> `python3 verdict.py --tag v3`
(v3_data.csv); ladder probe `probe_v3.py`, confirm probe `confirm_v3.py`.
