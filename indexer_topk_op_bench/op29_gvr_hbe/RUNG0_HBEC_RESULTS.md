# HBE-C rung-0 CRUX — hint-ladder placement replay: **GO** (2026-07-13, node 072)

Spec: DESIGN_HBEC_HINT_LADDER.md §6 rung 0. Scripts:
`scripts/replay_hbec_ladder.py` (C0 host replica, P1b-faithful) +
`scripts/parse_hbec_rung0.py` (policy sweep + GO verdict).
Data: `results/hbec_rung0/rung0.jsonl` (3 arms × 45 rr bundles × 60 realcap
rows; `rung0_v1_noW3a.jsonl` = first pass without the w3a arm).

## C0 replica (what was simulated, bit-faithful to the intended kernel)
preIdx stride-4 subsample (K/4 gathers, +1 mod N for cr=1, invalid skipped)
→ 256-bin hist over [lo,hi] of the subsample, `inv=(255+0.99)/rng` →
suffix-scan; rung m = bin LEFT edge at the `frac[m]×total` ge-crossing →
non-descending clamp. Bracket judged on FULL-row ground truth counts.

## Arms
- **ship**: per-K op25/op27 ship fracs — K512/K1024 (0.92,0.45,0.048),
  K2048 tail (0.75,0.45,0.048)
- **w3a**: (0.92,0.45,0.048) for ALL K  ← **winner**
- stock (0.75,0.5,0.25): reference only

## Policy result (collect rung × cap)
- **collect@loosest (c0, as DESIGN'd) is mandatory**: bracket lands on the
  LOOSEST rung in most real rows (rr real/best 15/15, pro realcap 28/30) —
  a mid-rung collect would miss 63-100%. Design's "append ≥ v3(lowest)" is
  exactly right.
- **cap 32×K entries** is the smallest tested cap that passes everywhere.

## GO line (w3a, collect@loosest, cap 32×K) — target E[p]≤1.2, miss≤10%
| slice | rows | miss | E[passes] |
|---|---|---|---|
| rr-real all N (65536..1M) | 15 | **0.0%** | **1.00** |
| rr-real pilot N≥131072 | 12 | **0.0%** | **1.00** |
| realcap all rows (3 models) | 60 | 1.7% | 1.03 |
| REAL AXIS (rr-real+realcap) | 75 | **1.3%** | **1.03** |
| guard rr-worst / rr-best | 15/15 | 0% / 0% | 1.00 |

ship-arm identical except K2048: top frac 0.75 < h=0.82 on rr-real
N=524288 → lt_K miss (cnts [1787,947,98] all <K). w3a's 0.92 top column
closes it (op30 finding "worst pole = hr 0.85-0.90" says h>0.75 is common
on the real axis; K2048 HBE-C must NOT reuse the op27 0.75 top column —
that geometry was tuned for the HLS fallback economics, not for bracket
coverage).

## Costs the kernel design must carry (measured tails)
- **Resolve set** (mini-hist atomics) cand@r*: rr med 1.4-3.7×K, realcap
  flash p90 7.5×K — fine for a 4096-bin DSMEM mini-hist.
- **Candidate buffer** cand@loosest: flash realcap med 17×K / p90 25×K
  (h≈0.15-0.45 + left-edge bias + subsample noise inflate the loosest-rung
  count well past the f/h model); v32 h≈0.01 layers 20-29×K (w3a max 33×K
  = the single realcap miss). Buffer = 32×K entries/row → per-CTA (÷8)
  smem 16/32/64KB @ K512/1024/2048 at 8B/entry — feasible on B200 227KB,
  spill path per DESIGN stays as fail-soft.
- realcap rows have N=14478/25154/70690 (< pilot floor mostly) — they
  stress the ESTIMATOR, not the prize domain; the prize cells (rr-real
  cluster N) are clean 0-miss.

## Next (per DESIGN §6)
rung 2 microbench: DSMEM M×8-scalar reduce vs 4096-bin all-reduce @ BS=1;
then rung 3 tier-5 kernel behind flag, gate 3-track + pilot + nsys.
