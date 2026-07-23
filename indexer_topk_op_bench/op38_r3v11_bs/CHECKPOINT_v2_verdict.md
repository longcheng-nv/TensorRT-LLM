# op38 v2 nsys verdict (2026-07-23, umb-b200-045, 8-GPU sharded sweep)

Goal: r3_v11 batched vs PR head, BS 2-1024, **mean >= 1.8x AND all cases >= 1.0x**.

## Verdict: v2 FAILS both bars

- 75 cells x BS{2..1024} = 750 BS>1 cases, **0 inexact** (750/750 + 75 BS=1 exact)
- BS>1 **geomean 1.2831, mean 1.3262, min 0.6524, 137/750 < 1.0**
- BS=1 (ref only, vs report pr): geomean 1.5877, min 1.0481 — BS=1 path unchanged, healthy

Per-BS geomean / #losing (of 75):

| BS | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|----|---|---|---|----|----|----|-----|-----|-----|------|
| gm | 1.565 | 1.548 | 1.508 | 1.268 | 1.238 | 1.289 | 1.222 | 1.075 | 1.098 | 1.136 |
| <1.0 | 1 | 1 | 1 | 14 | 17 | 3 | 15 | 34 | 31 | 20 |

## Loss structure

1. **Un-probed layers dominate the worst list**: dispatch v2 was tuned on
   L22/L30/L34 probe cells; the worst cells are pro L46/L14, v32 L54, pro L30
   at *other* isl — same (npad,K,BS) key, different data distribution
   (hit-rate). Per hard rule we may NOT dispatch on hit-rate.
2. **Two losing bands**: BS 16-32 (mid transition: reg ladder -> CS1-reg) and
   BS 256-1024 (streaming tier), worst pro_1024k_L46 0.652-0.743x across
   BS 128-1024, pro_512k_L46 0.731-0.795x.
3. BS 2-8 is essentially solved (gm ~1.5, 1 loser each: 0.986/0.969/0.950 —
   within noise band of 1.0).

## Open question for v3

Is the BS>=16 loss fixable by better variant choice (probe_cfg ladder on the
losing cells), or is it a structural arm-ceiling (no (TB,CS,MAXV,AR,HS) variant
beats pr on low-hit-rate layers at high BS)? If the latter, the 1.8-mean bar is
infeasible and the campaign converts to a bounded-verdict close-out
(cf. op37: BS=1 was the only win region, crossover BS=2).

Data: `v2_data.csv` (750+75 rows). Reproduce: `bash drive_v2_sweep.sh v2` then
`python3 verdict.py --tag v2`.
