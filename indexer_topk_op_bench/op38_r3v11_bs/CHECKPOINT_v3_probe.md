# op38 v3 ladder probe verdict (2026-07-23, umb-b200-045)

Full (TB,CS,MAXV,AR,HS) 52-variant ladder on all 137 v2-losing (cell,BS) cases,
plus paired local PR head (anchor check). Data: `v3_probe_s{0..7}.csv`.

## Campaign bars are DOUBLE-LOCKED INFEASIBLE

Bars: BS 2-1024 mean >= 1.8x vs PR head AND all cases >= 1.0x.

- Lock 1 (v2 750-case nsys sweep): mean 1.326, 137/750 < 1.0.
- Lock 2 (this probe): **133/137 losing cases have NO variant that reaches
  1.0x** vs report pr (best-variant gm 0.815, min 0.61); in the deepest losses
  `prod` is already the best variant.
- Oracle projection (per-case best over v2+ladder): gm 1.288 / mean 1.330 /
  min 0.778. Even clamping every residual loser to exactly 1.0:
  **mean 1.342 — the 1.8 bar is structurally out of reach**; min 0.778 —
  the zero-loss bar is also unreachable within this arm family.
- Consistent with op37 (BS=1 is r3_v11's only deep win region, crossover BS=2)
  and compB BS-ext (large-n crossover BS≈8).

## Anchor-drift note (measurement, not machine)

pr_local (CUDA-event around cuteDSL launch) vs pr_report (nsys kernel time)
drift: median 1.365, p95 69x — short kernels absorb cuteDSL host-launch
overhead into the event window. Long kernels (BS>=512 large-N) drift only
1.02-1.04, so the report anchor is sound on this machine. Judge remains
nsys-vs-nsys (hard rule since op7).

## Residual value (harvest for v3 dispatch)

81/137 cases have a better non-prod variant; key-level consistent examples:

- (pro, npad 262144, BS32): all 3 layers -> (1024,4,0,8,1), 1.22-1.23x vs prod
- (flash, 262144, BS32): all 3 layers -> (1024,4,0,8,1), 1.12-1.14x
- (pro, 131136, BS16/32): (1024,4,9,6,2), 1.05-1.39x
- (v32/flash/pro, 65600, BS16-128): streaming AR ladder variants, 1.04-1.20x
- BS16-32 transition band is broadly fixable; BS>=256 mostly ceiling-bound.

Hazard: some keys have split winners across layers (e.g. pro 262144 BS256-1024:
L46 wants (1024,1,0,4,1) +36% but L14/L30 prefer prod). v3 policy: switch a key
only after a confirmation probe over ALL layers sharing the key shows the new
cfg >= prod on every layer.
