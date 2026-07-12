# PLAN — <campaign name>

## Objective triple (human-supplied <date>; agent may not relax)
```yaml
objective:
  incumbent: <production default implementation — A/B always against this>
  rivals: []
  envelope: {N: , K: , dtype: [], BS: }     # outside = stress probes only
  verdict_axes: [worst, real, best]
  ship_rule: "worst improves AND real/best regression-free AND exactness green AND dispatch rules <= 3"
  hard_constraints: []
budget: {gpu_hours: , tokens_usd: }          # calibration: mid campaign ~15 GPU-h + $108
```

## Red lines (from ledger — do NOT re-propose; cite revival condition to override)
<!-- grep FALSIFIED.md / WALLS.md and paste relevant entries here at kickoff -->
- ...

## Feasibility priors (Phase 1.4, run BEFORE proposing directions)
- L2-trap: input_bytes = ____ vs L2 = ____ → traffic levers viable? Y/N
- Math floor: min_passes × bytes / BW = ____ µs vs target = ____ → feasible? Y/N
- Occupancy structure: grid @ envelope BS covers ____ / ____ SMs → structural? Y/N

## Anchor cell
- cell: <shape/dtype/BS> · expected: ____ µs ± 3% · node: ____

## Probe plan (Phase 3 ladder)
| # | Hypothesis | Crux question | Rung-0 tool | GO/NO-GO criterion |
|---|---|---|---|---|
