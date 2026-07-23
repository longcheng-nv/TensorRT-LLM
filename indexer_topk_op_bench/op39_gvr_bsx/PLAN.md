# op39: GVR top-K BS=2-1024 — throughput-arm campaign

## Objective triple (set by user 2026-07-23 via /goal; agent MAY NOT relax)

```yaml
objective:
  incumbent: PR head GvrTopKKernel (bs_real_layers.csv nsys anchors, §7b envelope)
  rivals: [op38 dispatch v3 (own prior), radix, sglang_v2 (context only)]
  envelope: {cells: 75 (§7b fp32 per-layer), BS: [2..1024], dtype: fp32}
  verdict_axes: [worst, real, best]   # real = §7b captures; worst/best via op24/op30 extremes
  ship_rule: "BS>1 arithmetic mean >= 1.8x vs PR head AND all 750 cases >= 1.0x
              AND 750/750 tie-aware exact AND dispatch keys within budget (<= op38 v3 + 3)"
  hard_constraints: [GVR framework (hint -> threshold -> exact refine) stays,
                     BS=1 path untouched (op37 verdict carries over),
                     baseline r3_v11 recoverable via git]
```

## Feasibility prior (Phase 1.4, 2026-07-23)

- **Regime flip vs op38's search space**: at BS>=256 the per-wave working set
  (BS x npad x 4B = 0.1-1.1 GB) >> L2 (126MB) -> every re-scan pass costs DRAM.
  The op14 L2-trap veto is domain-scoped to BS=1/small-N and does NOT apply here.
- Measured from op38 v3_data: PR head effective BW at BS>=256 median 1.20 TB/s,
  p90 2.48; candidate v3 the same (1.26). Ideal 1-pass @7TB/s UB vs pr:
  median 5.8x. Envelope mean projection if a 1-pass arm lands 60% of UB at
  BS>=256 only: **mean 3.46** — the 1.8 bar has prior headroom.
- Caveat: small (BS x N) cells (< ~256MB) are not DRAM-floor-bound; UB there is
  launch/latency-limited. Segment before trusting projections.
- Rung-0 crux (must run first): NCU dram bytes on pr + cand at a big cell
  (pro_1024k BS512). If pr already reads ~1x bytes, the lever is not traffic —
  re-diagnose (issue efficiency / occupancy) before designing the arm.

## Red lines (from ledgers)

- FALSIFIED (op38): (TB,CS,MAXV,AR,HS) ladder of the r3_v11 skeleton at
  BS 16-1024 — 52-variant full ladder, 133/137 losses unfixable. Revival
  condition: none within that parameter family; new arm shapes only.
- WALL (op37/op38): r3_v11 latency shape's deep-win region is BS=1; BS>=256
  losses concentrate on low-hit-rate layers (pro L46/L14, v32 L54).
- Hard rule: hit-rate is not dispatchable (inference-time unknowable).
- Judge: nsys-vs-nsys only; report-pr anchors from bs_real_layers.csv;
  single-GPU paired A/B for ship verdicts; saturated sweeps = L1 screening.
```
