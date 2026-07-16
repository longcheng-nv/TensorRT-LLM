# op35 falsification ledger (inherits op-bench global ledger; see PLAN.md red lines)
Inherited (do not re-tread): ms fused count+collect (1.47× slower, 07-15);
block-max separate prepass (op31 fixed-tax); P4-internal refine; SMEM residency
(L2 trap); P1 self-loop; multi-threshold P2 k=4; cluster DSM high-BS; small-N
single-CTA micro-opt ≤4% (op8/op32/op34).
- [op35 iter1] H3-tail qfracs (0.75,0.45,0.048) on PR R0+vseed: 0.95-0.97 event geomean,
  slower on ALL axes incl worst+real v32. Root cause: vseed already covers tail interior;
  +1 count column = pure P2 tax. Domain: K2048 fp32 BS=1. Evidence: event-paired ×30.
  Revival condition: none (superseded by vseed).
