# op34 — GVR-skeleton top-K to beat sglang_v2 by 30% on real V4 decode data

Node umbriel-b200-074 · B200 SM100 (148 SM) · branch omni/op21-gvr-prod · start 2026-07-14

## Objective triple (human-supplied 2026-07-14; agent MAY NOT relax)
```yaml
objective:
  incumbent-to-beat: sglang_v2   (the RIVAL; report vs THIS)
  start-kernel:      op26_r0auto  (iterate FROM here, keep GVR threshold skeleton)
  envelope: {N: [4K..1M real valid], K: [512 flash, 1024 pro], dtype: fp32(native),
             BS: 1, data: E2E_exp/indexer_decode_capture real decode logits}
  test-env: B200, nsys pure-kernel, cold-L2 (512MB evict) — IDENTICAL to op22 REPORT.html
  verdict_axes: [per-cell, geomean-over-layer, GRAND geomean over all (model,ISL)]
  ship_rule / GOAL: GRAND BS=1 fp32 geomean(new GVR) <= sglang/1.30
                    i.e. new <= 6.06us  (need 2.03x over op26_r0auto's 12.31us)
  hard_constraints: [keep GVR threshold-method skeleton, exactness (tie-aware
                     value-multiset vs same-dtype torch.topk), CUDA-graph safe,
                     fail-soft (guaranteed-fill-K)]
```

## Phase-1 numbers (analysis/, committed)
- sglang grand geomean **7.88us**; op26_r0auto **12.31us** (1.56x slower); base 18.60us.
- **0/459 (model,ISL,layer) cells: NO in-tree GVR arm beats sglang.**
- Deficit by regime (need× over r0): small N(4-16K) 2.2-2.6×, mid 1.75-2.1×, large 1.6-1.8×.
- Real data: pro hit-rate high (0.6-1.0), flash lower (0.3-0.76); boundary gap_rel ~1e-5 (razor thin).

## Feasibility priors (Phase 1.4 — from the master GVR falsification ledger; imported to FALSIFIED.md)
Three structural walls already established by prior campaigns bear directly on this target:
1. **Pass-count + occupancy wall (op#8 GVR-turbo INFEASIBLE proof).** radix/sglang = 1 global
   pass × multi-CTA-per-row (4-32 SM/row); GVR = ~2.5 passes × single-CTA-per-row (1/148 SM at
   BS=1). op#8 exhausted 4 levers (bf16 passes, M-ary, smem-resident, sw-pipeline), all EXACT,
   best 0.922× vs radix. The count+collect fusion that would cut a pass is Opt-L FALSIFIED.
2. **Short-N phase-chain floor (op32 WALL).** GVR skeleton nsys floor ~9.7us at N=8192; "the
   single-pass histogram that beats it (sglang v2 ~7us) is a DIFFERENT skeleton, excluded by
   the campaign constraint." Beating sglang's ~4-6us at small N inside the GVR skeleton = walled.
3. **Hint-on-better-skeleton control (op31 HBE-C).** Putting the GVR hint on sglang's OWN
   cluster kernel to eliminate the full scan reaches only PARITY in-envelope (geomean 0.991,
   wins only N≥524288 outside envelope). So the hint alone does not extract 30% even on the
   faster skeleton — a strong relaxed-constraint control against the target.

=> The 30%-grand-average target is, on priors, **probably structurally infeasible** within the
GVR skeleton on BS=1. This is NOT yet the op34 verdict: per protocol the negative conclusion
requires a FRESH double-lock on THIS (v4cap) data — math floor + a relaxed-constraint control
run here. The campaign's job is to run that double-lock honestly AND harvest any conditional
win region (e.g. a high-hr large-N sub-band) before ruling.

## Regime strategy
- **Large N (≥65K), high-hr (esp. pro):** best odds — smallest deficit + strongest hint.
  Levers not-yet-falsified on real v4cap: intra-CTA warp pipelining (deepest untouched),
  multi-CTA-per-row for BS=1 (match sglang parallelism) + hint-pruned work.
- **Small N (≤32K):** treat as op32 wall; bound with Phase-5 UB/LB, do not burn iterations.
- Report the regime map + honest wall attribution; the pre-authorized negative conclusion (PLAN
  AUTONOMY.md) applies if the double-lock confirms.

## Deliverables
scaffold (this dir) · analysis/ (Phase-1, done) · a BS=1 real-data cold-L2 nsys A/B harness
(scripts/) · iteration log · bilingual HTML report (report/).
