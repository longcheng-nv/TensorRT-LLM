# HBE-C rung-2 — DSMEM reduce microbench: **GO with C2 REVISED** (2026-07-13, node 072 GPU0)

Spec: DESIGN_HBEC_HINT_LADDER.md §6 rung 2. Bench:
`src/gvr29/hbec_rung2.cu` + `scripts/bench_hbec_rung2.py`
(BS=1 single 8-CTA cluster; per-round latency = slope of t(rounds) over
rounds {16,64,256}, launch overhead cancels; median of 200 reps; data =
`results/hbec_rung2/rung2.json`).

## Measured per-round net latency (minus 2-sync baseline 0.65us; 1 sync ≈ 0.33us)
| pattern | net us/row |
|---|---|
| stock dense 1024-bin all-reduce (TopKCluster Phase 1.5 port) | 0.76 |
| dense 4096-bin all-reduce (HBE 12-bit variant) | 2.33 |
| HBE-C M×8 scalar reduce (rung select) | 0.32 |
| REMOTE-atomic cand mini-hist 8K/16K/32K atomics → rank0 | 5.8 / 11.3 / 22.3 |
| **REVISED: per-CTA LOCAL mini-hist + 1024-bin all-reduce** (8K→32K cands) | **1.38-1.42** |

## Findings
1. **DESIGN §4 C2 as written ("distributed candidate mini-hist, DSMEM
   4096-bin, ~2-4K atomics") is FALSIFIED as a remote-atomic pattern**:
   cross-CTA smem atomics serialize — 5.8us @8K even, worse than the dense
   all-reduce it replaces. C2 must be: per-CTA LOCAL mini-hist build
   (cand-count-INSENSITIVE, 1.4us flat 8K→32K cands, local atomics spread
   over 1024 threads) + stock-style dense all-reduce.
2. **1024-bin mini-hist suffices** (stock cluster is 10-bit already; tie
   machinery bound kMaxNumTie=2048 unchanged) — 4096-bin would cost
   +1.6us for nothing.
3. The serial-chain swap itself (dense all-reduce → M×8 scalars) saves
   only **0.45us/row** — NOT the DESIGN §3 lever. Revised C2 total
   (0.32 scalar + 1.4 local-MH+all-reduce + 0.33 extra sync ≈ 2.0us) is
   ~1.3us MORE reduce latency than stock. The real BS=1 win must come
   from eliminating the full-N Phase-1 scan + the cheaper C1 body
   (≤3 cmps vs F2F+twiddle+smem-atomic), NOT from the reduce chain.
4. Engagement implication: BS=1 @131072 (10.9us cell): pass ≈1-2us,
   minus 1.3us reduce overhead → ≈wash, decide at pilot nsys. BS=1
   @512K/1M (22-33us): pass elimination >> overhead → engage. Mid-BS
   wave-bound prize cells: C2 latency amortizes across rows → unaffected.

## Next
rung 3: tier-5 kernel behind flag; C2 = scalars + LOCAL mini-hist +
1024-bin all-reduce; gate 3-track incl adversarial hints; pilot
(131072..1048576) × BS {1,16,64,256,512}; nsys same-batch 3 arms.
