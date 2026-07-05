# op21 learnings

## Inherited physics & red lines (condensed; full list in PLAN.md)
- op14: ALL DSv4 decode rows are L2-resident on B200 ⇒ pass-count reduction
  saves zero DRAM; levers = SMs/row, #rounds, barrier chain, per-row fixed cost.
- count_ge_multi_bench: M=2 fused count ≈ free (×1.00-1.06); M=4 ×1.01-1.46
  (grows with N); M counters must be static registers (no difference-array);
  one barrier M-column reduce; don't cache M× per-thread counts for P3.
- op20 iter4: slot-collect fused into the count scan is validated exact + wins
  at large N (P3 → O(candidates)); overflow fallback mandatory.
- Exact fixed-256-bin rank-scatter is a validated exact band-refine primitive.
- Real preIdx hit-rate: Pro 0.69-0.77, Flash 0.36-0.46, synth 0.6 — gathered
  prev-K value at sorted rank i has current global rank ≈ i/h ⇒ order-stat
  placement must be robust to h, or use g_min (rank K) as the guaranteed
  c ≥ K lower anchor (every gathered value ≥ g_min, K distinct positions).

## Iter log-worthy findings
- (iter0.5) P1 order-stat seeding law: gathered prev-K value at sorted rank i
  has current global rank ≈ i/h. Placement fractions must span below the
  worst-case hit-rate (0.35 real). g_min (rank K) is a GUARANTEED c≥K lower
  anchor — kills the `all_lt` failure mode entirely (0/55 rows).
- (iter0.5) miss mode is exclusively `all_ge` (5.5%) — round-2 must search
  ABOVE thr(f_min), i.e. inside the gathered top quarter; interp on the two
  highest counts.
- (iter0.5) speculative collect column is a per-K constant, not per-N: pro
  f=0.75 fits kC 30/30; flash needs lower f (h and K both smaller).
