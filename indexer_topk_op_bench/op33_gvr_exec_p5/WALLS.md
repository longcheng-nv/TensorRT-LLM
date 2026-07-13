# op33 — structural walls (config-insensitive, mechanism understood)

## W1 — BS=1 execution-throughput levers VOID (all N, all K)
Domain: BS=1, N∈[16K,262K], K∈{512,1024,2048}, fp32/bf16, B200.
Evidence: NCU iter0 — DRAM 0.06-0.5% (GPU-wide idle), per-active issue ~15%,
occupancy 25%(1cta)/50%(cluster) grid-limited. Root class: structural-wall.
Consequence: 256-bit vectorization (DRAM idle) and occupancy lift (grid≪SM) both
a-priori void. The residual gap to sglang_v2/flashinfer is the 5-phase-secant vs
2-phase-histogram SKELETON (user-forbidden to change), not a tunable. Extends
op15 (smem-resident) + op32 (short-row register-resident) to all-N at BS=1.
Revival condition: NONE within the GVR threshold skeleton.

## W2 — sub-wave grid up through BS≈148 ⇒ per-SM occupancy lift void
Domain: BS ≤ ~128, any N/K. Evidence: `waves_per_multiprocessor` = 0.43 (BS16,
grid64) .. 0.86 (BS128, grid128) < 1.0. Root class: structural-wall.
Consequence: raising registers→occupancy adds no blocks when each SM holds <1
block. Occupancy-class P5 levers only bite at grid ≥ 148 (BS≥256 for one row/CTA,
or the cluster fan-out). Revival condition: BS ≥ ~256 large-N (waves ≥ ~1.7),
where occ is genuinely reg-limited at 50% — this is the iter1 target.
