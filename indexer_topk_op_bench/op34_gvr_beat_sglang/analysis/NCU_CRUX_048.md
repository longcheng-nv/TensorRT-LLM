# op34 decisive NCU CRUX — WHY sglang wins (node 048, pro/256k N=65539, cold-L2)

ncu 2025.4, `--cache-control all` (flushes L2 between replays ⇒ cold, matches canonical).
Single launch each, after warmup. scripts/ncu_crux.py.

| metric | op26_r0auto | sglang_v2 (topk_small_batch_kernel<1>) |
|---|---|---|
| launch__grid_size | **1** (grid (1,1,1)) | **8** (grid (1,8,1)) |
| launch__block_size | 1024 | 1024 |
| gpu__time_duration | **43.26 µs** | **28.16 µs** |
| dram__bytes_read.sum | 361 KB | 358 KB |
| dram__throughput (% peak) | **0.11 %** | **0.17 %** |
| sm__throughput (% peak) | **0.23 %** | **0.83 %** |
| sm__warps_active (% peak) | 49.95 % | 49.90 % |
| gpc__cycles_elapsed.max | 49898 | 32461 |

## The three decisive facts
1. **Same cold-HBM traffic.** Both read ~360 KB ≈ 1× the N=65539 fp32 row (262 KB) + preIdx
   gather + out. Pass count does NOT differentiate them under cold-L2 — CONFIRMS the ANCHOR_048
   cold-L2 refinement. ⇒ pass-fusion (iter2 single-CTA) is a dead lever here.
2. **Neither is bandwidth- OR compute-bound.** DRAM < 0.2 % of peak, SM < 1 % of peak, for BOTH.
   At B200 ~8 TB/s a 262 KB read is ~33 ns of bandwidth; the kernels take 28–43 µs = ~1000× off.
   ⇒ **LATENCY-BOUND** (memory-latency chains of the strided scan + phase/barrier dependency
   chains), exactly as op32 found for BS=1 short rows. Throughput levers (vectorize, coalesce,
   tensor cores) are a priori void; the lever is **memory-level parallelism (MLP)** = # of
   outstanding loads in flight to hide latency.
3. **sglang's ONLY structural edge = 8× MLP via 8 CTAs.** grid 8 vs 1; time 28 vs 43 µs = 1.54×
   for 8× the SMs (sub-linear — latency hiding saturates + sglang carries fixed phase overhead).
   Occupancy is identical (~50 %), so it is NOT an occupancy-per-SM story; it is CTA COUNT.

## What this means for the campaign
- The single-SM occupancy wall (op#8 WALLS) is the operative wall, but its mechanism here is
  **latency, not bandwidth**. Two MLP levers remain live on real v4cap:
  (a) **multi-CTA-per-row** (match/exceed sglang's 8) — GVR at BS=1 wastes 147 SMs; nothing stops
      16/32/64 CTAs/row. If MLP keeps scaling past 8, GVR could BEAT sglang's fixed-8 on the scan.
  (b) **intra-CTA MLP** (deeper unroll / more vectors in flight / sw-pipeline) — cheaper, single-CTA.
- BUT the GVR skeleton is multi-phase (P1→P1b→R0→[R1]→P3→P4→fb), each phase a barrier =
  latency chain that does NOT shrink with more CTAs. sglang is lean (histogram→collect→radix).
  So even a perfect multi-CTA GVR scan carries phase-chain overhead sglang doesn't. Beating sglang
  by 30 % needs BOTH more MLP than 8-CTA AND phase overhead ≤ sglang — the open empirical question.

## Next probes (rung-2 microbench, decisive)
- **CRUX-A**: bare multi-CTA count scan, C ∈ {1,2,4,8,16,32,64} CTAs/row at N=65539 & 262144
  cold-L2 — does scan latency keep dropping past C=8? Sets the MLP ceiling + whether >8 beats sglang.
- **CRUX-B**: op26_r0 phase breakdown (clock64 stamps, L0 diagnosis-only) — scan vs P4 vs fb
  fraction ⇒ Amdahl ceiling of multi-CTA-ing only the scan phases.
