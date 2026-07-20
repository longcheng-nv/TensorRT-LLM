# GVR prod top-K kernel (PR#16457 head) — in-kernel per-phase breakdown

**Method.** `gvrpkgtimed/` = spliced copy of `gvrpkgprod2/` (PR head, md5 `3396037c`, untouched) with `# [ptime]`-marked `cute.arch.clock64()` stamps in `_run_phases`, written by the leader CTA (cta_in_cluster==0) thread 0 to a `phase_ts` int64[num_rows, 8] GMEM tensor threaded through `__call__ -> gvr_topk_kernel -> run_one_row -> _run_phases`. No barriers added. BS=1 fp32 real captured cells, 10 warmup + 30 cold-L2 launches (512MB evict between launches), per-phase MEDIAN cycles.

**Absolute us.** `us_est = phase fraction x nsys PROD kernel-duration median` (30 cold-L2 launches, NVTX-segmented single nsys pass). CUDA events on this node (umbriel-b200-081) quantize to 2.048us ticks and include ~4-5us launch overhead at BS=1, so nsys is the wall-time anchor; event walls are shown for reference only. Implied SM clock = window_cycles / nsys timed-wall (consistency check, ~1.6-1.7 GHz).

**Timestamp map.** t0 entry | t1 P1 preidx gather/stats | t2 smem stage (==t1, cache disabled in prod config) | t3 P1b h-space rungs | t4 threshold final (R0 M-ary count + admission + fb_fix refine; secant on the no-R0 path) | t5 P3 collect (leader's own; cs>1 handoff #1 included in the P3 bucket) | t6 P4 select incl. cluster DSMEM gather, rank-scatter, p4_exact_tail/p4tt | t7 end (final cluster barrier). At cs=8 the t5->t6 bucket also absorbs the leader's wait on the slowest peer's collect (handoff #2).

| cell | K | N | cs | T | hit | P1 gather/stats | smem-stage | P1b rungs | P2 count+adm(+refine) | P3 collect | P4 select(+tail) | epilogue | total us (nsys prod) | timed vs prod (nsys) | exact | mono |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flash/32k/L22 | 512 | 8195 | 1 | 512 | 0.69 | 1.50us (16%) | 0.05us (1%) | 0.51us (6%) | 1.94us (21%) | 1.48us (16%) | 3.61us (40%) | 0.01us (0%) | 9.10 | +1.8% | Y | Y |
| flash/128k/L22 | 512 | 32771 | 8 | 512 | 0.70 | 1.47us (11%) | 0.07us (0%) | 0.41us (3%) | 2.41us (18%) | 1.08us (8%) | 7.63us (58%) | 0.08us (1%) | 13.15 | +2.7% | Y | Y |
| pro/128k/L30 | 1024 | 32771 | 8 | 512 | 0.33 | 1.69us (12%) | 0.06us (0%) | 0.61us (4%) | 2.47us (17%) | 1.14us (8%) | 8.37us (58%) | 0.09us (1%) | 14.46 | +0.7% | Y | Y |
| pro/512k/L30 | 1024 | 131075 | 8 | 1024 | 0.23 | 1.78us (10%) | 0.11us (1%) | 0.57us (3%) | 3.40us (19%) | 2.21us (12%) | 9.66us (54%) | 0.09us (0%) | 17.79 | +6.7% | Y | Y |
| v32/128k/L34 | 2048 | 131087 | 8 | 512 | 0.62 | 2.09us (12%) | 0.05us (0%) | 0.87us (5%) | 3.45us (19%) | 1.96us (11%) | 9.60us (53%) | 0.09us (0%) | 18.14 | +2.1% | Y | Y |
| flash/1024k/L22 | 512 | 262127 | 8 | 1024 | 0.42 | 1.58us (9%) | 0.09us (0%) | 0.46us (2%) | 3.79us (21%) | 2.96us (16%) | 9.38us (51%) | 0.13us (1%) | 18.38 | +3.5% | Y | Y |

## Validation

- (a) Output exactness: timed AND untimed index value-sets exact vs torch.topk on all 6 cells (unique count == K, gathered-value sets bitwise equal).
- (b) Instrumentation overhead (nsys kernel medians, timed vs prod): flash/32k/L22 +1.8%, flash/128k/L22 +2.7%, pro/128k/L30 +0.7%, pro/512k/L30 +6.7%, v32/128k/L34 +2.1%, flash/1024k/L22 +3.5% — all within the ~7% gate (worst pro/512k +6.7%).
- (c) Monotonic t0<=t1<=...<=t7 on every one of the 30 launches per cell.
- CUDA-event walls (quantized, launch-inclusive; reference only): flash/32k/L22 14.3/14.3us, flash/128k/L22 18.4/18.4us, pro/128k/L30 18.4/18.4us, pro/512k/L30 22.5/22.5us, v32/128k/L34 22.5/22.5us, flash/1024k/L22 22.5/24.6us.

## Per-cell findings

### flash/32k/L22 (K=512, N=8195, cr=4, hit=0.686)
- cfg `{'cluster_size': 1, 'num_threads': 512, 'use_256bit_load': True, 'min_blocks_per_mp': 1, 'enable_warp_parallel_reduce': False}`; nsys prod 9.10us / timed 9.26us (+1.8%); window 14883 cyc; implied SM clock ~1.61 GHz.
- Single-CTA cell near the BS=1 latency floor (~9us). P4 select is already the largest phase (40%, 3.6us) ahead of P2 admission (21%) and the P1 gather (17%). R0 admits on the rung ladder (high hit 0.69) so P2 stays cheap; the smem stage and epilogue are negligible.

### flash/128k/L22 (K=512, N=32771, cr=4, hit=0.701)
- cfg `{'cluster_size': 8, 'num_threads': 512, 'use_256bit_load': True, 'min_blocks_per_mp': 1, 'enable_warp_parallel_reduce': False}`; nsys prod 13.15us / timed 13.50us (+2.7%); window 24160 cyc; implied SM clock ~1.79 GHz.
- cs=8 parallelizes the P2/P3 row scans (P3 down to 8%) but P4 runs leader-only, so it balloons to 58% (7.6us) — the t5-t6 window also absorbs the cluster handoff wait for the slowest peer's collect. P2 count+admission is the only other material phase (18%).

### pro/128k/L30 (K=1024, N=32771, cr=4, hit=0.326)
- cfg `{'cluster_size': 8, 'num_threads': 512, 'use_256bit_load': True, 'min_blocks_per_mp': 1, 'enable_warp_parallel_reduce': False}`; nsys prod 14.46us / timed 14.56us (+0.7%); window 25111 cyc; implied SM clock ~1.72 GHz.
- Same shape as flash/128k but K=1024 and low hit (0.33): P4 select stays 58% (8.4us) and P2 17%. The rung ladder still admits (no visible refine tax; t3-t4 matches flash/128k within 3%), so low hit-rate cost shows up mostly as a slightly longer P1b/P1, not extra count passes.

### pro/512k/L30 (K=1024, N=131075, cr=4, hit=0.230)
- cfg `{'cluster_size': 8, 'num_threads': 1024, 'use_256bit_load': True, 'min_blocks_per_mp': 1, 'enable_warp_parallel_reduce': True}`; nsys prod 17.79us / timed 18.99us (+6.7%); window 32365 cyc; implied SM clock ~1.70 GHz.
- The p4_exact_tail + p4tt firing cell: P4 (incl. tail fast path) is 54% (9.7us) and P2 grows to 19% (4.3us) at N=131k/CTA-slice 16k. This is also the worst instrumentation overhead cell (+6.7% nsys) — the extra stamps sit on the leader's critical path around the tail select.

### v32/128k/L34 (K=2048, N=131087, cr=1, hit=0.620)
- cfg `{'cluster_size': 8, 'num_threads': 512, 'use_256bit_load': True, 'min_blocks_per_mp': 1, 'enable_warp_parallel_reduce': False}`; nsys prod 18.14us / timed 18.53us (+2.1%); window 32615 cyc; implied SM clock ~1.76 GHz.
- K=2048 cr=1 with the kNumBins=512 diet: P4 still 53% (9.6us) — the 4x-smaller histogram does not change the leader-only structural picture. P1/P1b are the largest among all cells in cycles (K=2048 gather + rung build), but remain <17% combined.

### flash/1024k/L22 (K=512, N=262127, cr=4, hit=0.422)
- cfg `{'cluster_size': 8, 'num_threads': 1024, 'use_256bit_load': True, 'min_blocks_per_mp': 1, 'enable_warp_parallel_reduce': True}`; nsys prod 18.38us / timed 19.02us (+3.5%); window 35734 cyc; implied SM clock ~1.88 GHz.
- Largest N (262k): P2 (21%, 3.8us) and P3 (16%, 3.0us) grow with the per-CTA slice (33k elems), yet P4 remains dominant at 51% (9.4us). Epilogue/final cluster barrier stays <1% everywhere — cluster teardown is not a cost.

## Cross-cell summary

P4 select(+tail) is the dominant phase everywhere: 40% at the single-CTA cell and 51-58% at every cs=8 cell (7.6-9.7us absolute), because Phase 4 runs leader-only while cs=8 parallelizes only P2/P3 — and the bucket additionally absorbs peer-collect wait. P2 count+admission is the #2 phase (17-21%) and scales mildly with per-CTA slice; P1 gather/stats is 9-17%; P1b rungs 3-6%; smem-stage and epilogue are noise (<1%). This silicon-confirms the op35 finding that the P4 block is the battleground for further BS=1 optimization.
