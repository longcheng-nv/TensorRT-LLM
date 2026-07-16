# ITERATIONS — op35 APEX top-K

## iter 0 — 2026-07-16 — GO (rung-0.1 floor calibration, b200-038 GPU0/1)
Hypothesis (ledger: fresh): frontier sits 3.17x above info floor; a 1-pass
filter kernel can approach the floor incl. at BS=1 (where op32/op34 walls were
GVR-skeleton-specific, not physical).
Probe: rung0 microbench floor_probe.cu (k_empty / k_stream_reduce / k_filter_append,
warp-agg append + last-CTA ticket). Event L1 + nsys L2 (30 reps, cold-L2, GPU1).
Result (nsys median span us; frontier = per-cell 6-arm best, us_span):
  shape                R      F    frontier  F/R-tax  frontier/F
  BS1    N131072     3.62   4.51    11.6      1.25      2.57x
  BS1    N262144     3.65   5.22    15.7      1.43      3.01x
  BS1    N1048576    4.66   7.84    20.6      1.68      2.63x
  BS32   N262144    10.11  18.26    24.0      1.81      1.31x
  BS256  N262144    54.91 100.40   105.7      1.83      1.05x
  BS1024 N65536     54.08  74.67    93.0      1.38      1.25x
  empty-kernel span ~1.0us; event floor 8.3-8.5us (host launch, L1 axis only).
Diagnosis: BS=1 wall falsified for lean 1-pass design (GO). Filter inner loop is
the bottleneck at throughput shapes: dynamic-index register buffer spills to
local + warp-prefix coordination runs ~every group. Pure-read ceiling measured
4.9-6.3 TB/s (big BS), 3.3 TB/s @BS32 (wave ineff., fixable), ~0.9 TB/s @BS1 1M.
Ledger write-back: WALLS candidate removed (BS1 launch floor NOT a wall down to
~4.5us span); Opt-L revival PARTIALLY validated (append at 3% admit is cheap at
BS1, coordination too heavy at big BS -> v2 = SMEM-atomic staging).
Next: iter 1 filter v2 (SMEM staging, target F/R <= 1.10) + rung0.2 FR band math
on real captures + rung0.3 rival source study.

## iter 1 — 2026-07-16 — PIVOT (filter v2) + GO (rung-0.2 band math)
Hypothesis: SMEM-atomic staging removes big-BS coordination tax.
Result: v2 exact (3/3 screens); BS1 improved (tax 1.43->1.24, frontier/F 2.75-3.63x)
but BS256 N262144 COLLAPSED 226us (4.1x tax) — SCAP=2048 overflow -> per-element
global atomics (probe admit 3% x 262144 = 7.9K >> SCAP). BS1024 slight regress.
Diagnosis: overflow path is the killer, not SMEM atomics; also probe admit rate
was unrealistic (production admit ~= c*K/N ~= 0.3-1.5%, not 3%).
Ledger: (SMEM staging w/o bulk flush, admits>SCAP; nsys) = FALSIFIED-domain;
fix = periodic bulk flush (no overflow path at any admit count).
rung-0.2 band math on ALL real captures (25 shapes x 3 layers x fp32/bf16 x 8 seeds):
  - contiguous 32-blk sampling: miss ~10% (real logits spatially clustered — matches
    posz finding). blk4: miss 1.5-2%.
  - IID: miss 2-4/1000+, admit/K med 1.27-1.44.
  - STRATIFIED-JITTERED (1 elem per N/s stripe): **0 miss / 3312 trials**, admit/K
    med 1.29-1.39, p95 1.81-2.5, max 3.19; bf16 == fp32 (no tie inflation).
  => sampling design locked: stratified-jittered, s~1024-4096 (scale w/ q), z=3,
  in-kernel fallback pass for residual miss risk (correctness anyway).
Architecture locked (H0 refined): big-BS = self-contained row-per-CTA (sample own
row -> smem 2-level key-histogram nth-element -> filter own row -> in-CTA tail);
small-BS = multi-CTA/row single-wave + arrival-counter spin sync; 2 grid modes,
ONE algorithm (pick_config-style, allowed).
Next: iter 2 filter v3 (bulk flush) — target F/R tax <= 1.10 at BS256/BS1024.

## iter 2-9 — 2026-07-16 — filter-pass optimization ladder (nsys, GPU1, cold-L2)
Goal: 1-pass filter at ~read speed. Ladder (tax = filter/pure-read span):
  v3 barrier-storm flush: FALSIFIED (flush-every-iter at TPB512; BS256 3.44x tax)
  v4 predicated-skip + warp-prefix + GLOBAL atomic: 2.2-2.4x big-BS
  v5 = v4 + register double-buffer prefetch: BS256 1.69 (load pipelining real)
  NCU@BS32: V5 7x instructions, issue 19% -> GLOBAL-ATOMIC RETURN LATENCY in loop
  v6 smem-slot alloc + peek flush: BS32 1.49 (theory confirmed)
  v7 = v6 minus per-iter __syncthreads_or (band math bounds admits<=SCAP=4096;
     overflow -> flag+retry contract): 1.33-1.47 big-BS  <-- smem-atomic best
  v8 per-warp smem counters: FALSIFIED (no gain; contention was NOT residual)
  v9 zero-atomic ballot-rank + per-warp global segments: **BS1 tax 1.01-1.10**
     (BS1 frontier/filter 3.07-4.18x) but big-BS ~1.4-1.5 unchanged
  NCU@BS256: V9 12x inst (rare path fires 63% of warp-groups, 16-20 inst) AND
     warps_active 42% (256 CTA/148 SM wave quantization) with issue 44.6% -> BOTH
     instruction count and occupancy bind.
  v10<NT> group-prefix unordered append + TPB template:
     BS32@1024 12.06us tax 1.17 (frontier 1.99x) · BS256@1024 62.56 tax 1.14
     (1.69x) · BS1024 71.07 tax 1.31 (1.31x, wave-quantized; persistent multi-row
     per CTA = next lever) · BS1@512 3.74-4.85 tax ~1.0-1.09 (3.1-4.25x frontier)
Exactness: v9/v10 admit-set/index-set/count screens OK at both NT (segmented layout).
Ledger write-backs: (global-atomic slot alloc in stream loop; big-BS; NCU) =
FALSIFIED — latency chains; (per-warp smem counter split; BS>=32) = FALSIFIED —
no effect; (TPB 1024 at BS>=32 filter) = SHIPPED-lever; unordered-within-group
append is FREE (tail re-ranks).
Next: full APEX kernel v0 = stratified-sample+threshold (in-kernel redundant per
CTA at small BS / per-row at big BS) + v10 filter + last-CTA tail select + miss/
overflow retry; then 3-track exactness gate; then frontier A/B on the op26 grids.

## iter 10-11 — 2026-07-16 — APEX kernel v0/v1/v2 E2E (b200-072; read anchors == 038)
v0 fused kernel (A sample+16bit-hist band -> B v10 filter -> C last-CTA tail,
in-CTA full-row radix fallback, no 2nd launch, graph-compatible). Exactness:
32/32 synth+oddN+bf16-plateau+const-row+self-clean GREEN from first run, and
held through v1/v2. nsys iter10: E2E tax 3-15x over read — three root causes
attributed via mode={1,2,3} phase probes + in-kernel globaltimer + NCU:
  1. 16-bit-truncated t_lo inflates M on dense data (uniform: M=8152@1M vs
     band ~1.7K) AND z=3 misses at BS1024 (P(miss/row)~0.003, Poisson tail;
     badflags=2 -> 250us full-row fallback). FIX (v1): 4-round exact 32-bit
     sample threshold (=rung0.2 band math exactly) + z=6 + s scaled {2k,4k,8k}
     + float4-granular strata (4 samples/sector, 4x traffic cut; quad
     correlation absorbed by z=6 + exact fallback). badflags -> 0 everywhere.
  2. tail emit was 14us @BS1 (global re-scan of 2368 sparse segments +
     per-element same-address smem atomic). FIX (v2): cand as int2{bits,idx}
     pairs (1x8B ldcg), 64KB dynamic smem pair staging, warp-aggregated
     ballot emission from smem. emit -> 0.8us. gather prefix-scan+binary-search
     -> 6.2us (still slow: sparse segment scatter is structural).
  3. NCU @BS1024 mode1: 48 regs/thread -> Block Limit Registers = 1 CTA/SM at
     NT=1024 (occupancy 50%) — the fused fat kernel halves residency vs
     standalone v10 (<=32 regs). __launch_bounds__(NT,2048/NT) FALSIFIED:
     forces hot-loop spills, BS32 A+B 58->84us. Register pressure must be
     solved by SPLITTING, not clamping.
Current spans (event probe, cold-L2): BS1 full ~30-31us event (~22 span) vs
frontier 11.6 — NOT competitive yet. Decision for iter12 (architecture):
  a. filter admits -> per-warp regions in 96KB dynamic smem, ONE end-of-CTA
     flush (block prefix + 1 global atomicAdd reserve + coalesced copy).
     Kills nseg/segcap/warr/scan/binary-search; tail gather becomes a
     coalesced M x 8B read. Overflow (region or global cap) -> flag+fallback
     unchanged (constant-row degenerate covered).
  b. BS>=32 (cpr==1): split into 3 lean kernels (thr/filter/tail) to restore
     v10 register count & 2 CTA/SM; BS<=16 keep single fused launch (<=148
     CTAs, occupancy irrelevant, launch latency dominant).
