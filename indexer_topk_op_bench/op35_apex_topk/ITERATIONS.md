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

## iter 12-13 — 2026-07-16 — v3 architecture + full-envelope position (b200-072)
v3 = smem-staged filter (per-warp regions in 96KB dyn smem, ONE end-of-CTA
flush: block prefix + 1 global atomicAdd reserve + coalesced copy) + split
kernels for BS>=32 (k_thr/k_filter/k_tail; filter back to v10 registers,
12.3/62.6/66.9us == v10) + coalesced tail gather + t_hi/c_hi CUT (unused) +
phase A one-pass 2048-bin window hist (kmin/kmax prefix skip; 2-level 32x64
warp find; reg-resident samples s/NT<=16) + adaptive tail smem cap.
Anchor spans: BS1 17.2 (0.63-0.89x F) · BS32 31 (0.76) · BS256 89 (1.16 WIN)
· BS1024 123-129 (0.72-0.75).
FALSIFIED along the way: match_any/MIO as thr bottleneck (reg-resident samples
moved nothing — cost is PASS/BARRIER serialization at ~0.5-1GHz effective);
find_bin2k 64-serial-reads-per-lane (thr 31->57us, fixed by 2-level).
DVFS hypothesis falsified directly (hot vs cold+idle50ms: 55 vs 61us).

iter13 FULL-ENVELOPE sweep (op26 rival grid, fp32, 347 cells, protocol ==
rival_harness measure_cell; scripts/iter13_{sweep,report}.py):
  run1 (pre-fix): gm 0.51 BUT catastrophic tail (0.01-0.12x): op26 synth data
    is SPATIALLY CLUSTERED -> float4-quad samples correlated (s_eff ~ s/4) ->
    genuinely wide bands (t_lo 2.11 vs kth 4.33, M=14.5K at K2048/1M; NOT
    ties) -> M>cap / miss -> 2.7ms full-row fallbacks.
  fixes: (a) graceful spill: filter region overflow -> warp-aggregated global
    append (no flag, no loss); (b) big-M tail path: tail_cap<M<=GCAP=32768 ->
    radix+emit direct from global cand (cost ~M not ~N); (c) full-row fallback
    only M<K or M>GCAP; (d) quad-correlation-corrected margins (sig x2, z=6)
    + per-K lambda {512:4,1024:8,2048:16} s policy.
  BUG (exactness, 1 cell): cand row stride left at PAIR_CAP after GCAP buffer
    introduced -> cross-row contamination (dup indices). sed fix -> 347/347
    exact incl. all 55 real cells.
  FINAL iter13 position: overall gm frontier/apex 0.468 (synth 0.477, real
    0.420). Regimes: BS256-1024/N131-262k 0.65-0.71 (best cells 0.86-0.94);
    N<=16k 0.30-0.42 EVERYWHERE (launch+A+tail overhead vs sglang_v2 ~4.5us
    near-floor); BS1 0.35-0.65.
Next levers (leverage order): (1) small-N full-scan threshold mode (s=N,
exact counts, no miss, M~K) for N<=16-32k — 68 cells at ~0.35; (2) tail
single-pass 2048-bin window select (replace byte rounds; tail radix still
3-9us for M~3K — pass/barrier bound); (3) thr 2-pass data-independent window
(drop minmax dependency); (4) BS1 cooperative single-wave (smem row cache) —
hard; op34 lesson caps expectations. HONEST: even all of 1-3 lands ~0.75-0.9
overall; 1.5x geomean bar needs a qualitatively bigger idea or envelope
re-scope (16-bit cells? frontier arm weaknesses?).

## iter 14 — 2026-07-16 — FALSIFIED x2 (small-N single-CTA; window-select tail)
(a) row-in-smem single-CTA-per-row kernel (N<=32768, 1 DRAM read, exact in-smem
select): gm 0.468 -> 0.392. Per-CTA serial instruction cost at real effective
clocks (~0.5-1GHz for micro kernels; DVFS-idle falsified hot-vs-cold 55/61us)
fully devours the traffic win: N32768 BS1 = 55us vs sampling path ~20us.
LEDGER: (single-CTA-per-row whole-row designs; any N; nsys) = FALSIFIED —
per-CTA pass serialization is the currency, parallelism-free designs lose.
(b) window_select (2048-bin) for the staged tail: 2 hist_adds/key + 2080-word
zero per pass loses to byte-skip 256-bin rounds at M~1-4K. Kept ONLY for the
big-M global path. Violated one-variable discipline (a+b landed together;
cost an extra sweep to untangle).

## iter 15 — 2026-07-16 — scalar strata + sig x1 (GO, 0.488)
Quad-correlation root-caused to float4-granular sampling; scalar strata
restore IID margins: sig x1 z=6, s = clamp(pow2(8N/K),1024,8192). No misses,
no catastrophic tail (worst 0.27 = small-N overhead, not fallback). gm 0.488.

## iter 16 — 2026-07-17 — 16-bit support DONE; 16-bit-as-lever FALSIFIED
bf16/fp16 templated pipeline (uint4 16B loads = 8 halves, half->float exact
convert, internals unchanged). EXACT: 1041/1041 all-dtype envelope cells.
Recon had suggested frontier scales only 0.94x at 16-bit (and sglang_v2
absent from the 16-bit frontier) => structural tailwind. MEASURED: apex
16/32 time ratio 0.93 vs frontier 0.94 — REGIME-matched (BW cells: we 0.78,
frontier 0.71; small-N: both ~1.0-1.2). Relative position UNCHANGED:
0.506/0.501 (bf16/fp16). LEDGER: (16-bit dtype as relative-position lever;
whole envelope; nsys) = FALSIFIED — the aggregate 0.94 was dilution, not
frontier weakness.

## iter 17 — 2026-07-17 — mixed dispatch; FINAL CAMPAIGN POSITION
All-fused vs split experiment: wash overall (0.501 vs 0.500) but
complementary (fused wins N<=65536 launch-bound; split wins BW-bound).
Mixed dispatch (split only BS>=32 && N>65536):
  **FINAL: fp32 0.507 · bf16 0.516 · fp16 0.512 (geomean frontier/apex,
  347 cells each, 1041/1041 exact incl. all real captures)**
Best regime BS128-1024 x N131-262k ~0.70 (cells to 0.94); worst N<=16k ~0.4.

## VERDICT — 2026-07-17 — +50% over composite frontier: STRUCTURALLY INFEASIBLE
for the sampling-filter-select family on this envelope. The rung-0 feasibility
case rested on floor = max(bytes/7TBps, 4us launch); that floor is not
achievable by ANY exact top-K kernel at these shapes:
- launch-bound cells (N<=16k): the true floor includes threshold acquisition
  + emission; sglang_v2/flashinfer already sit at 4.5-8us ~ 1.1-1.5x of the
  TRUE floor. Beating them 1.5x would require < 1 launch + < 1 row pass.
- BW-bound cells: frontier ~1.6-1.9x pure-read; our filter tax alone is
  1.14-1.31x, +thr +tail => ~2.0-2.4x read. Beating frontier 1.5x requires
  total <= 1.1-1.27x pure-read — BELOW the measured filter-only floor.
- Instruction/pass tax: micro-kernel phases cost ~3x boost-clock estimates
  (NCU: IPC 1.13, not DVFS); every added pass/barrier is 3x more expensive
  than paper math. This killed all "clever" multi-pass structures.
Remaining unexplored idea (bounded upside): persistent single-wave
cooperative kernel (cross-row pipelining of thr/filter/tail + zero launches).
Estimated +10-25% overall, NOT 3x. Not pursued; documented in RESUME.

## iter 18 — 2026-07-17 — REGIME CAMPAIGN (disposition 1) — KILLED per criterion
Objective re-scoped (user): beat frontier on BS>=128 x N>=131k via cross-phase
pipelining ("persistent" family). Kill line: regime cells must clear 1.0x.
Probes, in order:
  a. Python 3-stream chunked pipeline: FALSIFIED — host orchestration floor
     ~260us flat (stream switches + event churn at us-kernel scale).
  b. C++ 3-stream chunked pipeline (static stream/event pools, ~us host):
     FALSIFIED — regime gm 0.675 (seq) -> 0.481 (pipe). Chunking narrows each
     stage's grid (BS64: 64-CTA filter waves ~ 0.43 wave); machine-fill loss
     exceeds hidden thr/tail exposure. Full-width sequential stages already
     saturate within-stage; only the 2 stage boundaries are exposed.
  c. all-fused per-row pipelining: already measured iter17 = wash (occupancy).
  d. __noinline__ register isolation: FALSIFIED — fused stays 48 REG (+64
     STACK); ptxas allocates worst-case across ABI calls. ALSO discovered:
     even lean k_apex_filter is 39-40 REG -> NT1024 was ALWAYS 1 CTA/SM
     (v10 baseline too) — "2 CTA/SM" never existed; numbers are the baseline.
  e. Cost autopsy of regime cells (nsys per-kernel): BS512/N262144/K512 =
     thr 66 (scalar-sample scatter, 2M sectors) + filter 153 (1.6x read) +
     tail 26. ARITHMETIC CEILING: even with thr+tail FREE, filter-only vs
     frontier = 180/153 ~ 1.18x. The 1.5x-class win does not exist in this
     family on this regime; 1.0x regime geomean needs near-perfect overlap
     that mechanisms (a)-(d) show is not available.
VERDICT: kill criterion triggered (0/60 regime cells >= 1.0x across all
mechanisms). Disposition-1 campaign CLOSED. Remaining artifacts: C++
apex_pipe (opt-in, cfg["pipeline"]=True) kept for the record; PDL
grid-boundary overlap left unexplored (bounded upside ~10-15%, cannot clear
the 1.18x arithmetic ceiling gap to 1.5x).
