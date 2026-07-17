# op35 APEX top-K — RESUME (updated 2026-07-17, post iter-17 VERDICT, b200-072)

## 1-minute context
Campaign: beat 6-arm composite frontier (rival_long.csv) ~1.5x geomean with ONE
new algorithm. **VERDICT (iter17): +50% is STRUCTURALLY INFEASIBLE for the
sampling-filter-select family on this envelope** — see ITERATIONS VERDICT +
WALLS.md (corrected floor analysis: launch-bound cells' true floor includes
threshold acquisition + emission; BW-bound cells need total <= 1.27x pure-read,
below our measured filter-only tax 1.14-1.31; pass/barrier cost ~3x boost math).

FINAL CAMPAIGN POSITION (@9bfda350b3, mixed dispatch):
  geomean frontier/apex: **fp32 0.507 · bf16 0.516 · fp16 0.512**
  (347 cells/dtype; 1041/1041 EXACT incl. all real captures)
  best regime BS128-1024 x N131-262k ~0.70 (cells to 0.94); worst N<=16k ~0.4.

## Deliverable state
- src/apex_topk.cu + src/apex_op.py: APEX-FR v3, fp32/bf16/fp16, exact,
  CUDA-graph compatible, self-cleaning. Mixed dispatch: fused<512> single
  launch (BS<=16 OR N<=65536), split thr/filter/tail (BS>=32 && N>65536).
- Fallback ladder: staged (<=tail_cap) -> big-M window_select from global
  (<=GCAP 32768) -> full-row byte-radix (M<K or >GCAP, near-never).
- scripts/: iter10_screen (exactness), iter10_nsys+parse (anchors),
  iter10_phase_probe (mode1/2/3 + globaltimer dbg), iter13_sweep+report
  (full envelope, --dtype fp32|bf16|fp16), iter11_ncu_target.
- Ledgers: ITERATIONS.md (iter0-17 + VERDICT), WALLS.md, FALSIFIED.md.

## Disposition 1 (regime campaign) EXECUTED and KILLED (iter18, 2026-07-17)
User chose the persistent/pipelining attack on BS>=128 x N131-262k. Kill line
(regime cells >= 1.0x) triggered: 0/60 cells across ALL mechanisms —
python pipe (~260us host floor), C++ chunked 3-stream pipe (0.675 -> 0.481,
narrow-grid machine-fill loss > hidden exposure), all-fused (wash, iter17),
__noinline__ register isolation (48 REG unchanged). Cost autopsy: regime
arithmetic ceiling ~1.18x (filter alone = 1.6x read vs frontier 1.9x read)
— the 1.5x-class win does not exist in this family on this regime. Discovered
en route: NT1024 was ALWAYS 1 CTA/SM (filter 39-40 REG; v10 baseline too).
apex_pipe kept as opt-in (cfg["pipeline"]=True). PDL grid-boundary overlap
left unexplored (bounded ~10-15%, cannot bridge to the bar).

## Remaining dispositions (user decision)
2. Harvest components: exact-sample-threshold band (rung0.2 math, z=6 scalar
   strata), smem-staged filter flush, tie-aware ballot emission — reusable in
   GVR-family kernels (op26_r0auto post-PR#16457).
3. Close campaign; negative-result REPORT (like op34) + COST accounting per
   the original user objective.

## Gotchas (carried; full list in prior version @31b19b9ad1 + FALSIFIED.md)
- match_any/ballot call sites must be warp-uniform, single-call-site predicated.
- workspace cand row stride == GCAP (NOT PAIR_CAP).
- probes with mode<3 leave counts dirty (tail does the self-clean).
- miss-rate validation MUST use op26 synth + real captures, never torch.rand.
- nsys span axis only; event probes carry ~8.4us launch floor; effective SM
  clock in micro kernels ~0.5-1GHz — budget passes at 3x paper cost.
- sweep exactness refs must be PER-ROW when rows differ (iter16 test-bug).
