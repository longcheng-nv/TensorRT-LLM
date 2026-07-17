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

## If reopened, the ONLY remaining structural idea
Persistent single-wave cooperative kernel: 148 resident CTAs stream a row
work-queue; thr/filter/tail of DIFFERENT rows pipeline across the wave; zero
launch overhead; grid-sync via atomics (single wave => safe). Bounded upside
estimate +10-25% overall (NOT 3x) — it fixes launch + wave exposure, not the
filter tax or the small-N true floor. Kill fast if BS-large N131-262k cells
do not clear 1.0x.

## Alternative dispositions (user decision)
1. Re-scope objective to "beat frontier on the BS>=128 x N>=131k regime"
   (currently 0.70, cells to 0.94; persistent kernel might clear 1.0-1.2x).
2. Harvest components: the exact-sample-threshold band (rung0.2 math, z=6
   scalar strata), the smem-staged filter flush, and the tie-aware ballot
   emission are reusable in GVR-family kernels (op26_r0auto post-PR#16457).
3. Close campaign; keep as negative result + REPORT (like op34).

## Gotchas (carried; full list in prior version @31b19b9ad1 + FALSIFIED.md)
- match_any/ballot call sites must be warp-uniform, single-call-site predicated.
- workspace cand row stride == GCAP (NOT PAIR_CAP).
- probes with mode<3 leave counts dirty (tail does the self-clean).
- miss-rate validation MUST use op26 synth + real captures, never torch.rand.
- nsys span axis only; event probes carry ~8.4us launch floor; effective SM
  clock in micro kernels ~0.5-1GHz — budget passes at 3x paper cost.
- sweep exactness refs must be PER-ROW when rows differ (iter16 test-bug).
