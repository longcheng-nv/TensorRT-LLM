# op35 APEX top-K — campaign plan (omni-kernel v2)

## Objective triple (USER-set 2026-07-16, may not be relaxed by agent)
```yaml
objective:
  incumbent: composite per-cell frontier of ALL 6 arms
             (gvr_base, gvr_pr@eae374554c, op26_r0auto, radix_cutedsl, sglang_v2, flashinfer_topk)
             from op26 report rival_long.csv (2772 cells, b200-044/081/094 rival sweep)
  rivals: each individual arm (report per-arm geomeans too)
  envelope: op26 REPORT.html grids — synth best/worst x K{512,1024,2048} x dtype{fp32,bf16,fp16}
            x N 4k-1M x BS 1-1024  +  real V4 flash/pro (9 ISL) + V3.2 (7 ISL) x dtype x BS
  verdict_axes: [worst(synth cold-hint), real, best(synth warm-hint)]
  ship_rule: "geomean >= ~1.5x vs composite frontier on the envelope; exactness
             (tie-aware value-multiset) green on all 3 tracks; NO per-case-tuned
             mega-dispatch (launch-shape policy à la pick_config allowed;
             algorithm must be ONE); regression-free vs frontier not required
             per-cell but report the loss tail honestly"
  hard_constraints: [exact top-K only, CUDA-graph compatible (no host sync in hot path),
                     data-distribution features via distribution-FREE math (order statistics)
                     or hint, not per-case constants]
```

## A-priori feasibility (notes/floor_map.txt, 2026-07-16)
- floor(cell) = max(BS*N*sizeof/7TBps, ~4us launch): frontier headroom geomean 3.17x
  (regime gm 1.70-6.36; tightest = BS1/N<=16k 1.70 @4us assumption).
- kernel at alpha x floor everywhere => vs frontier: alpha=1.2 -> 2.64x, 1.5 -> 2.11x, 2.0 -> 1.58x.
- => 50% target is a-priori FEASIBLE if we build a ~<=2x-floor kernel; BS1 small-N is
  the sensitive zone (launch-floor calibration = rung-0 #1).

## Algorithm hypothesis H0: "APEX-FR" — GPU Floyd-Rivest sampled-quantile filter
1 (+eps) full-N read total:
- SEED: per-row threshold ladder (2-3 rungs) from (a) hint quantiles (gathered prev-topK
  values, ~K elems) fused with (b) distribution-free sampled order statistics
  (contiguous-warp random sample, ~1-3% traffic) — Floyd-Rivest band, width from
  concentration bounds, NOT per-case tuning.
- FILTER PASS: stream N once; per element 1-2 compares; warp-aggregated append of
  admitted elements (expected ~c*K, c~2-4) to global scratch + per-rung counts.
  Cost scales with admitted count (revival of falsified Opt-L whose death was
  per-ELEMENT slot-reserve).
- TAIL: last-arriving CTA (atomic ticket) selects exact K among admitted candidates
  of the tightest rung with count>=K (~2K elems, smem radix/bitonic) and writes indices.
- MISS: all rungs underfill => 1 extra corrected pass (probability bounded by FR math;
  measure on real+adversarial).
- GRID: row-parallel at big BS, multi-CTA-per-row at small BS — same kernel body,
  shape policy only (pick_config-style, allowed).
vs rivals: sglang=2-pass histogram (halve traffic), radix=1-pass heavy-ALU digit
histogram + candidate rounds (we do ~2 compares/elem), GVR=~2.5 pass + cluster
barriers (we have none), flashinfer(read source).

## Probe ladder (Phase 3)
- rung0.1 CRUX: nsys span floor — empty kernel + pure 1-pass streaming reduce across
  (BS,N) grid shapes incl. atomic last-CTA finalize. Calibrates LAUNCH + achievable BW.
  KILL: if BS1 floor ~>= best-arm/1.2 => BS1 small-N declared wall, target re-scoped to
  report per-regime.
- rung0.2 CRUX: FR band math on real captures + synth (host/torch): sample size s,
  band z -> admitted count c*K + miss rate, incl. 16-bit tie plateaus + worst (hit .05).
  KILL: if c*K needed > ~8K or miss% > ~2% on real data => ladder redesign.
- rung0.3: read sglang_v2/radix/flashinfer sources -> pass counts + primitive inventory.
- rung2: filtered-pass microbench — compare+warp-aggregated-append at 3% admit vs pure
  read BW (target >= 0.85x pure-read BW).
- rung3: kernel (CUDA C++ .cu + torch cpp_extension, standalone like sglang_v2_op).

## Language decision (1.2, once): CUDA C++ (.cu via cpp_extension, tvm-ffi-free standalone)
— new algorithm, warp-aggregated appends + global atomics, fastest iteration; op-bench
already has the build infra (ops/sglang_v2 pattern). cuteDSL reserved as last-resort.

## Measurement
- L1 = cold-L2 CUDA-graph sweep (harness reuse: op26 rival_harness batch protocol).
- L2 = nsys x3-median, single-GPU paired back-to-back A/B vs frontier arms; anchor cell
  = synth K512 fp32 N131072 BS1 best (gvr_pr expected ~14.4us on 038-class node).
- 3-axis verdicts always [worst, real, best]; cold-L2 canonical.
- Exactness: tie-aware value-multiset vs torch.topk + real captured refs + adversarial
  (1-2ulp tie bands, 16-bit plateaus, degenerate hint).
