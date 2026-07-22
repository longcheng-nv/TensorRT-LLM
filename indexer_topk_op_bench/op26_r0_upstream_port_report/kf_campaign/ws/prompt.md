# DeepSeek-V4 Indexer Top-K Decode (BS=1, fp32, B200) — Optimize the Production GVR Kernel

## Problem

Sparse-attention indexer top-K selection at decode time. One row of real
captured indexer logits (`logits[1, npad]`, fp32, valid length `n_valid`,
tail padded so pad never enters the top-k). Return the `int32` indices of
the `k` largest values, any order; ties at the k-th value boundary may be
resolved either way (the correctness checker is index-SET based and
tie-robust). Exactness is non-negotiable: every index whose value is
strictly greater than the k-th value must appear, on every run.

`pre_idx[1, k]` is the PREVIOUS decode step's top-k (temporal warm hint).
Overlap with the true top-k ranges 0.02–1.0 across workloads (typically
>0.5). Exploiting it is REQUIRED (see skeleton below), but correctness and
the no-regression bar must hold even at 0.02 overlap. You may NOT branch
on any estimate of hint quality computed outside the kernel (hit-rate is
unknowable at inference); in-kernel admission escape / lagged feedback is
fine.

Workloads are REAL production captures from three models, n up to ~1.05M:
- V4-Flash: k=512,  n rungs 4K / 32K / 128K / 512K / 1M
- V4-Pro:   k=1024, n rungs 4K / 32K / 128K / 512K / 1M  (highest priority)
- V3.2:     k=2048, n rungs 4K / 32K / 128K / 256K

Two workloads per (model, rung): a low-hint-overlap layer and a
high-overlap layer. The logits distribution is NOT random — heavy-tailed
real indexer scores (near-exponential CCDF); algorithms that look good on
`randn` behave differently here. Priority for effort allocation:
V4-Pro > others; n ≥ 32K > small n.

## Baseline

The baseline you must beat is the CURRENT PRODUCTION kernel: the
guess-verify-refine (GVR) top-K from TensorRT-LLM PR#16457 (latest head,
including its K=2048 tail-ladder tuning). Its structure: seed a threshold
guess from `pre_idx`, verify/refine the threshold with a secant solve in
log space, then exactly collect the surviving candidates. It is written in
CuTe DSL (Python); a verbatim source digest — full config/dispatch/
orchestration layers plus the signature and docstring of every phase
primitive — is in the APPENDIX at the bottom of this brief. Read it first —
your job is to make THIS algorithm faster (re-expressed in CUDA C++), not
to replace it. You must find the profitable directions yourself by
profiling and analysis.

The per-workload baseline timings (your speedup denominator) are EXTERNAL
nsys cold-L2 pure-kernel-time medians of exactly that production kernel on
an idle B200 — they contain none of the ~15µs harness floor your own
platform measurements include. Consequence: at true kernel parity your
platform speedup will read ~0.5-0.9x, NOT 1.0x; the compression is
strongest on the smallest workloads. Do not be discouraged by sub-1.0
readings early on — track your RELATIVE progress across submissions, and
judge absolute wins by how much kernel time you save on the large-n cells
where the floor is a smaller fraction.

One measured fact about where its time goes (from the external report on
the full 865-cell grid): the FINAL-COLLECT block (threshold handoff +
refine + writeback, "P4") dominates — it is the largest phase on 827 of
865 cells, median ~37% of kernel time (range 23–58%). The mid scan/count
passes are second. How to attack that is yours to work out.

## Target

- **Required:** geomean speedup > 1.0× over the given baseline with NO
  workload slower (no-regression is a hard acceptance bar — a kernel that
  wins big on average but loses any cell will be rejected downstream).
  The external acceptance goal is +60% geomean on the full 865-cell grid;
  every incremental win counts toward it.
- Final acceptance re-measures externally with nsys cold-L2 on all 865
  real cells (the cells here are a stratified subset). Platform timings
  have a ~15µs floor that compresses your true speedup roughly 1.3–1.4× —
  do not tune to the harness floor; win in kernel time. Do not overfit to
  these exact n values: `n` is dynamic (up to ~1.05M), `k ∈ {512, 1024,
  2048}` at runtime, hint quality is dynamic.

## Required algorithmic skeleton — HARD compliance rule

Keep the GVR skeleton: (a) `pre_idx` as the threshold prior, (b) a
secant+log-transform style exact threshold solve (or an equivalent
threshold-refinement structure), (c) an exact refine of the surviving
candidates. Any per-stage restructuring that preserves exactness is
allowed. Mature primitives (histogram ladders, radix digit passes) may be
absorbed INTO stages.

**Non-negotiable:** a submission that abandons the `pre_idx` prior, or
replaces the threshold-prior structure wholesale with a prior-free
selection algorithm (plain radix-select, full sort, sampling-based
selection), is NON-COMPLIANT and will be rejected even if it is faster.
Likewise, do not build a per-case dispatcher across unrelated top-k
operators. The goal of this campaign is a better GVR, not a different
algorithm.

## Dead ends — measured net-negative on THIS workload/hardware; do not re-discover

1. BS=1 is latency-bound, not bandwidth-bound (24% occupancy, <1% DRAM at
   small n). Bandwidth-oriented rewrites miss the bottleneck.
2. `pre_idx` warm-hint grafted onto radix-select: no win (hint only helps
   threshold-style skeletons) — and prior-free pivots are banned anyway.
3. Private per-warp histograms to avoid smem atomics: loses (SM100
   pipelines same-address atomics fine).
4. Multi-CTA for SMALL n (< ~8K): launch/sync overhead dominates;
   single-CTA wins there.
5. More than 8–32 CTAs at large n: merge cost eats the scan win.
6. Per-element ballot/popc slot-reservation to fuse count+collect into
   one pass: coordination ≈ a full extra pass.
7. Staging the row into shared memory first: row re-reads are cheap L2
   hits.
8. Extra secant/interpolation refinement rounds: each is a
   barrier-separated pass; keep passes ≤2.
9. CUB DeviceRadixSort / full sort: ~10× too slow at these sizes (and
   banned as a wholesale replacement).
10. Fusing the final-collect histogram into the P3 scan loop: pollutes
    the scan inner loop, −15%.
11. Shrinking histogram bins below 512 for k=2048: exact-tail scratch
    overflows (silent UB); kNumBins=512 at k=2048 is already in the
    baseline.
12. Launch-config-only retuning: ceiling measured at ~1.025×.
13. CUDA graphs / replay amortization are banned by the compliance judge —
    win inside the kernel.

## Correctness traps

- The k-th-value tie boundary: the checker requires ALL indices with
  value strictly greater than the k-th value, plus any tie subset to fill
  the remainder. Arrival-order races on the boundary bin under concurrent
  compaction are the classic silent bug — never drop a strictly-greater
  element.
- Real data is UNDERSHOOT-biased for hint-seeded thresholds (the seeded
  count almost always comes in below k, not above): guards that only fire
  on overshoot are dead code here.
- On cluster launches, `cluster.arrive_relaxed()` has no release
  semantics: a DSMEM read of a just-written scalar can observe stale
  data. Use `cluster_arrive()` (release) or an acq_rel cluster fence on
  the write side. Symptom: wrong indices clustered by CTA slice.

## Requirements

- CUDA C++ (sm_100a Blackwell). fp32 in, int32 indices out.
- Exact per the tie-robust set semantics above — no approximation.
- Dynamic `n` (up to ~1.05M, padded width `npad = ceil(n/64)*64`), dynamic
  hint quality, `k ∈ {512, 1024, 2048}` at runtime.
- Deterministic output not required (any tie resolution accepted), but
  the index set must be exactly right on every run.
- One kernel launch preferred (or 2 with programmatic dependent launch);
  launch overhead is material at 3–29 µs.
