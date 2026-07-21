# DeepSeek-V4 Indexer Top-K Decode (BS=1, fp32, B200) — Beat the Production GVR Kernel

## Problem

Sparse-attention indexer top-K selection at decode time. One row of real
captured indexer logits (`logits[1, npad]`, fp32, valid length `n_valid`,
tail padded with `FLT_MIN`-lowest so pad never enters the top-k). Return the
`int32` indices of the `k` largest values, any order; ties at the k-th value
boundary may be resolved either way (the correctness checker is index-SET
based and tie-robust).

`pre_idx[1, k]` is the PREVIOUS decode step's top-k (temporal warm hint).
Its overlap with the true top-k ranges 0.02–1.0 across workloads (typically
>0.5). You may exploit it (e.g. threshold seeding) but correctness and the
no-regression bar must hold even at 0.02 overlap.

Workloads are REAL production captures from three models:
- V4-Flash: k=512,  n ∈ {1027, 8195, 32771, 131075, 262127}
- V4-Pro:   k=1024, n ∈ {1027, 8195, 32771, 131075, 262127}
- V3.2:     k=2048, n ∈ {4111, 32783, 131087, 163775}

Two workloads per (model, n): a low-hint-overlap layer and a high-overlap
layer. The logits distribution is NOT random — it is a heavy-tailed real
indexer score distribution (near-exponential CCDF); algorithms that look
good on `randn` can behave very differently here.

## Baseline

The baseline timings are the production TensorRT-LLM GVR (guess-verify-refine)
kernel from PR #16457, measured with nsys (pure kernel time, cold-L2) on the
same B200 SKU. It runs 5–29 µs per workload. It is a single-CTA-per-row
(cluster multi-CTA at large n) histogram-ladder + secant-refine design.

## Target

- **Required:** geomean speedup ≥ 1.30× over the given baselines, and NO
  workload slower than ~1.05× (no-regression is a hard acceptance bar —
  a kernel that wins big on average but loses any cell will be rejected
  downstream).
- The final acceptance re-measures your kernel externally with nsys cold-L2
  on all 865 real cells (the 28 here are a stratified subset), so do not
  overfit to warm-cache effects or to these exact n values — `n` is dynamic
  (any value up to ~262k), `k ∈ {512, 1024, 2048}`.

## Hard-won structural knowledge (from months of manual optimization — do not re-discover these dead ends)

1. **BS=1 is latency-bound, not bandwidth-bound.** A single CTA on one of
   148 SMs leaves ~90% of cycles idle (measured 24% occupancy, <1% DRAM/SM
   util at small n). The single biggest structural lever is
   **multi-CTA-per-row cooperation** (e.g. 8 CTAs scanning disjoint chunks
   with a cheap global-atomic merge). The fastest known kernel on this
   workload (an sglang-style 2-kernel streaming top-k, ~1.4× over baseline)
   wins exactly this way.
2. **Histogram/radix select over fp32-as-ordered-uint is the strong family.**
   One global pass building a 256-bin (or hierarchical 256×256) histogram of
   the monotone key `u = f ^ ((f >> 31) | 0x80000000)`, prefix-scan to find
   the k-th bin, then a collect pass (or fused collect of a slightly
   over-admitted candidate set) with a tiny tail-select on the boundary bin.
3. Dead ends measured net-negative on THIS workload/hardware (B200):
   - per-element ballot/popc slot-reservation to fuse count+collect into one
     pass (coordination cost ≈ a full extra pass);
   - staging the row into shared memory first (row re-reads are already
     cheap L2 hits);
   - multi-round secant/interpolation refinement of a scalar threshold —
     converges but each round is a barrier-separated pass; keep passes ≤2.
   - CUB DeviceRadixSort / full sort: ~10× too slow at these sizes.
4. **Launch overhead matters at 5–29 µs.** Prefer ONE kernel launch
   (or 2 with programmatic dependent launch). CUDA graphs are banned by the
   compliance judge — win inside the kernel, not by replay amortization.
5. Beware the k-th-value tie boundary: the checker requires all indices with
   value strictly greater than the k-th value, and allows any tie subset to
   fill the remainder. Do not drop a strictly-greater element under
   concurrent compaction (arrival-order races on the boundary bin are the
   classic silent bug here).

## Requirements

- CUDA C++ (sm_100a Blackwell). fp32 in, int32 indices out.
- Exact per the tie-robust set semantics above — no approximation.
- Dynamic `n` (up to 262144, padded width `npad = ceil(n/64)*64`), dynamic
  hint quality. `k` is one of {512, 1024, 2048} at runtime.
- Deterministic output not required (any tie resolution accepted), but the
  index set must be exactly right on every run.
