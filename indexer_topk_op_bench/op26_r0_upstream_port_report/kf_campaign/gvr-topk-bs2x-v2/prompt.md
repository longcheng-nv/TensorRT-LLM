# DeepSeek Indexer Top-K Decode, BATCHED (BS 1-1024, fp32, B200) — Scale the GVR Champion Across Batch

## Problem

Batched sparse-attention indexer top-K at decode time. `logits[b, npad]` fp32
(REAL captured indexer rows; every batch row is a materialized copy of the
same captured row — distinct memory, so no L2-aliasing shortcuts), valid
length `n_valid`, tail padded with -FLT_MAX. `pre_idx[b, k]` int32 is the
previous step's top-k per row (again identical copies). Return `indices[b, k]`
int32: per-row exact top-k index set (tie-robust checker per row). Entry:
  run(logits, pre_idx, n_valid, indices)   # DPS, torch binding

Axes: b in {1..1024} (platform workloads sample b in {1,4,32,128,256,1024};
asset-size limits keep the largest-b x largest-n corner OFF the platform —
external acceptance still measures the FULL 75-cell x BS{2..1024} grid, so
optimize the whole envelope, especially b>=256 at n>=128k), k in {512,1024,2048},
n up to ~1.05M (`npad = ceil(n/64)*64`). Real heavy-tailed logits; hint
overlap 0.27-1.0. Do not overfit exact axis values.

## Baseline & starting point

Per-workload baseline timings are EXTERNAL nsys cold-L2 kernel times of the
PRODUCTION kernel (TensorRT-LLM PR#16457 GVR, CuTe DSL) run NATIVELY BATCHED
([b, npad] in one launch) on an idle B200. Beat THAT. The external acceptance
bars on the full 750-case real grid (75 cells x 10 BS points):
  1) average (geomean) speedup vs the production batched kernel >= 2.0x;
  2) EVERY case >= 1.0x (no regression anywhere), and the campaign objective
     beyond the bars is to MAXIMIZE THE MINIMUM per-case speedup;
  3) BS=1 must retain the existing BS=1 champion's performance (the champion
     source digest below IS the starting point — at BS=1 it already runs
     1.63x geomean over production on 865 cells; do not give that back).

Platform timings carry a fixed per-eval overhead; at BS>=32 it is negligible.
Track relative progress at small BS.

## Required algorithmic skeleton — HARD compliance rule (unchanged)

Keep the GVR skeleton per row: (a) `pre_idx` as threshold prior, (b) a
secant/log-style exact threshold solve (equivalent refinement structures OK),
(c) exact refine of surviving candidates. The npad<=12288 direct path (the
analytic trivial-convergence limit of the threshold solve) is compliant.
Wholesale prior-free replacements (plain radix-select/sort over rows) are
NON-COMPLIANT even if faster. No dispatch on hint quality (unknowable);
in-kernel admission escape is fine. CUDA graphs / framework kernels banned.

## What is already known about batch scaling THIS kernel (measured)

- The BS=1 champion below is a LATENCY design: 16-CTA cluster + full
  register residency per row. Naively batching it with grid.y=b is
  OCCUPANCY-LOCKED: high register counts pin it at 1 CTA/SM, and 16 CTAs
  PER ROW saturates 148 SMs at b>=10. At large b the problem flips from
  latency-bound to THROUGHPUT-bound: per-row resources must shrink
  (fewer CTAs/row, fewer registers, smaller rung ladders) as b grows.
- Measured on this data: a streaming 1-CTA/row variant with a reduced
  4-rung ladder + 2-deep hint seeding unlocked large-N high-BS
  (flash 512k BS=1024 ~1.9x vs production). That is a floor, not a target.
- The production baseline ITSELF batches natively and amortizes well —
  your denominator is strongest exactly where the champion is weakest
  (mid/high BS). Expect the required curve shape: keep the BS=1 wins,
  then win throughput at BS>=32 via occupancy, not per-row latency.
- Per-row work is IDENTICAL across rows (same data): divergence between
  rows is zero; per-row adaptive passes still differ only via atomics
  noise. Do not rely on that (correctness must hold for arbitrary rows),
  but you may exploit uniformity for performance (e.g. shared threshold
  seeding across rows is legitimate ONLY as a performance hint if
  correctness never depends on rows being equal).

## Dead ends already measured at BS=1 (do not re-discover; most still apply per row)

1. Prior-free radix/sort rewrites: banned and ~10x slow.
2. Private per-warp histograms (SM100 same-address atomics are fine).
3. Per-element ballot/popc fused count+collect: costs a full pass.
4. Staging whole row into smem before scanning: L2 re-reads are cheap.
5. >2 extra secant rounds; keep passes bounded.
6. cp.async pipelining on the scan loop (barrier overhead > latency hidden).
7. Launch-config-only retuning of the production kernel: ceiling 1.025x.
8. CUDA graphs / replay amortization: banned by the judge.

## Correctness traps (unchanged from BS=1 campaign)

- Tie boundary: never drop a strictly-greater element; ties fill remainder.
- Real data is undershoot-biased for hint-seeded thresholds.
- Cluster DSMEM: `arrive_relaxed` has no release semantics — parity-bank
  or fence your count exchanges.
- Batched twist: out-of-bounds writes to a NEIGHBOR ROW's output slice are
  the new silent killer; per-row slot counters must be row-local.

## Requirements

- CUDA C++ (sm_100a). fp32 in, int32 out, DPS, torch binding.
- Entry signature EXACTLY: run(logits, pre_idx, n_valid, indices).
- Exact per-row (tie-robust), every run, all b in 1..1024.
- One launch preferred (or few); at b=1024 launch count matters less than
  occupancy, at b=1 it is 3-29us of the budget — dispatch on (b, npad, k)
  (all known at launch) is expected.

---

## APPENDIX — BS=1 champion source digest (verbatim spans; this is the kernel whose BS=1 performance you must retain)

Full file is 1344 lines; spans below give the constants, the register-resident core, and the complete launch/dispatch layer. Reconstruct the rest from structure (phases mirror the production GVR: P1 hint-CCDF rung ladder, P2 multi-threshold count + log-secant, P3 DSMEM collect, P4 radix refine + tie-ticket writeback; plus the direct exact path for npad<=12288).

```cuda
// GVR (guess-verify-refine) top-k for DeepSeek indexer decode, BS=1, fp32, B200 (sm_100a).
//
// Skeleton (kept from the production GVR kernel, re-expressed in CUDA C++):
//   P1: gather logits[pre_idx] (temporal warm hint), build a rung ladder of
//       candidate thresholds from the hint value CCDF.
//   P2: one multi-threshold count pass over the row measures the global CCDF at
//       all 8 rungs at once; if no rung lands in [K, kC], iterate a log-space
//       secant solve (exponential-CCDF model) one threshold per pass.
//   P3: collect all (value, index) pairs >= threshold into CTA0 shared memory
//       (cluster DSMEM pushes, offsets from cached per-thread counts).
//   P4: exact refine: bitwise radix select of the K-th key among candidates,
//       then a strict-greater + tie-ticket writeback (tie-robust exactness).
// A value-plateau fallback (max-below descent + direct global emit) keeps the
// kernel exact even when no threshold can land a count inside [K, kC].
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cfloat>
#include <cmath>

namespace cg = cooperative_groups;

constexpr int RUNGS = 8;        // thresholds measured per count pass
constexpr int KCMAX = 8192;     // candidate buffer capacity
constexpr int MAXPASS = 8;      // multi-rung refine shrinks the bracket ~9x per pass

template <int TB>
struct Smem {
  static constexpr int NWARP = TB / 32;
  unsigned long long cand[KCMAX];  // packed (key<<32 | index) candidates (CTA0 only)
  int ptcnt[RUNGS * TB];            // per-thread >= counts per rung (P3 offsets)
  int hist[256];                   // P4 radix histogram / P1 hint histogram (64 bins)
  float rungs[RUNGS];              // thresholds, descending
  int rcnt_local[RUNGS];           // this CTA's slice counts
  int rcnt[RUNGS];                 // cluster-wide counts
  int rpre[RUNGS];                 // exclusive prefix of peer CTA counts
  int ipartial[2][RUNGS];          // parity-banked DSMEM exchange slots
  float fpartial[2];               // parity slots for max-below exchange
  float fwred[2 * NWARP];          // warp reduce scratch (min/max)
  int iwred[NWARP];                // warp scan scratch
  float hminmax[2];
  int sel_above[2];                // P4: selected bin, count strictly above it
  int sel_count;                   // P4: population of the selected radix bin
  int cnt_m;                       // emit: strict-greater slot counter
  int cnt_t;                       // emit: tie ticket counter
};

__device__ __forceinline__ unsigned f2u(float f) {
  unsigned u = __float_as_uint(f);
  return u ^ ((u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

// Count elements >= rungs[r] over this CTA's slice for R thresholds in one read.
// Caches per-thread counts in s->ptcnt and accumulates CTA totals in s->rcnt_local.
// Caller must zero s->rcnt_local and __syncthreads() before; call exchange after.

// …(smem struct, count/exchange/max_below/phase1/gvr_topk_kernel: as in the production-mirroring structure)…

// ---- Register-resident GVR (npad <= CS*TB*MAXV*4) --------------------------
// Same phases as gvr_topk_kernel, but each thread loads its <=MAXV float4s of
// the row into REGISTERS once, up front (the loads retire while the P1 hint
// gather runs). Every count pass, the max-below descent, the collect, and the
// plateau emit then run from registers: the row is read from memory exactly
// once, so refine iterations cost only ALU + the count exchange. Out-of-range
// lanes hold -FLT_MAX (same as the row pad), which can never pass a threshold.

template <int TB, int R, int MAXV>
__device__ __forceinline__ void count_reg(Smem<TB>* s, const float4 (&a)[MAXV], int tid) {
  float tr[R];
#pragma unroll
  for (int r = 0; r < R; ++r) tr[r] = s->rungs[r];
  int c[R];
#pragma unroll
  for (int r = 0; r < R; ++r) c[r] = 0;
#pragma unroll
  for (int u = 0; u < MAXV; ++u)
#pragma unroll
    for (int r = 0; r < R; ++r)
      c[r] += (int)(a[u].x >= tr[r]) + (int)(a[u].y >= tr[r]) + (int)(a[u].z >= tr[r]) +
              (int)(a[u].w >= tr[r]);
#pragma unroll
  for (int r = 0; r < R; ++r) s->ptcnt[r * TB + tid] = c[r];
}

template <int TB, int CS, int MAXV>
__device__ __forceinline__ float max_below_reg(Smem<TB>* s, const float4 (&a)[MAXV], float thi,
                                               int par, int tid) {
  constexpr int NWARP = TB / 32;
  float m = -FLT_MAX;
#pragma unroll
  for (int u = 0; u < MAXV; ++u) {
    if (a[u].x < thi) m = fmaxf(m, a[u].x);
    if (a[u].y < thi) m = fmaxf(m, a[u].y);
    if (a[u].z < thi) m = fmaxf(m, a[u].z);
    if (a[u].w < thi) m = fmaxf(m, a[u].w);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFu, m, o));
  int lane = tid & 31, wid = tid >> 5;
  if (lane == 0) s->fwred[wid] = m;
  __syncthreads();
  if (tid == 0) {
    float mm = -FLT_MAX;
    for (int w = 0; w < NWARP; ++w) mm = fmaxf(mm, s->fwred[w]);
    s->fpartial[par] = mm;
  }
  if constexpr (CS == 1) {
    __syncthreads();
    return s->fpartial[par];
  } else {
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    float mm = -FLT_MAX;
#pragma unroll
    for (int rr = 0; rr < CS; ++rr) {
      Smem<TB>* ps = (Smem<TB>*)cl.map_shared_rank(s, rr);
      mm = fmaxf(mm, ps->fpartial[par]);
    }
    return mm;

// …(gvr_topk_reg body, direct path, P4 helpers)…


template <int TB>
static void launch_direct(const float* logits, int* out, int npad, int K, cudaStream_t stream) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(direct_topk_kernel<TB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(DSmem<TB>));
    inited = true;
  }
  direct_topk_kernel<TB><<<1, TB, sizeof(DSmem<TB>), stream>>>(logits, out, npad, K);
}

template <int CS, int TB>
static void launch_gvr(const float* logits, const int* pre_idx, int* out, int npad, int K, int kC,
                       cudaStream_t stream) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(gvr_topk_kernel<CS, TB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(Smem<TB>));
    if (CS > 8)
      cudaFuncSetAttribute(gvr_topk_kernel<CS, TB>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    inited = true;
  }
  if constexpr (CS == 1) {
    gvr_topk_kernel<1, TB><<<1, TB, sizeof(Smem<TB>), stream>>>(logits, pre_idx, out, npad, K, kC);
  } else {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CS, 1, 1);
    cfg.blockDim = dim3(TB, 1, 1);
    cfg.dynamicSmemBytes = sizeof(Smem<TB>);
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, gvr_topk_kernel<CS, TB>, logits, pre_idx, out, npad, K, kC);
  }
}


template <int CS, int TB, int MAXV, int AR = RUNGS>
static void launch_reg(const float* logits, const int* pre_idx, int* out, int npad, int K, int kC,
                       cudaStream_t stream) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(gvr_topk_reg<CS, TB, MAXV, AR>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(Smem<TB>));
    if (CS > 8)
      cudaFuncSetAttribute(gvr_topk_reg<CS, TB, MAXV, AR>,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    inited = true;
  }
  if constexpr (CS == 1) {
    gvr_topk_reg<1, TB, MAXV, AR><<<1, TB, sizeof(Smem<TB>), stream>>>(logits, pre_idx, out, npad, K, kC);
  } else {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CS, 1, 1);
    cfg.blockDim = dim3(TB, 1, 1);
    cfg.dynamicSmemBytes = sizeof(Smem<TB>);
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, gvr_topk_reg<CS, TB, MAXV, AR>, logits, pre_idx, out, npad, K, kC);
  }
}

void gvr_topk_launch(const float* logits, const int* pre_idx, int* out, int npad, int K,
                     cudaStream_t stream) {
  int kC = (K >= 2048) ? 8192 : 6144;
  // MAXV is matched tightly to each tier: mostly-dummy register slots cost
  // real time (predicated loads + dead compares), so keep slots nearly full.
  if (npad <= DKCMAX)
    launch_direct<1024>(logits, out, npad, K, stream);
  else if (npad < 16384)
    launch_reg<1, 512, 8>(logits, pre_idx, out, npad, K, kC, stream);   // vpc <= 4095
  else if (npad < 32768)
    launch_reg<4, 512, 4>(logits, pre_idx, out, npad, K, kC, stream);   // vpc <= 2048
  else if (npad <= 49152)
    launch_reg<8, 512, 3>(logits, pre_idx, out, npad, K, kC, stream);   // vpc <= 1536
  else if (npad <= 65536)
    launch_reg<8, 512, 4>(logits, pre_idx, out, npad, K, kC, stream);   // vpc <= 2048
  else if (npad <= 131072)
    launch_reg<8, 512, 8>(logits, pre_idx, out, npad, K, kC, stream);   // vpc <= 4096
  else if (npad <= 163840)
    // AR6's shifted quantile ladder measured faster on every K=2048 cell of
    // this tier (-0.5 to -3.1us); K<=1024 regressed (+3us convergence misses).
    if (K >= 2048)
      launch_reg<16, 512, 5, 6>(logits, pre_idx, out, npad, K, kC, stream);  // vpc <= 2560
    else
      launch_reg<16, 512, 5>(logits, pre_idx, out, npad, K, kC, stream);
  else if (npad <= 262144)
    // AR6 measured faster for K=512 (-0.7 to -1.3us) and K=1024 (r2-validated)
    // at this tier; K=2048 unmeasured here, keep the denser AR8 ladder.
    if (K == 2048)
      launch_reg<16, 512, 8>(logits, pre_idx, out, npad, K, kC, stream);  // vpc <= 4096
    else
      launch_reg<16, 512, 8, 6>(logits, pre_idx, out, npad, K, kC, stream);
  else
    launch_gvr<16, 512>(logits, pre_idx, out, npad, K, kC, stream);     // streaming for huge n
}
```
