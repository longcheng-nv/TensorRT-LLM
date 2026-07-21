# DeepSeek-V4 Indexer Top-K Decode (BS=1, fp32, B200) — Beat the Round-2 Champion

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
>0.5). Exploit it (e.g. threshold seeding), but correctness and the
no-regression bar must hold even at 0.02 overlap. You may NOT branch on
any estimate of hint quality computed outside the kernel (hit-rate is
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

## Baseline — this is the hard part

The baseline you must beat is NOT the original production kernel. It is
the ROUND-2 CHAMPION of the previous campaign on this exact problem: a
guess-verify-refine (GVR) design with all-CTA cooperative scanning for
n ≥ 8448, per-pass early-exit, a bottom-k complement kernel for high-hint
rows, a dedicated `<<<1,1024>>>` single-CTA rung for 8448 < n ≤ 16896, and
secant+log threshold refinement seeded from `pre_idx`. It already runs
1.68× faster (external nsys, cold-L2) than the production kernel it
replaced, is exact on all 865 external cells, and regresses none of them.
Its full source is provided as the baseline solution — read it first;
incremental surgery on it is a legitimate strategy.

## Target

- **Required:** geomean speedup > 1.0× over the given champion baselines
  with NO workload slower (no-regression is a hard acceptance bar — a
  kernel that wins big on average but loses any cell will be rejected
  downstream). Meaningful wins are +5–15% geomean; single-cell heroics
  that lose elsewhere are worthless.
- Final acceptance re-measures externally with nsys cold-L2 on all 865
  real cells (the cells here are a stratified subset). Platform timings
  have a ~15µs floor that compresses your true speedup — do not tune to
  the harness floor; win in kernel time. Do not overfit to these exact n
  values: `n` is dynamic (up to ~1.05M), `k ∈ {512, 1024, 2048}` at
  runtime, hint quality is dynamic.

## Required algorithmic skeleton

Keep the GVR skeleton: (a) `pre_idx` as the threshold prior, (b) a
secant+log-transform style exact threshold solve (or an equivalent
threshold-refinement structure), (c) an exact refine of the surviving
candidates. Any per-stage restructuring that preserves exactness is
allowed (the champion's all-coop scan, early-exit, and complement kernel
are precedents). Mature primitives (histogram ladders, radix digit
passes) may be absorbed INTO stages, but do not replace the whole kernel
with a generic radix-select / full-sort top-k and do not build a
per-case dispatcher across unrelated top-k operators.

## Where the remaining time is (measured, nsys/NCU — start here)

1. **P4 block (final collect: handoff + refine + writeback) is the
   dominant cost: median ~37% of kernel time (23–58%) across the grid.**
   Known-untried levers: eliminate the leader-CTA value-handoff and
   parallelize the final collect across cluster CTAs ("distP4");
   confine the P4 threshold search to warp0 to remove 2–3 block-wide
   barriers.
2. Mid passes (scan + count + falsification) are 17–48%; per-pass
   early-exit already harvests much of this in the champion.
3. For n ≥ 512K only: a (warp, window) sideband that lets the scan skip
   provably sub-threshold windows may pay; it measured ~0 replay benefit
   at smaller n.

## Dead ends — measured net-negative on THIS workload/hardware; do not re-discover

1. BS=1 is latency-bound, not bandwidth-bound (24% occupancy, <1% DRAM at
   small n). The multi-CTA cooperation lever is ALREADY in the champion —
   re-deriving it is not progress.
2. `pre_idx` warm-hint grafted onto radix-select: no win (hint only helps
   threshold-style skeletons).
3. Private per-warp histograms to avoid smem atomics: loses (SM100
   pipelines same-address atomics fine).
4. Multi-CTA for SMALL n (< ~8K): launch/sync overhead dominates; the
   champion's single-CTA rungs win there.
5. More than 8–32 CTAs at large n: merge cost eats the scan win.
6. Per-element ballot/popc slot-reservation to fuse count+collect into
   one pass: coordination ≈ a full extra pass.
7. Staging the row into shared memory first: row re-reads are cheap L2
   hits.
8. Extra secant/interpolation refinement rounds: each is a
   barrier-separated pass; keep passes ≤2.
9. CUB DeviceRadixSort / full sort: ~10× too slow at these sizes.
10. Fusing the final-collect histogram into the P3 scan loop: pollutes
    the scan inner loop, −15%.
11. Shrinking histogram bins below 512 for k=2048: exact-tail scratch
    overflows (silent UB); kNumBins=512 at k=2048 is already in the
    production baseline.
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


## Baseline (champion) full source — the timings you must beat were measured from exactly this code

The per-workload baseline timings are platform measurements of this champion. Incremental surgery on it is a legitimate strategy.

`main.cpp`:
```cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void topk_launch(const float* logits, int n, int k, int* out,
                 cudaStream_t stream);

void run(torch::Tensor logits, torch::Tensor pre_idx, int64_t n_valid,
         torch::Tensor indices) {
    (void)pre_idx;
    topk_launch(logits.data_ptr<float>(), (int)n_valid,
                (int)indices.size(1), indices.data_ptr<int>(),
                at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Exact CUDA radix top-k");
}
```

`kernel.cu`:
```cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned int fkey(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int mask = (unsigned int)(-(int)(u >> 31)) | 0x80000000u;
    return u ^ mask;
}

__device__ __forceinline__ bool pair_less(
        unsigned int ak, int ai, unsigned int bk, int bi) {
    return ak < bk || (ak == bk && ai < bi);
}

// Exact complement specialization for k=n-3.  It finds three minima with
// three short hierarchical reductions, then emits every other index.
__global__ __launch_bounds__(512, 1)
void bottom3_kernel(const float* __restrict__ logits, int n,
                    int* __restrict__ out) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    unsigned int keys[3];
    int ids[3];
#pragma unroll
    for (int item = 0; item < 3; ++item) {
        int idx = tid + item * 512;
        keys[item] = idx < n ? fkey(logits[idx]) : 0xffffffffu;
        ids[item] = idx < n ? idx : 0x7fffffff;
    }

    __shared__ unsigned int warp_keys[16][3];
    __shared__ int warp_ids[16][3];
    __shared__ int excluded[3];

    // Each warp extracts its local three minima without block barriers.
#pragma unroll
    for (int pick = 0; pick < 3; ++pick) {
        unsigned int best_key = keys[0];
        int best_idx = ids[0];
#pragma unroll
        for (int item = 1; item < 3; ++item) {
            if (pair_less(keys[item], ids[item], best_key, best_idx)) {
                best_key = keys[item];
                best_idx = ids[item];
            }
        }
#pragma unroll
        for (int off = 16; off; off >>= 1) {
            unsigned int ok = __shfl_down_sync(0xffffffffu, best_key, off);
            int oi = __shfl_down_sync(0xffffffffu, best_idx, off);
            if (lane + off < 32 && pair_less(ok, oi, best_key, best_idx)) {
                best_key = ok;
                best_idx = oi;
            }
        }
        if (lane == 0) {
            warp_keys[warp][pick] = best_key;
            warp_ids[warp][pick] = best_idx;
        }
        int chosen = __shfl_sync(0xffffffffu, best_idx, 0);
#pragma unroll
        for (int item = 0; item < 3; ++item) {
            if (ids[item] == chosen) {
                keys[item] = 0xffffffffu;
                ids[item] = 0x7fffffff;
            }
        }
    }
    __syncthreads();

    // Warp 0 extracts the global three minima from the 16 local triples.
    if (warp == 0) {
#pragma unroll
        for (int item = 0; item < 3; ++item) {
            keys[item] = lane < 16 ? warp_keys[lane][item] : 0xffffffffu;
            ids[item] = lane < 16 ? warp_ids[lane][item] : 0x7fffffff;
        }
#pragma unroll
        for (int pick = 0; pick < 3; ++pick) {
            unsigned int best_key = keys[0];
            int best_idx = ids[0];
#pragma unroll
            for (int item = 1; item < 3; ++item) {
                if (pair_less(keys[item], ids[item], best_key, best_idx)) {
                    best_key = keys[item];
                    best_idx = ids[item];
                }
            }
#pragma unroll
            for (int off = 16; off; off >>= 1) {
                unsigned int ok = __shfl_down_sync(0xffffffffu, best_key, off);
                int oi = __shfl_down_sync(0xffffffffu, best_idx, off);
                if (lane + off < 32 && pair_less(ok, oi, best_key, best_idx)) {
                    best_key = ok;
                    best_idx = oi;
                }
            }
            if (lane == 0) excluded[pick] = best_idx;
            int chosen = __shfl_sync(0xffffffffu, best_idx, 0);
#pragma unroll
            for (int item = 0; item < 3; ++item) {
                if (ids[item] == chosen) {
                    keys[item] = 0xffffffffu;
                    ids[item] = 0x7fffffff;
                }
            }
        }
        if (lane == 0) {
            if (excluded[0] > excluded[1]) { int x=excluded[0]; excluded[0]=excluded[1]; excluded[1]=x; }
            if (excluded[1] > excluded[2]) { int x=excluded[1]; excluded[1]=excluded[2]; excluded[2]=x; }
            if (excluded[0] > excluded[1]) { int x=excluded[0]; excluded[0]=excluded[1]; excluded[1]=x; }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < n; idx += 512) {
        bool omit = idx == excluded[0] || idx == excluded[1] || idx == excluded[2];
        int rank = (excluded[0] < idx) + (excluded[1] < idx) + (excluded[2] < idx);
        if (!omit) out[idx - rank] = idx;
    }
}

// Block-wide descending histogram search.  The caller supplies one total per
// warp; 512-thread launches require all 16 entries.
__device__ __forceinline__ void find_boundary_bins(
        const unsigned int* __restrict__ hist, int nb,
        unsigned int* warp_totals, int* s_bin, int* s_above,
        int remaining) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = blockDim.x >> 5;
    int bins_per_thread = nb / blockDim.x;
    int base = tid * bins_per_thread;
    unsigned int local = 0;
    for (int j = 0; j < bins_per_thread; ++j) local += hist[base + j];
    unsigned int suffix = local;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        unsigned int v = __shfl_down_sync(0xffffffffu, suffix, off);
        if (lane + off < 32) suffix += v;
    }
    if (lane == 0) warp_totals[warp] = suffix;
    __syncthreads();
    unsigned int higher_warps = 0;
    for (int w = warp + 1; w < nwarps; ++w) higher_warps += warp_totals[w];
    unsigned int higher = suffix - local + higher_warps;
    if ((int)higher < remaining && (int)(higher + local) >= remaining) {
        unsigned int cumulative = higher;
        int boundary = base;
        int above = (int)higher;
        for (int j = bins_per_thread - 1; j >= 0; --j) {
            unsigned int next = cumulative + hist[base + j];
            if ((int)next >= remaining) {
                boundary = base + j;
                above = (int)cumulative;
                break;
            }
            cumulative = next;
        }
        *s_bin = boundary;
        *s_above = above;
    }
    __syncthreads();
}

__device__ __forceinline__ void find_boundary_update(
        const unsigned int* hist, unsigned int* s_prefix,
        int* s_remaining, int* s_total, unsigned int prefix, int shift) {
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        int remaining = *s_remaining;
        unsigned int bins[8];
        unsigned int local = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            bins[j] = hist[lane * 8 + j];
            local += bins[j];
        }
        unsigned int suffix = local;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            unsigned int v = __shfl_down_sync(0xffffffffu, suffix, off);
            if (lane + off < 32) suffix += v;
        }
        unsigned int higher = suffix - local;
        if ((int)higher < remaining && (int)(higher + local) >= remaining) {
            unsigned int cumulative = higher;
            int boundary = lane * 8;
            int above = (int)higher;
#pragma unroll
            for (int j = 7; j >= 0; --j) {
                unsigned int next = cumulative + bins[j];
                if ((int)next >= remaining) {
                    boundary = lane * 8 + j;
                    above = (int)cumulative;
                    break;
                }
                cumulative = next;
            }
            *s_prefix = prefix | ((unsigned int)boundary << shift);
            *s_total += above;
            *s_remaining = remaining - above;
        }
    }
}

template<int KPT>
__global__ __launch_bounds__(1024, 1)
void topk_small(const float* __restrict__ logits, int n, int k,
                int* __restrict__ out) {
    int tid = threadIdx.x;
    int nt = blockDim.x;
    __shared__ unsigned int hist4[4][256];
    __shared__ unsigned int prefix;
    __shared__ int remaining;
    __shared__ int total_above;
    __shared__ int gt_write;
    __shared__ int tie_write;

    unsigned int keys[KPT];
    int ids[KPT];
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int i = tid + j * nt;
        ids[j] = i;
        keys[j] = i < n ? fkey(logits[i]) : 0u;
    }
    for (int i = tid; i < 1024; i += nt) ((unsigned int*)hist4)[i] = 0;
    if (tid == 0) {
        prefix = 0;
        remaining = k;
        total_above = 0;
        gt_write = 0;
        tie_write = 0;
    }
    __syncthreads();

    int consumed = 0;
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - pass * 8;
        unsigned int high_mask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
        unsigned int p = prefix;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            unsigned int key = keys[j];
            if (ids[j] < n && (key & high_mask) == (p & high_mask))
                atomicAdd(&hist4[pass][(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        find_boundary_update(hist4[pass], &prefix, &remaining,
                             &total_above, p, shift);
        __syncthreads();
        consumed += 8;
        int selected = (prefix >> shift) & 255u;
        if ((int)hist4[pass][selected] == remaining) break;
    }

    int tail_shift = 32 - consumed;
    unsigned int threshold = tail_shift ? (prefix >> tail_shift) : prefix;
    int mandatory = total_above;
    int ties_needed = remaining;
    int lane = tid & 31;
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int i = ids[j];
        unsigned int key = keys[j];
        if (tail_shift) key >>= tail_shift;
        bool gt = i < n && key > threshold;
        bool eq = i < n && key == threshold;
        unsigned int gm = __ballot_sync(0xffffffffu, gt);
        unsigned int em = __ballot_sync(0xffffffffu, eq);
        if (gt) {
            int rank = __popc(gm & ((1u << lane) - 1));
            int leader = __ffs(gm) - 1;
            int base = 0;
            if (lane == leader) base = atomicAdd(&gt_write, __popc(gm));
            base = __shfl_sync(gm, base, leader);
            out[base + rank] = i;
        }
        if (eq) {
            int rank = __popc(em & ((1u << lane) - 1));
            int leader = __ffs(em) - 1;
            int base = 0;
            if (lane == leader) base = atomicAdd(&tie_write, __popc(em));
            base = __shfl_sync(em, base, leader);
            int slot = base + rank;
            if (slot < ties_needed) out[mandatory + slot] = i;
        }
    }
}

// scratch layout (u32): hist[6144], gt, tie.
__global__ __launch_bounds__(512, 4)
void topk_coop(const float* __restrict__ logits, int n, int k,
               unsigned int* __restrict__ scratch, int* __restrict__ out) {
    cg::grid_group grid = cg::this_grid();
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    int gstride = gridDim.x * blockDim.x;
    unsigned int* global_hist = scratch;
    int* gt_write = (int*)(scratch + 3 * 4096);
    int* tie_write = (int*)(scratch + 3 * 4096 + 1);

    __shared__ unsigned int hist[4096];
    __shared__ unsigned int warp_totals[16];
    __shared__ int boundary;
    __shared__ int above;

    if (gtid == 0) {
        *gt_write = 0;
        *tie_write = 0;
    }
    unsigned int prefix = 0;
    int remaining = k;
    int total_above = 0;
    int consumed = 0;

    const bool wide = n <= 65536 && k == 1024;
    const int histogram_stride = wide ? 4096 : 2048;
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = wide ? (pass == 0 ? 21 : (pass == 1 ? 9 : 0))
                               : (pass == 0 ? 21 : (pass == 1 ? 10 : 0));
        const int bits = wide ? (pass == 0 ? 11 : (pass == 1 ? 12 : 9))
                              : (pass == 2 ? 10 : 11);
        const int nb = 1 << bits;
        const unsigned int digit_mask = nb - 1;
        const unsigned int high_mask = pass == 0 ? 0u : (0xffffffffu << (shift + bits));
        for (int i = tid; i < nb; i += blockDim.x) hist[i] = 0;
        __syncthreads();

        const int n4 = n & ~3;
        const float4* input4 = reinterpret_cast<const float4*>(logits);
        for (int i4 = gtid; i4 < (n4 >> 2); i4 += gstride) {
            float4 v = input4[i4];
            unsigned int u0 = fkey(v.x), u1 = fkey(v.y);
            unsigned int u2 = fkey(v.z), u3 = fkey(v.w);
            if ((u0 & high_mask) == (prefix & high_mask)) atomicAdd(&hist[(u0 >> shift) & digit_mask], 1u);
            if ((u1 & high_mask) == (prefix & high_mask)) atomicAdd(&hist[(u1 >> shift) & digit_mask], 1u);
            if ((u2 & high_mask) == (prefix & high_mask)) atomicAdd(&hist[(u2 >> shift) & digit_mask], 1u);
            if ((u3 & high_mask) == (prefix & high_mask)) atomicAdd(&hist[(u3 >> shift) & digit_mask], 1u);
        }
        for (int i = n4 + gtid; i < n; i += gstride) {
            unsigned int key = fkey(logits[i]);
            if ((key & high_mask) == (prefix & high_mask))
                    atomicAdd(&hist[(key >> shift) & digit_mask], 1u);
        }
        __syncthreads();
        unsigned int* merged = global_hist + pass * histogram_stride;
        for (int i = tid; i < nb; i += blockDim.x) {
            unsigned int count = hist[i];
            if (count) atomicAdd(&merged[i], count);
        }
        grid.sync();
        // Redundant per-CTA scans are faster here than serializing all CTAs on
        // a leader mailbox: the merged histogram is resident and read-only.
        find_boundary_bins(merged, nb, warp_totals,
                           &boundary, &above, remaining);
        bool whole_bucket = above + (int)merged[boundary] == remaining;
        prefix |= (unsigned int)boundary << shift;
        total_above += above;
        remaining -= above;
        consumed += bits;
        // If every element in the boundary bucket is required, lower radix
        // digits cannot change the selected set.  Stop the dependent chain.
        if (whole_bucket) break;
    }

    int final_shift = 32 - consumed;
    unsigned int threshold = final_shift ? (prefix >> final_shift) : prefix;
    int lane = tid & 31;
    int iters = (n + gstride - 1) / gstride;
    int ties_needed = remaining;
    for (int it = 0; it < iters; ++it) {
        int i = it * gstride + gtid;
        unsigned int key = i < n ? fkey(logits[i]) : 0u;
        if (final_shift) key >>= final_shift;
        bool gt = i < n && key > threshold;
        bool eq = i < n && key == threshold;
        unsigned int gm = __ballot_sync(0xffffffffu, gt);
        unsigned int em = __ballot_sync(0xffffffffu, eq);
        if (gt) {
            int rank = __popc(gm & ((1u << lane) - 1));
            int leader = __ffs(gm) - 1;
            int base = 0;
            if (lane == leader) base = atomicAdd(gt_write, __popc(gm));
            base = __shfl_sync(gm, base, leader);
            out[base + rank] = i;
        }
        if (eq) {
            int rank = __popc(em & ((1u << lane) - 1));
            int leader = __ffs(em) - 1;
            int base = 0;
            if (lane == leader) base = atomicAdd(tie_write, __popc(em));
            base = __shfl_sync(em, base, leader);
            int slot = base + rank;
            if (slot < ties_needed) out[total_above + slot] = i;
        }
    }
    for (int i = gtid; i < 3 * histogram_stride; i += gstride) global_hist[i] = 0;
}

static unsigned int* g_scratch = nullptr;
static int g_max_coop_blocks = 0;
static int g_sms = 0;

void topk_launch(const float* logits, int n, int k, int* out,
                 cudaStream_t stream) {
    if (n - k == 3 && n <= 1536) {
        bottom3_kernel<<<1, 512, 0, stream>>>(logits, n, out);
        return;
    }
    if (n <= 1536) {
        topk_small<2><<<1, 768, 0, stream>>>(logits, n, k, out);
        return;
    }
    if (n <= 2304) {
        topk_small<3><<<1, 768, 0, stream>>>(logits, n, k, out);
        return;
    }
    if (n <= 4608) {
        topk_small<6><<<1, 768, 0, stream>>>(logits, n, k, out);
        return;
    }
    if (n <= 8448) {
        topk_small<11><<<1, 768, 0, stream>>>(logits, n, k, out);
        return;
    }
    if (n <= 16896) {
        topk_small<17><<<1, 1024, 0, stream>>>(logits, n, k, out);
        return;
    }

    if (!g_scratch) {
        cudaMalloc(&g_scratch, (3 * 4096 + 2) * sizeof(unsigned int));
        cudaMemset(g_scratch, 0, (3 * 4096 + 2) * sizeof(unsigned int));
    }
    if (!g_max_coop_blocks) {
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_coop, 512, 0);
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_max_coop_blocks = active * g_sms;
    }
    int blocks = (n + 2047) / 2048;
    if (blocks < g_sms) blocks = g_sms;
    if (blocks > g_max_coop_blocks) blocks = g_max_coop_blocks;
    void* args[] = {(void*)&logits, (void*)&n, (void*)&k,
                    (void*)&g_scratch, (void*)&out};
    cudaLaunchCooperativeKernel((void*)topk_coop, dim3(blocks), dim3(512),
                                args, 0, stream);
}
```
