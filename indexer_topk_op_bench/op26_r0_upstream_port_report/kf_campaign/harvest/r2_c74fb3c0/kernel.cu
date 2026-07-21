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
__global__ __launch_bounds__(768, 1)
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
