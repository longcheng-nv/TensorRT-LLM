#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned int fkey(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int mask = (unsigned int)(-(int)(u >> 31)) | 0x80000000u;
    return u ^ mask;
}

__device__ __forceinline__ void find_boundary_bins(
        const unsigned int* __restrict__ hist, int nb,
        unsigned int* warp_totals, int* boundary, int* above,
        int remaining) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const int bins_per_thread = nb / blockDim.x;
    const int base = tid * bins_per_thread;
    unsigned int local_sum = 0;
#pragma unroll
    for (int j = 0; j < bins_per_thread; ++j) local_sum += hist[base + j];
    unsigned int suffix = local_sum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        unsigned int v = __shfl_down_sync(0xffffffffu, suffix, off);
        if (lane + off < 32) suffix += v;
    }
    if (lane == 0) warp_totals[warp] = suffix;
    __syncthreads();
    unsigned int higher_warps = 0;
    for (int w = warp + 1; w < nwarps; ++w) higher_warps += warp_totals[w];
    unsigned int higher = suffix - local_sum + higher_warps;
    if ((int)higher < remaining && (int)(higher + local_sum) >= remaining) {
        unsigned int cumulative = higher;
        int b = base;
        int a = (int)higher;
#pragma unroll
        for (int j = bins_per_thread - 1; j >= 0; --j) {
            unsigned int next = cumulative + hist[base + j];
            if ((int)next >= remaining) {
                b = base + j;
                a = (int)cumulative;
                break;
            }
            cumulative = next;
        }
        *boundary = b;
        *above = a;
    }
    __syncthreads();
}

__device__ __forceinline__ void find_boundary_update(
        const unsigned int* hist, unsigned int* prefix, int* remaining,
        int* total_above, unsigned int old_prefix, int shift) {
    const int tid = threadIdx.x;
    if (tid < 32) {
        const int lane = tid;
        unsigned int local[8];
        unsigned int local_sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            local[i] = hist[lane * 8 + i];
            local_sum += local[i];
        }
        unsigned int suffix = local_sum;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            unsigned int v = __shfl_down_sync(0xffffffffu, suffix, off);
            if (lane + off < 32) suffix += v;
        }
        unsigned int higher = suffix - local_sum;
        int wanted = *remaining;
        if ((int)higher < wanted && (int)(higher + local_sum) >= wanted) {
            unsigned int cumulative = higher;
            int b = lane * 8;
            int a = (int)higher;
#pragma unroll
            for (int i = 7; i >= 0; --i) {
                unsigned int next = cumulative + local[i];
                if ((int)next >= wanted) {
                    b = lane * 8 + i;
                    a = (int)cumulative;
                    break;
                }
                cumulative = next;
            }
            *prefix = old_prefix | ((unsigned int)b << shift);
            *total_above += a;
            *remaining = wanted - a;
        }
    }
}

template<int KPT>
__global__ void __launch_bounds__(1024, 1) topk_small(
        const float* __restrict__ logits, int n, int k,
        int* __restrict__ out) {
    const int tid = threadIdx.x;
    const int NT = blockDim.x;
    __shared__ unsigned int hist[4][256];
    __shared__ unsigned int prefix;
    __shared__ int remaining;
    __shared__ int total_above;
    __shared__ int gt_cursor;
    __shared__ int tie_cursor;

    unsigned int keys[KPT];
    int ids[KPT];
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int i = tid + j * NT;
        ids[j] = i;
        keys[j] = i < n ? fkey(logits[i]) : 0u;
    }
    for (int i = tid; i < 1024; i += NT) ((unsigned int*)hist)[i] = 0;
    if (tid == 0) {
        prefix = 0;
        remaining = k;
        total_above = 0;
        gt_cursor = 0;
        tie_cursor = 0;
    }
    __syncthreads();

#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        const unsigned int high_mask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
        const unsigned int p = prefix;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            unsigned int u = keys[j];
            if (ids[j] < n && (u & high_mask) == (p & high_mask))
                atomicAdd(&hist[pass][(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        find_boundary_update(hist[pass], &prefix, &remaining,
                             &total_above, p, shift);
        __syncthreads();
    }

    const unsigned int threshold = prefix;
    const int mandatory = total_above;
    const int ties_needed = remaining;
    const int lane = tid & 31;
#pragma unroll
    for (int j = 0; j < KPT; ++j) {
        const int idx = ids[j];
        const unsigned int u = keys[j];
        const bool gt = idx < n && u > threshold;
        const bool eq = idx < n && u == threshold;
        const unsigned int gm = __ballot_sync(0xffffffffu, gt);
        const unsigned int em = __ballot_sync(0xffffffffu, eq);
        if (gt) {
            const int rank = __popc(gm & ((1u << lane) - 1u));
            int base = 0;
            if (rank == 0) base = atomicAdd(&gt_cursor, __popc(gm));
            base = __shfl_sync(gm, base, __ffs(gm) - 1);
            out[base + rank] = idx;
        }
        if (eq) {
            const int rank = __popc(em & ((1u << lane) - 1u));
            int base = 0;
            if (rank == 0) base = atomicAdd(&tie_cursor, __popc(em));
            base = __shfl_sync(em, base, __ffs(em) - 1);
            const int slot = base + rank;
            if (slot < ties_needed) out[mandatory + slot] = idx;
        }
    }
}

__device__ __forceinline__ bool pair_less(
        unsigned int ak, int ai, unsigned int bk, int bi) {
    return ak < bk || (ak == bk && ai < bi);
}

__global__ void __launch_bounds__(512, 1) bottom3_kernel(
        const float* __restrict__ logits, int* __restrict__ out, int n) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    unsigned int keys[3];
    int ids[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int idx = tid + j * 512;
        keys[j] = idx < n ? fkey(logits[idx]) : 0xffffffffu;
        ids[j] = idx < n ? idx : 0x7fffffff;
    }
    __shared__ unsigned int warp_keys[16];
    __shared__ int warp_ids[16];
    __shared__ int excluded[3];
#pragma unroll
    for (int pick = 0; pick < 3; ++pick) {
        unsigned int best_key = keys[0];
        int best_idx = ids[0];
#pragma unroll
        for (int j = 1; j < 3; ++j) {
            if (pair_less(keys[j], ids[j], best_key, best_idx)) {
                best_key = keys[j];
                best_idx = ids[j];
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            unsigned int ok = __shfl_down_sync(0xffffffffu, best_key, off);
            int oi = __shfl_down_sync(0xffffffffu, best_idx, off);
            if (lane + off < 32 && pair_less(ok, oi, best_key, best_idx)) {
                best_key = ok;
                best_idx = oi;
            }
        }
        if (lane == 0) {
            warp_keys[warp] = best_key;
            warp_ids[warp] = best_idx;
        }
        __syncthreads();
        if (warp == 0) {
            best_key = lane < 16 ? warp_keys[lane] : 0xffffffffu;
            best_idx = lane < 16 ? warp_ids[lane] : 0x7fffffff;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                unsigned int ok = __shfl_down_sync(0xffffffffu, best_key, off);
                int oi = __shfl_down_sync(0xffffffffu, best_idx, off);
                if (lane + off < 32 && pair_less(ok, oi, best_key, best_idx)) {
                    best_key = ok;
                    best_idx = oi;
                }
            }
            if (lane == 0) excluded[pick] = best_idx;
        }
        __syncthreads();
        const int chosen = excluded[pick];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            if (ids[j] == chosen) {
                keys[j] = 0xffffffffu;
                ids[j] = 0x7fffffff;
            }
        }
    }
    if (tid == 0) {
        if (excluded[0] > excluded[1]) { int t = excluded[0]; excluded[0] = excluded[1]; excluded[1] = t; }
        if (excluded[1] > excluded[2]) { int t = excluded[1]; excluded[1] = excluded[2]; excluded[2] = t; }
        if (excluded[0] > excluded[1]) { int t = excluded[0]; excluded[0] = excluded[1]; excluded[1] = t; }
    }
    __syncthreads();
    for (int idx = tid; idx < n; idx += 512) {
        const bool omit = idx == excluded[0] || idx == excluded[1] || idx == excluded[2];
        const int rank = (excluded[0] < idx) + (excluded[1] < idx) + (excluded[2] < idx);
        if (!omit) out[idx - rank] = idx;
    }
}

__global__ void __launch_bounds__(512, 4) topk_coop(
        const float* __restrict__ logits, int n, int k,
        unsigned int* __restrict__ scratch, int* __restrict__ out) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;
    unsigned int* ghist = scratch;
    int* gt_cursor = (int*)(scratch + 3 * 2048);
    int* tie_cursor = (int*)(scratch + 3 * 2048 + 1);
    __shared__ unsigned int local_hist[2048];
    __shared__ unsigned int warp_totals[16];
    __shared__ int boundary;
    __shared__ int above;

    if (gtid == 0) {
        *gt_cursor = 0;
        *tie_cursor = 0;
    }
    unsigned int prefix = 0;
    int remaining = k;
    int total_above = 0;
    int consumed = 0;
    const int shifts[3] = {21, 10, 0};
    const int widths[3] = {11, 11, 10};

#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = shifts[pass];
        const int width = widths[pass];
        const int bins = 1 << width;
        const unsigned int digit_mask = bins - 1;
        const unsigned int high_mask = pass == 0 ? 0u : (0xffffffffu << (shift + width));
        for (int i = tid; i < bins; i += blockDim.x) local_hist[i] = 0;
        __syncthreads();
        const int n4 = n & ~3;
        const float4* v4 = reinterpret_cast<const float4*>(logits);
        for (int p = gtid; p < (n4 >> 2); p += stride) {
            float4 v = v4[p];
            unsigned int u0 = fkey(v.x), u1 = fkey(v.y);
            unsigned int u2 = fkey(v.z), u3 = fkey(v.w);
            if ((u0 & high_mask) == (prefix & high_mask)) atomicAdd(&local_hist[(u0 >> shift) & digit_mask], 1u);
            if ((u1 & high_mask) == (prefix & high_mask)) atomicAdd(&local_hist[(u1 >> shift) & digit_mask], 1u);
            if ((u2 & high_mask) == (prefix & high_mask)) atomicAdd(&local_hist[(u2 >> shift) & digit_mask], 1u);
            if ((u3 & high_mask) == (prefix & high_mask)) atomicAdd(&local_hist[(u3 >> shift) & digit_mask], 1u);
        }
        for (int i = n4 + gtid; i < n; i += stride) {
            unsigned int u = fkey(logits[i]);
            if ((u & high_mask) == (prefix & high_mask))
                atomicAdd(&local_hist[(u >> shift) & digit_mask], 1u);
        }
        __syncthreads();
        unsigned int* global_pass = ghist + pass * 2048;
        for (int i = tid; i < bins; i += blockDim.x) {
            unsigned int count = local_hist[i];
            if (count) atomicAdd(&global_pass[i], count);
        }
        grid.sync();
        const int before = remaining;
        find_boundary_bins(global_pass, bins, warp_totals,
                           &boundary, &above, before);
        const int b = boundary;
        const int a = above;
        prefix |= (unsigned int)b << shift;
        total_above += a;
        remaining = before - a;
        consumed += width;
        if ((int)global_pass[b] == remaining) break;
    }

    const int tail_shift = 32 - consumed;
    const unsigned int threshold = tail_shift ? (prefix >> tail_shift) : prefix;
    const int mandatory = total_above;
    const int ties_needed = remaining;
    const int lane = tid & 31;
    const int iterations = (n + stride - 1) / stride;
    for (int it = 0; it < iterations; ++it) {
        const int idx = it * stride + gtid;
        const bool valid = idx < n;
        const unsigned int u = valid ? fkey(logits[idx]) : 0u;
        const unsigned int hi = tail_shift ? (u >> tail_shift) : u;
        const bool gt = valid && hi > threshold;
        const bool eq = valid && hi == threshold;
        const unsigned int gm = __ballot_sync(0xffffffffu, gt);
        const unsigned int em = __ballot_sync(0xffffffffu, eq);
        if (gt) {
            const int rank = __popc(gm & ((1u << lane) - 1u));
            int base = 0;
            if (rank == 0) base = atomicAdd(gt_cursor, __popc(gm));
            base = __shfl_sync(gm, base, __ffs(gm) - 1);
            out[base + rank] = idx;
        }
        if (eq) {
            const int rank = __popc(em & ((1u << lane) - 1u));
            int base = 0;
            if (rank == 0) base = atomicAdd(tie_cursor, __popc(em));
            base = __shfl_sync(em, base, __ffs(em) - 1);
            const int slot = base + rank;
            if (slot < ties_needed) out[mandatory + slot] = idx;
        }
    }
    for (int i = gtid; i < 3 * 2048; i += stride) ghist[i] = 0;
}

static unsigned int* scratch_ptr = nullptr;
static int max_coop_blocks = 0;
static int sm_count = 0;

void topk_launch(const float* logits, int n, int k, int* out,
                 cudaStream_t stream) {
    if (n - k == 3 && n <= 1536) {
        bottom3_kernel<<<1, 512, 0, stream>>>(logits, out, n);
        return;
    }
    if (n <= 1024) topk_small<2><<<1, 768, 0, stream>>>(logits, n, k, out);
    else if (n <= 2048) topk_small<3><<<1, 768, 0, stream>>>(logits, n, k, out);
    else if (n <= 4096) topk_small<6><<<1, 768, 0, stream>>>(logits, n, k, out);
    else if (n <= 8448) topk_small<11><<<1, 768, 0, stream>>>(logits, n, k, out);
    else if (n <= 16384) topk_small<22><<<1, 768, 0, stream>>>(logits, n, k, out);
    else if (n <= 16896) topk_small<17><<<1, 1024, 0, stream>>>(logits, n, k, out);
    else {
        if (!scratch_ptr) {
            cudaMalloc(&scratch_ptr, (3 * 2048 + 2) * sizeof(unsigned int));
            cudaMemset(scratch_ptr, 0, (3 * 2048 + 2) * sizeof(unsigned int));
        }
        if (!max_coop_blocks) {
            int per_sm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, topk_coop, 512, 0);
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
            max_coop_blocks = per_sm * sm_count;
        }
        int blocks = (n + 2047) / 2048;
        if (blocks < sm_count) blocks = sm_count;
        if (blocks > max_coop_blocks) blocks = max_coop_blocks;
        void* args[] = {(void*)&logits, (void*)&n, (void*)&k,
                        (void*)&scratch_ptr, (void*)&out};
        cudaLaunchCooperativeKernel((void*)topk_coop, dim3(blocks),
                                    dim3(512), args, 0, stream);
    }
}
