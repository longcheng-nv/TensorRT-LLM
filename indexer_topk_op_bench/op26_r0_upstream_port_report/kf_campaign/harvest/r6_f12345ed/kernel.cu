#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cfloat>

namespace cg = cooperative_groups;

// Monotonic float->uint key: larger float => larger key.
__device__ __forceinline__ unsigned int fkey(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int mask = (u >> 31) ? 0xFFFFFFFFu : 0x80000000u;
    return u ^ mask;
}

// ---------------- Single-block path (small/medium n) ----------------
__global__ void select_sb(const float* __restrict__ logits,
                          int* __restrict__ out,
                          int n, int k) {
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    __shared__ int hist[256];
    __shared__ int ssum[256];    // suffix sums
    __shared__ unsigned int s_prefix;
    __shared__ int s_kk;
    __shared__ int s_gt;
    __shared__ int s_tie;

    if (tid == 0) { s_prefix = 0u; s_kk = k; }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        const unsigned int prefix = s_prefix;
        const unsigned int hmask  = (shift + 8 >= 32) ? 0u
                                    : (0xFFFFFFFFu << (shift + 8));
        for (int i = tid; i < n; i += nt) {
            unsigned int u = fkey(logits[i]);
            if ((u & hmask) == (prefix & hmask))
                atomicAdd(&hist[(u >> shift) & 0xFF], 1);
        }
        __syncthreads();
        // Parallel inclusive suffix-sum (Hillis-Steele): ssum[b]=sum_{j>=b}hist[j].
        const int kk = s_kk;
        if (tid < 256) ssum[tid] = hist[tid];
        __syncthreads();
        #pragma unroll
        for (int off = 1; off < 256; off <<= 1) {
            int v = 0;
            if (tid < 256 && tid + off < 256) v = ssum[tid + off];
            __syncthreads();
            if (tid < 256 && tid + off < 256) ssum[tid] += v;
            __syncthreads();
        }
        // threshold bin = largest b with ssum[b] >= kk (i.e. suffix above < kk).
        if (tid < 256) {
            int hi = (tid == 255) ? 0 : ssum[tid + 1];  // strictly-above count
            if (ssum[tid] >= kk && hi < kk) {
                s_prefix = prefix | ((unsigned int)tid << shift);
                s_kk = kk - hi;
            }
        }
        __syncthreads();
    }

    const unsigned int T = s_prefix;
    const int need_tie = s_kk;
    const int ngt = k - need_tie;
    if (tid == 0) { s_gt = 0; s_tie = 0; }
    __syncthreads();

    for (int i = tid; i < n; i += nt) {
        unsigned int u = fkey(logits[i]);
        if (u > T) {
            int p = atomicAdd(&s_gt, 1);
            out[p] = i;
        } else if (u == T) {
            int p = atomicAdd(&s_tie, 1);
            if (p < need_tie) out[ngt + p] = i;
        }
    }
}

// Broadcast row0's k indices to all other rows.
__global__ void broadcast_rows(int* __restrict__ out, long long total, int k) {
    long long gstride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)k + blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += gstride) {
        int j = (int)(idx & (k - 1));
        out[idx] = out[j];
    }
}

// ---------------- Cooperative path (large n) ----------------
__device__ int g_hist[4][256];
__device__ int g_gtslot;
__device__ int g_tieslot;

__global__ void topk_coop(const float* __restrict__ logits,
                          int* __restrict__ out,
                          int n, int k, int b) {
    cg::grid_group grid = cg::this_grid();
    const int tid   = threadIdx.x;
    const int nt    = blockDim.x;
    const int gtid  = blockIdx.x * blockDim.x + tid;
    const int gstride = gridDim.x * blockDim.x;

    __shared__ int sh[256];
    __shared__ int ssum[256];
    __shared__ unsigned int s_prefix;
    __shared__ int s_kk;

    for (int i = gtid; i < 4 * 256; i += gstride) ((int*)g_hist)[i] = 0;
    if (gtid == 0) { g_gtslot = 0; g_tieslot = 0; }
    if (tid == 0) { s_prefix = 0u; s_kk = k; }
    __syncthreads();
    grid.sync();

    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int i = tid; i < 256; i += nt) sh[i] = 0;
        __syncthreads();
        const unsigned int prefix = s_prefix;
        const unsigned int hmask  = (shift + 8 >= 32) ? 0u
                                    : (0xFFFFFFFFu << (shift + 8));
        for (int i = gtid; i < n; i += gstride) {
            unsigned int u = fkey(logits[i]);
            if ((u & hmask) == (prefix & hmask))
                atomicAdd(&sh[(u >> shift) & 0xFF], 1);
        }
        __syncthreads();
        for (int i = tid; i < 256; i += nt)
            if (sh[i]) atomicAdd(&g_hist[pass][i], sh[i]);
        grid.sync();
        const int kk = s_kk;
        if (tid < 256) ssum[tid] = g_hist[pass][tid];
        __syncthreads();
        #pragma unroll
        for (int off = 1; off < 256; off <<= 1) {
            int v = 0;
            if (tid < 256 && tid + off < 256) v = ssum[tid + off];
            __syncthreads();
            if (tid < 256 && tid + off < 256) ssum[tid] += v;
            __syncthreads();
        }
        if (tid < 256) {
            int hi = (tid == 255) ? 0 : ssum[tid + 1];
            if (ssum[tid] >= kk && hi < kk) {
                s_prefix = prefix | ((unsigned int)tid << shift);
                s_kk = kk - hi;
            }
        }
        __syncthreads();
    }

    const unsigned int T = s_prefix;
    const int need_tie = s_kk;
    const int ngt = k - need_tie;

    for (int i = gtid; i < n; i += gstride) {
        unsigned int u = fkey(logits[i]);
        if (u > T) {
            int p = atomicAdd(&g_gtslot, 1);
            out[p] = i;
        } else if (u == T) {
            int p = atomicAdd(&g_tieslot, 1);
            if (p < need_tie) out[ngt + p] = i;
        }
    }
    if (b > 1) {
        grid.sync();
        long long total = (long long)b * k;
        for (long long idx = (long long)k + gtid; idx < total; idx += gstride) {
            int j = (int)(idx & (k - 1));
            out[idx] = out[j];
        }
    }
}

void topk_launcher(const float* logits, int* indices,
                   int b, int npad, int n_valid, int k,
                   cudaStream_t stream) {
    (void)npad;
    const int SB_LIMIT = 16384;

    if (n_valid <= SB_LIMIT) {
        select_sb<<<1, 1024, 0, stream>>>(logits, indices, n_valid, k);
        if (b > 1) {
            long long total = (long long)b * k;
            int block = 256;
            int grid = (int)((total - k + block - 1) / block);
            if (grid < 1) grid = 1;
            if (grid > 8192) grid = 8192;
            broadcast_rows<<<grid, block, 0, stream>>>(indices, total, k);
        }
        return;
    }

    const int block = 256;
    int numSM = 0;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, (void*)topk_coop, block, 0);
    int maxGrid = numSM * blocksPerSM;
    if (maxGrid < 1) maxGrid = numSM;

    int wantGrid = (int)(((long long)n_valid + block - 1) / block);
    int grid = wantGrid < maxGrid ? wantGrid : maxGrid;
    if (grid < 1) grid = 1;

    void* args[] = { (void*)&logits, (void*)&indices,
                     (void*)&n_valid, (void*)&k, (void*)&b };
    cudaLaunchCooperativeKernel((void*)topk_coop, grid, block, args,
                                0, stream);
}
