#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

// Monotone transform: float bits -> uint so that uint order == float order.
__device__ __forceinline__ unsigned int fkey(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int mask = (unsigned int)(-(int)(u >> 31)) | 0x80000000u;
    return u ^ mask;
}

// Warp-0 boundary search over a 256-bin histogram held in s_hist.
// Finds largest bin b with sum_{j>=b} hist[j] >= remaining.
// Writes boundary bin to *s_boundary and sum_{j>b} to *s_above.
// MUST be called by the whole block; only warp 0 does work, then result
// is published to shared and made visible by the caller's __syncthreads().
// Block-wide boundary search over nb bins in histogram gh[] (shared or global).
// nt=256 (8 warps). Each thread owns bpT = nb/256 contiguous bins. Finds largest
// bin b with sum_{j>=b} gh[j] >= remaining; writes b to *s_b, sum_{j>b} to *s_a.
__device__ __forceinline__ void find_boundary_bins(const unsigned int* __restrict__ gh,
        int nb, unsigned int* wtot, int* s_b, int* s_a, int remaining) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nwarps = nt >> 5;
    const int bpT = nb / nt;                 // bins per thread
    const int base_bin = tid * bpT;
    unsigned int lane_sum = 0u;
    for (int j = 0; j < bpT; ++j) lane_sum += gh[base_bin + j];
    unsigned int suf = lane_sum;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        unsigned int v = __shfl_down_sync(0xFFFFFFFFu, suf, off);
        if (lane + off < 32) suf += v;
    }
    if (lane == 0) wtot[wid] = suf;
    __syncthreads();
    unsigned int higher_w = 0u;
    for (int w = wid + 1; w < nwarps; ++w) higher_w += wtot[w];
    unsigned int higher = (suf - lane_sum) + higher_w;
    bool mine = ((int)higher < remaining) && ((int)(higher + lane_sum) >= remaining);
    if (mine) {
        unsigned int cum = higher; int b = base_bin; int above = (int)higher;
        for (int j = bpT - 1; j >= 0; --j) {
            unsigned int c = cum + gh[base_bin + j];
            if ((int)c >= remaining) { b = base_bin + j; above = (int)cum; break; }
            cum = c;
        }
        *s_b = b; *s_a = above;
    }
    __syncthreads();
}

__device__ __forceinline__ void find_boundary_warp(const unsigned int* s_hist,
                                                    int* s_boundary, int* s_above,
                                                    int remaining) {
    const int tid = threadIdx.x;
    if (tid < 32) {
        const int lane = tid;
        unsigned int local[8];
        unsigned int lane_sum = 0u;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { local[i] = s_hist[lane * 8 + i]; lane_sum += local[i]; }
        // inclusive suffix scan across lanes: suf = sum_{m>=lane} lane_sum[m]
        unsigned int suf = lane_sum;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            unsigned int v = __shfl_down_sync(0xFFFFFFFFu, suf, off);
            if (lane + off < 32) suf += v;
        }
        unsigned int higher = suf - lane_sum;  // sum over lanes with bin range > this lane
        // The unique lane whose 8-bin range straddles the boundary.
        bool mine = ((int)higher < remaining) && ((int)(higher + lane_sum) >= remaining);
        if (mine) {
            unsigned int acc = higher;   // sum of bins strictly greater than current
            int b = lane * 8;
            int above = (int)higher;
            // walk from highest bin (7) downward
            unsigned int cum = higher;
            #pragma unroll
            for (int i = 7; i >= 0; --i) {
                unsigned int c = cum + local[i];
                if ((int)c >= remaining) { b = lane * 8 + i; above = (int)cum; break; }
                cum = c;
            }
            *s_boundary = b;
            *s_above = above;
        }
    }
}

// Warp-0 boundary search that also updates shared radix state in one shot.
// Finds largest bin b with sum_{j>=b} hist[j] >= *s_remaining, then:
//   *s_prefix   |= b << shift ;  *s_mtotal += sum_{j>b} ;  *s_remaining -= that.
__device__ __forceinline__ void find_boundary_update(const unsigned int* s_hist,
        unsigned int* s_prefix, int* s_remaining, int* s_mtotal,
        unsigned int prefix, int shift) {
    const int tid = threadIdx.x;
    if (tid < 32) {
        const int lane = tid;
        int remaining = *s_remaining;
        unsigned int local[8];
        unsigned int lane_sum = 0u;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { local[i] = s_hist[lane * 8 + i]; lane_sum += local[i]; }
        unsigned int suf = lane_sum;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            unsigned int v = __shfl_down_sync(0xFFFFFFFFu, suf, off);
            if (lane + off < 32) suf += v;
        }
        unsigned int higher = suf - lane_sum;
        bool mine = ((int)higher < remaining) && ((int)(higher + lane_sum) >= remaining);
        if (mine) {
            unsigned int cum = higher; int b = lane * 8; int above = (int)higher;
            #pragma unroll
            for (int i = 7; i >= 0; --i) {
                unsigned int c = cum + local[i];
                if ((int)c >= remaining) { b = lane * 8 + i; above = (int)cum; break; }
                cum = c;
            }
            *s_prefix = prefix | (((unsigned int)b) << shift);
            *s_mtotal += above;
            *s_remaining = remaining - above;
        }
    }
}

// =================== SINGLE-BLOCK KERNEL (small n) ===================
// Keys are loaded ONCE into per-thread registers (strided so lane i owns
// indices tid, tid+nt, ...) and reused across all 4 radix passes + collect,
// eliminating the repeated L2 load latency that dominates the latency-bound
// single-SM regime. KPT = max keys per thread (nt * KPT >= n).
template<int KPT>
__global__ void __launch_bounds__(1024, 1)
topk_kernel_sb(const float* __restrict__ logits, int n, int k,
               int* __restrict__ out) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    // 4 separate 8-bit histogram buffers (one per pass) so no re-zero barrier.
    __shared__ unsigned int s_hist4[4][256];
    __shared__ unsigned int s_prefix;
    __shared__ int s_remaining;
    __shared__ int s_mtotal;
    __shared__ int s_gt;
    __shared__ int s_tie;

    // Load & encode keys once, reused across all radix passes + collect.
    unsigned int rk[KPT];
    int ridx[KPT];
    #pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int i = tid + j * nt;
        ridx[j] = i;
        rk[j] = (i < n) ? fkey(logits[i]) : 0u;
    }

    for (int i = tid; i < 4 * 256; i += nt) ((unsigned int*)s_hist4)[i] = 0u;
    if (tid == 0) { s_prefix = 0; s_remaining = k; s_mtotal = 0; s_gt = 0; s_tie = 0; }
    __syncthreads();

    for (int d = 0; d < 4; ++d) {
        const int s = 24 - 8 * d;
        const unsigned int mask_high = (d == 0) ? 0u : (0xFFFFFFFFu << (s + 8));
        unsigned int prefix = s_prefix;
        unsigned int* s_hist = s_hist4[d];
        #pragma unroll
        for (int j = 0; j < KPT; ++j) {
            if (ridx[j] < n) {
                unsigned int key = rk[j];
                if ((key & mask_high) == (prefix & mask_high))
                    atomicAdd(&s_hist[(key >> s) & 0xFFu], 1u);
            }
        }
        __syncthreads();
        find_boundary_update(s_hist, &s_prefix, &s_remaining, &s_mtotal, prefix, s);
        __syncthreads();
    }


    const unsigned int tau = s_prefix;
    const int m = s_mtotal;
    const int ties_needed = s_remaining;
    const int lane = tid & 31;
    #pragma unroll
    for (int j = 0; j < KPT; ++j) {
        int i = ridx[j];
        bool valid = i < n;
        unsigned int key = rk[j];
        bool gt = valid && (key > tau);
        bool eq = valid && (key == tau);
        unsigned int gtmask = __ballot_sync(0xFFFFFFFFu, gt);
        unsigned int eqmask = __ballot_sync(0xFFFFFFFFu, eq);
        if (gt) {
            int rank = __popc(gtmask & ((1u << lane) - 1u));
            int base;
            if (rank == 0) base = atomicAdd(&s_gt, __popc(gtmask));
            base = __shfl_sync(gtmask, base, __ffs(gtmask) - 1);
            out[base + rank] = i;
        }
        if (eq) {
            int rank = __popc(eqmask & ((1u << lane) - 1u));
            int base;
            if (rank == 0) base = atomicAdd(&s_tie, __popc(eqmask));
            base = __shfl_sync(eqmask, base, __ffs(eqmask) - 1);
            int slot = base + rank;
            if (slot < ties_needed) out[m + slot] = i;
        }
    }
}

// =================== COOPERATIVE MULTI-CTA KERNEL (large n) ===================
// scratch: g_hist[4*256] uint, then counters: [gt, tie] int
__global__ void __launch_bounds__(512, 4)
topk_kernel_coop(const float* __restrict__ logits, int n, int k,
                 unsigned int* __restrict__ scratch,
                 int* __restrict__ out) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int gtid = blockIdx.x * nt + tid;
    const int gstride = gridDim.x * nt;

    unsigned int* g_hist = scratch;           // 3*2048
    int* g_gt = (int*)(scratch + 3 * 2048);
    int* g_tie = (int*)(scratch + 3 * 2048 + 1);

    // 3-pass radix with 11/11/10-bit digits (2048 bins) => 3 merge grid.syncs.
    const int BW[3] = {11, 11, 10};
    const int SH[3] = {21, 10, 0};
    __shared__ unsigned int s_big[2048];
    __shared__ unsigned int wtot[8];
    __shared__ int s_boundary;
    __shared__ int s_above;

    // Histograms are pre-zeroed (once at alloc, then re-zeroed by the previous
    // call's collect pass). Counters are zeroed here and become visible via
    // pass-0's merge grid.sync (well before collect). No dedicated zeroing sync.
    if (gtid == 0) { g_gt[0] = 0; g_tie[0] = 0; }
    unsigned int prefix = 0;
    int remaining = k;
    int mtotal = 0;

    for (int d = 0; d < 3; ++d) {
        const int s = SH[d];
        const int bw = BW[d];
        const unsigned int bmask = (1u << bw) - 1u;
        const unsigned int mask_high = (d == 0) ? 0u : (0xFFFFFFFFu << (s + bw));
        const int nb = 1 << bw;
        for (int i = tid; i < nb; i += nt) s_big[i] = 0u;
        __syncthreads();
        // vectorized float4 loads over the bulk; scalar tail.
        int n4 = n & ~3;
        const float4* l4 = reinterpret_cast<const float4*>(logits);
        for (int i4 = gtid; i4 < (n4 >> 2); i4 += gstride) {
            float4 v = l4[i4];
            float fv[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                unsigned int key = fkey(fv[q]);
                if ((key & mask_high) == (prefix & mask_high))
                    atomicAdd(&s_big[(key >> s) & bmask], 1u);
            }
        }
        for (int i = n4 + gtid; i < n; i += gstride) {
            unsigned int key = fkey(logits[i]);
            if ((key & mask_high) == (prefix & mask_high))
                atomicAdd(&s_big[(key >> s) & bmask], 1u);
        }
        __syncthreads();
        unsigned int* gh = g_hist + d * 2048;
        for (int i = tid; i < nb; i += nt) {
            unsigned int v = s_big[i];
            if (v) atomicAdd(&gh[i], v);
        }
        grid.sync();
        find_boundary_bins(gh, nb, wtot, &s_boundary, &s_above, remaining);
        __syncthreads();
        int boundary = s_boundary;
        int above = s_above;
        mtotal += above;
        remaining -= above;
        prefix |= ((unsigned int)boundary) << s;
    }

    const unsigned int tau = prefix;
    const int m = mtotal;
    const int ties_needed = remaining;
    const int lane = tid & 31;
    const int iters = (n + gstride - 1) / gstride;
    for (int it = 0; it < iters; ++it) {
        int i = it * gstride + gtid;
        bool valid = i < n;
        unsigned int key = valid ? fkey(logits[i]) : 0u;
        bool gt = valid && (key > tau);
        bool eq = valid && (key == tau);
        unsigned int gtmask = __ballot_sync(0xFFFFFFFFu, gt);
        unsigned int eqmask = __ballot_sync(0xFFFFFFFFu, eq);
        if (gt) {
            int rank = __popc(gtmask & ((1u << lane) - 1u));
            int base;
            if (rank == 0) base = atomicAdd(g_gt, __popc(gtmask));
            base = __shfl_sync(gtmask, base, __ffs(gtmask) - 1);
            out[base + rank] = i;
        }
        if (eq) {
            int rank = __popc(eqmask & ((1u << lane) - 1u));
            int base;
            if (rank == 0) base = atomicAdd(g_tie, __popc(eqmask));
            base = __shfl_sync(eqmask, base, __ffs(eqmask) - 1);
            int slot = base + rank;
            if (slot < ties_needed) out[m + slot] = i;
        }
    }
    // Re-zero the histogram regions for the NEXT call (they are dead now: all
    // boundary reads happened before collect). No barrier needed - purely a
    // write of scratch that this call will never read again.
    for (int i = gtid; i < 3 * 2048; i += gstride) g_hist[i] = 0u;
}

static unsigned int* g_scratch = nullptr;
static int g_coopBlocks = 0;
static int g_numSM = 0;

void topk_launch(const float* logits, int n, int k, int* out, cudaStream_t stream) {
    const int SMALL_N = 16384;
    if (n <= SMALL_N) {
        if (n <= 1024)       topk_kernel_sb<2><<<1, 768, 0, stream>>>(logits, n, k, out);
        else if (n <= 2048)  topk_kernel_sb<3><<<1, 768, 0, stream>>>(logits, n, k, out);
        else if (n <= 4096)  topk_kernel_sb<6><<<1, 768, 0, stream>>>(logits, n, k, out);
        else if (n <= 8192)  topk_kernel_sb<11><<<1, 768, 0, stream>>>(logits, n, k, out);
        else                 topk_kernel_sb<11><<<1, 768, 0, stream>>>(logits, n, k, out);
        return;
    }
    if (!g_scratch) {
        cudaMalloc(&g_scratch, (3 * 2048 + 2) * sizeof(unsigned int));
        cudaMemset(g_scratch, 0, (3 * 2048 + 2) * sizeof(unsigned int));  // one-time init
    }
    if (!g_coopBlocks) {
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, topk_kernel_coop, 512, 0);
        cudaDeviceGetAttribute(&g_numSM, cudaDevAttrMultiProcessorCount, 0);
        g_coopBlocks = nb * g_numSM;
        if (g_coopBlocks < 1) g_coopBlocks = 1;
    }
    int want = (n + 2047) / 2048;
    if (want < g_numSM) want = g_numSM;
    if (want > g_coopBlocks) want = g_coopBlocks;
    void* args[] = {(void*)&logits, (void*)&n, (void*)&k,
                    (void*)&g_scratch, (void*)&out};
    dim3 grid(want), blk(512);
    cudaLaunchCooperativeKernel((void*)topk_kernel_coop, grid, blk, args, 0, stream);
}
