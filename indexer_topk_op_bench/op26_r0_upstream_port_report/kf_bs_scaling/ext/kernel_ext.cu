#include <cuda_runtime.h>
#include <cstdint>
#include <cooperative_groups.h>

namespace aefm {

// ---------------------------------------------------------------------------
// Exact top-k index selection, radix-select family.
//
// Large-n path (single regular launch, grid sized to co-residency):
//   pass 0: 11-bit MSB histogram of monotonic float keys, built from
//           register-cached keys, merged to global; ONE spinning global
//           barrier (sense-token, launch-generation unique).
//   Every block then finds the boundary bin.  Three adaptive finishes:
//     (a) whole bucket needed -> classify straight to output, done.
//     (b) boundary bucket small (<= CAP) -> smem-staged compaction of the
//         bucket into a global tie buffer; blocks arrive at a non-spinning
//         rendezvous and exit; the LAST arriver alone refines the low 21
//         bits with two shared-memory histogram passes.  No further global
//         barriers, no barrier drain.
//     (c) boundary bucket large -> classic 11/11/10 ladder (two more
//         spinning barriers) over register-cached keys.
// Scratch is re-zeroed by the last-arriving block only, after all readers
// are provably done (they arrived), so repeated launches stay race-free.
// ---------------------------------------------------------------------------

#define BLOCK 512
#define CAP 4096            // max boundary-bucket size for the fast tail
#define PT (CAP / BLOCK)    // tie pairs per thread in the tail block

// scratch layout (u32 words)
#define HIST0 0             // 2048
#define HIST1 2048          // 2048
#define HIST2 4096          // 1024
#define CNT   5120          // gt_write, tie_write, arrive, release, tail_arrive
#define TIEBUF 5128         // CAP uint2 pairs (2*CAP words)
#define SCRATCH_WORDS (TIEBUF + 2 * CAP)

__device__ __forceinline__ unsigned int fkey(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int mask = (unsigned int)(-(int)(u >> 31)) | 0x80000000u;
    return u ^ mask;
}

__device__ __forceinline__ bool pair_less(
        unsigned int ak, int ai, unsigned int bk, int bi) {
    return ak < bk || (ak == bk && ai < bi);
}

// Spinning grid barrier for a regular launch (all blocks co-resident).
// `sense` is globally monotonic (host generation * 8 + barrier index), so
// stale release values from previous launches are never mistaken for the
// current barrier and no per-launch reset is needed.
__device__ __forceinline__ void global_barrier(unsigned int* arrive,
                                               unsigned int* release,
                                               int gridsz, unsigned int sense) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int a = atomicAdd(arrive, 1u) + 1u;
        if (a == (unsigned int)gridsz) {
            atomicExch(arrive, 0u);
            atomicExch(release, sense);
        } else {
            while (atomicAdd(release, 0u) != sense) { }
        }
    }
    __syncthreads();
}

// Block-wide descending boundary search over an nb-bin histogram (global or
// shared).  nb must be a multiple of blockDim.x.  Exactly one thread finds
// the boundary and publishes bin/above; ends with __syncthreads().
__device__ __forceinline__ void find_boundary_bins(
        const unsigned int* __restrict__ hist, int nb,
        unsigned int* warp_totals, int* s_bin, int* s_above,
        int remaining) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int bins_per_thread = nb >> 9;   // nb / BLOCK, BLOCK == 512
    const int base = tid * bins_per_thread;
    unsigned int b[4];
    unsigned int local = 0;
    if (bins_per_thread == 4) {
        uint4 v = *reinterpret_cast<const uint4*>(hist + base);
        b[0] = v.x; b[1] = v.y; b[2] = v.z; b[3] = v.w;
        local = v.x + v.y + v.z + v.w;
    } else {
        uint2 v = *reinterpret_cast<const uint2*>(hist + base);
        b[0] = v.x; b[1] = v.y; b[2] = 0u; b[3] = 0u;
        local = v.x + v.y;
    }
    unsigned int suffix = local;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        unsigned int v = __shfl_down_sync(0xffffffffu, suffix, off);
        if (lane + off < 32) suffix += v;
    }
    if (lane == 0) warp_totals[warp] = suffix;
    __syncthreads();
    unsigned int higher_warps = 0;
#pragma unroll
    for (int w = 0; w < (BLOCK >> 5); ++w)
        if (w > warp) higher_warps += warp_totals[w];
    unsigned int higher = suffix - local + higher_warps;
    if ((int)higher < remaining && (int)(higher + local) >= remaining) {
        unsigned int cumulative = higher;
        int boundary = base;
        int above = (int)higher;
#pragma unroll
        for (int j = 3; j >= 0; --j) {
            if (j >= bins_per_thread) continue;
            unsigned int next = cumulative + b[j];
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

// ---------------------------------------------------------------------------
// Small-n single-CTA kernels (unchanged champion structure).
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(512, 1)
void bottom3_kernel(const float* __restrict__ logits, long rstride, int n,
                    int* __restrict__ out) {
    logits += (size_t)blockIdx.y * (size_t)rstride;
    out += (size_t)blockIdx.y * (size_t)(n - 3);
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
void topk_small(const float* __restrict__ logits, long rstride, int n, int k,
                int* __restrict__ out) {
    logits += (size_t)blockIdx.y * (size_t)rstride;
    out += (size_t)blockIdx.y * (size_t)k;
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

// ---------------------------------------------------------------------------
// Large-n kernel.
// ---------------------------------------------------------------------------

template<int MINB>
__global__ __launch_bounds__(BLOCK, MINB)
void topk_fast(const float* __restrict__ logits, long rstride, int n, int k,
               unsigned int* __restrict__ scratch, int* __restrict__ out,
               unsigned int gen, int team) {
    const int row = blockIdx.x / team;
    const int bx = blockIdx.x - row * team;
    logits += (size_t)row * (size_t)rstride;
    out += (size_t)row * (size_t)k;
    scratch += (size_t)row * (size_t)SCRATCH_WORDS;
    const int gridsz = team;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int gtid = bx * BLOCK + tid;
    const int gstride = gridsz * BLOCK;

    unsigned int* hist0_g = scratch + HIST0;
    unsigned int* gt_write = scratch + CNT + 0;
    unsigned int* tie_write = scratch + CNT + 1;
    unsigned int* g_arrive = scratch + CNT + 2;
    unsigned int* g_release = scratch + CNT + 3;
    unsigned int* tail_arrive = scratch + CNT + 4;
    uint2* tiebuf = reinterpret_cast<uint2*>(scratch + TIEBUF);

    // shared: 2048-word histogram, reused after the first barrier as the
    // per-block compaction staging area (gt indices + tie pairs).
    __shared__ __align__(16) unsigned char smem_raw[2048 * 4 + CAP * 8];
    unsigned int* sh_hist = reinterpret_cast<unsigned int*>(smem_raw);
    int* st_gt = reinterpret_cast<int*>(smem_raw);
    uint2* st_tie = reinterpret_cast<uint2*>(smem_raw + 2048 * 4);

    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin;
    __shared__ int s_above;
    __shared__ int s_ngt, s_ntie, s_gtbase, s_tiebase, s_cnt, s_flag;
    __shared__ unsigned int s_eqk[32];
    __shared__ int s_eqv[32];

    // ---- register-cached keys: one float4 + one tail scalar per thread ----
    const int n4 = n & ~3;
    const int nv4 = n4 >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(logits);
    unsigned int kv[4];
    const bool vok = gtid < nv4;
    if (vok) {
        float4 f = in4[gtid];
        kv[0] = fkey(f.x); kv[1] = fkey(f.y);
        kv[2] = fkey(f.z); kv[3] = fkey(f.w);
    } else {
        kv[0] = kv[1] = kv[2] = kv[3] = 0u;
    }
    const int ptail = n4 + gtid;
    const bool tok = ptail < n;
    const unsigned int ktail = tok ? fkey(logits[ptail]) : 0u;
    // overflow region (n beyond register capacity); never taken for the
    // benchmark shapes but keeps the kernel correct for any n.
    const int ov_iters = nv4 > gstride
        ? (nv4 - gstride + gstride - 1) / gstride : 0;

    if (gtid == 0) { *gt_write = 0u; *tie_write = 0u; }

    // ---- pass 0: 11-bit MSB histogram ----
    for (int i = tid; i < 2048; i += BLOCK) sh_hist[i] = 0u;
    __syncthreads();
    if (vok) {
        atomicAdd(&sh_hist[kv[0] >> 21], 1u);
        atomicAdd(&sh_hist[kv[1] >> 21], 1u);
        atomicAdd(&sh_hist[kv[2] >> 21], 1u);
        atomicAdd(&sh_hist[kv[3] >> 21], 1u);
    }
    if (tok) atomicAdd(&sh_hist[ktail >> 21], 1u);
    for (int it = 0; it < ov_iters; ++it) {
        int i4 = gstride + it * gstride + gtid;
        if (i4 < nv4) {
            float4 f = in4[i4];
            atomicAdd(&sh_hist[fkey(f.x) >> 21], 1u);
            atomicAdd(&sh_hist[fkey(f.y) >> 21], 1u);
            atomicAdd(&sh_hist[fkey(f.z) >> 21], 1u);
            atomicAdd(&sh_hist[fkey(f.w) >> 21], 1u);
        }
    }
    __syncthreads();
    // merge as packed 64-bit adds: per-bin totals < 2^20, no carry between
    // the two 32-bit halves; halves the global atomic count.
    {
        const uint2* sh2 = reinterpret_cast<const uint2*>(sh_hist);
        unsigned long long* g2 = reinterpret_cast<unsigned long long*>(hist0_g);
        for (int i = tid; i < 1024; i += BLOCK) {
            uint2 c = sh2[i];
            if (c.x | c.y)
                atomicAdd(&g2[i], ((unsigned long long)c.y << 32) | c.x);
        }
    }
    global_barrier(g_arrive, g_release, gridsz, gen * 8u);

    find_boundary_bins(hist0_g, 2048, warp_totals, &s_bin, &s_above, k);
    const int b0 = s_bin;
    const int above0 = s_above;
    const int T = (int)hist0_g[b0];
    const int R = k - above0;
    const bool whole = T == R;

    if (whole || T <= CAP) {
        // ---- fast finishes: staged compaction, then arrive-and-exit ----
        if (tid == 0) { s_ngt = 0; s_ntie = 0; }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (vok) {
                unsigned int d = kv[j] >> 21;
                if ((int)d > b0) {
                    st_gt[atomicAdd(&s_ngt, 1)] = 4 * gtid + j;
                } else if ((int)d == b0) {
                    st_tie[atomicAdd(&s_ntie, 1)] =
                        make_uint2(kv[j], (unsigned int)(4 * gtid + j));
                }
            }
        }
        if (tok) {
            unsigned int d = ktail >> 21;
            if ((int)d > b0) st_gt[atomicAdd(&s_ngt, 1)] = ptail;
            else if ((int)d == b0)
                st_tie[atomicAdd(&s_ntie, 1)] =
                    make_uint2(ktail, (unsigned int)ptail);
        }
        for (int it = 0; it < ov_iters; ++it) {
            int i4 = gstride + it * gstride + gtid;
            if (i4 < nv4) {
                float4 f = in4[i4];
                unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    unsigned int d = kk[j] >> 21;
                    if ((int)d > b0) st_gt[atomicAdd(&s_ngt, 1)] = 4 * i4 + j;
                    else if ((int)d == b0)
                        st_tie[atomicAdd(&s_ntie, 1)] =
                            make_uint2(kk[j], (unsigned int)(4 * i4 + j));
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            s_gtbase = s_ngt ? (int)atomicAdd(gt_write, (unsigned int)s_ngt) : 0;
            s_tiebase = s_ntie ? (int)atomicAdd(tie_write, (unsigned int)s_ntie) : 0;
        }
        __syncthreads();
        for (int i = tid; i < s_ngt; i += BLOCK) out[s_gtbase + i] = st_gt[i];
        if (whole) {
            for (int i = tid; i < s_ntie; i += BLOCK)
                out[above0 + s_tiebase + i] = (int)st_tie[i].y;
        } else {
            // L2-direct atomic stores: visible to the last arriver through the
            // same atomic-chain contract the spinning barrier relies on.
            unsigned long long* tb64 =
                reinterpret_cast<unsigned long long*>(tiebuf + s_tiebase);
            for (int i = tid; i < s_ntie; i += BLOCK) {
                uint2 e = st_tie[i];
                atomicExch(&tb64[i], ((unsigned long long)e.y << 32) | e.x);
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int t = atomicAdd(tail_arrive, 1u);
            s_flag = (t + 1u == (unsigned int)gridsz) ? 1 : 0;
        }
        __syncthreads();
        if (!s_flag) return;
        // ---- last arriver: exclusive owner of scratch from here ----
        if (tid == 0) atomicExch(tail_arrive, 0u);
        asm volatile("fence.acq_rel.gpu;" ::: "memory");

        if (!whole) {
            // refine low 21 bits of the T (<= CAP) boundary-bucket pairs
            unsigned int tk[PT];
            int tv[PT];
#pragma unroll
            for (int j = 0; j < PT; ++j) {
                if (j * BLOCK >= T) break;
                int p = tid + j * BLOCK;
                if (p < T) {
                    uint2 e = tiebuf[p];
                    tk[j] = e.x; tv[j] = (int)e.y;
                } else { tk[j] = 0u; tv[j] = 0; }
            }
            __syncthreads();
            // pass A: bits 20..10 (2048 bins)
            for (int i = tid; i < 2048; i += BLOCK) sh_hist[i] = 0u;
            __syncthreads();
#pragma unroll
            for (int j = 0; j < PT; ++j) {
                if (j * BLOCK >= T) break;
                if (tid + j * BLOCK < T)
                    atomicAdd(&sh_hist[(tk[j] >> 10) & 2047u], 1u);
            }
            __syncthreads();
            find_boundary_bins(sh_hist, 2048, warp_totals, &s_bin, &s_above, R);
            const int bA = s_bin;
            const int aboveA = s_above;
            const int TA = (int)sh_hist[bA];
            const bool wholeA = aboveA + TA == R;
            if (tid == 0) { s_cnt = 0; s_ntie = 0; }
            __syncthreads();
            // combined collect: pass-A winners straight to out; boundary-bin
            // ties gathered into a 32-slot shared list for the warp finisher.
#pragma unroll
            for (int j = 0; j < PT; ++j) {
                if (j * BLOCK >= T) break;
                bool active = tid + j * BLOCK < T;
                int binA = active ? (int)((tk[j] >> 10) & 2047u) : -1;
                bool p = wholeA ? (binA >= bA) : (binA > bA);
                unsigned int m = __ballot_sync(0xffffffffu, p);
                if (p) {
                    int r = __popc(m & ((1u << lane) - 1));
                    int leader = __ffs(m) - 1;
                    int base = 0;
                    if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                    base = __shfl_sync(m, base, leader);
                    out[above0 + base + r] = tv[j];
                }
                if (!wholeA && binA == bA) {
                    int q = atomicAdd(&s_ntie, 1);
                    if (q < 32) { s_eqk[q] = tk[j]; s_eqv[q] = tv[j]; }
                }
            }
            __syncthreads();
            if (!wholeA) {
                const int R2 = R - aboveA;
                if (TA <= 32) {
                    // warp-serial top-R2 of the <=32 boundary ties: R2 rounds
                    // of max-reduction over the low 10 bits (full-key ties at
                    // the k-th value may resolve either way).
                    if (tid < 32) {
                        unsigned int score =
                            tid < TA ? ((s_eqk[tid] & 1023u) + 1u) : 0u;
                        for (int r = 0; r < R2; ++r) {
                            unsigned int bs = score;
                            int bl = tid;
#pragma unroll
                            for (int off = 16; off; off >>= 1) {
                                unsigned int os =
                                    __shfl_down_sync(0xffffffffu, bs, off);
                                int ol = __shfl_down_sync(0xffffffffu, bl, off);
                                if (tid + off < 32 && os > bs) { bs = os; bl = ol; }
                            }
                            int winner = __shfl_sync(0xffffffffu, bl, 0);
                            if (tid == winner) {
                                out[above0 + aboveA + r] = s_eqv[tid];
                                score = 0u;
                            }
                        }
                    }
                } else {
                    // rare big-tie fallback: pass B over bits 9..0 (1024 bins)
                    __syncthreads();
                    for (int i = tid; i < 1024; i += BLOCK) sh_hist[i] = 0u;
                    __syncthreads();
#pragma unroll
                    for (int j = 0; j < PT; ++j) {
                        if (j * BLOCK >= T) break;
                        if ((tid + j * BLOCK < T) &&
                            (int)((tk[j] >> 10) & 2047u) == bA)
                            atomicAdd(&sh_hist[tk[j] & 1023u], 1u);
                    }
                    __syncthreads();
                    find_boundary_bins(sh_hist, 1024, warp_totals,
                                       &s_bin, &s_above, R2);
                    const int bB = s_bin;
#pragma unroll
                    for (int j = 0; j < PT; ++j) {
                        if (j * BLOCK >= T) break;
                        bool inA = (tid + j * BLOCK < T) &&
                                   (int)((tk[j] >> 10) & 2047u) == bA;
                        bool p = inA && (int)(tk[j] & 1023u) > bB;
                        unsigned int m = __ballot_sync(0xffffffffu, p);
                        if (p) {
                            int r = __popc(m & ((1u << lane) - 1));
                            int leader = __ffs(m) - 1;
                            int base = 0;
                            if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                            base = __shfl_sync(m, base, leader);
                            out[above0 + base + r] = tv[j];
                        }
                    }
                    __syncthreads();
                    // full-key ties at the exact k-th value: any subset fills
#pragma unroll
                    for (int j = 0; j < PT; ++j) {
                        if (j * BLOCK >= T) break;
                        bool inA = (tid + j * BLOCK < T) &&
                                   (int)((tk[j] >> 10) & 2047u) == bA;
                        bool p = inA && (int)(tk[j] & 1023u) == bB;
                        unsigned int m = __ballot_sync(0xffffffffu, p);
                        if (p) {
                            int r = __popc(m & ((1u << lane) - 1));
                            int leader = __ffs(m) - 1;
                            int base = 0;
                            if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                            base = __shfl_sync(m, base, leader);
                            int slot = base + r;
                            if (slot < R) out[above0 + slot] = tv[j];
                        }
                    }
                }
            }
        }
        // re-zero pass-0 histogram for the next launch (only block touching
        // scratch at this point; every other block has already arrived).
        for (int i = tid; i < 2048; i += BLOCK) hist0_g[i] = 0u;
        return;
    }

    // ---- fallback: boundary bucket too large; classic 11/11/10 ladder ----
    int remaining = R;
    int total_above = above0;
    unsigned int prefix = ((unsigned int)b0) << 21;
    int consumed = 11;
#pragma unroll
    for (int pass = 1; pass < 3; ++pass) {
        const int shift = pass == 1 ? 10 : 0;
        const int bits = pass == 1 ? 11 : 10;
        const int nb = 1 << bits;
        const unsigned int digit_mask = (unsigned int)(nb - 1);
        const unsigned int high_mask = 0xffffffffu << (shift + bits);
        for (int i = tid; i < nb; i += BLOCK) sh_hist[i] = 0u;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (vok && (kv[j] & high_mask) == (prefix & high_mask))
                atomicAdd(&sh_hist[(kv[j] >> shift) & digit_mask], 1u);
        if (tok && (ktail & high_mask) == (prefix & high_mask))
            atomicAdd(&sh_hist[(ktail >> shift) & digit_mask], 1u);
        for (int it = 0; it < ov_iters; ++it) {
            int i4 = gstride + it * gstride + gtid;
            if (i4 < nv4) {
                float4 f = in4[i4];
                unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if ((kk[j] & high_mask) == (prefix & high_mask))
                        atomicAdd(&sh_hist[(kk[j] >> shift) & digit_mask], 1u);
            }
        }
        __syncthreads();
        unsigned int* merged = scratch + (pass == 1 ? HIST1 : HIST2);
        for (int i = tid; i < nb; i += BLOCK) {
            unsigned int c = sh_hist[i];
            if (c) atomicAdd(&merged[i], c);
        }
        global_barrier(g_arrive, g_release, gridsz, gen * 8u + (unsigned int)pass);
        find_boundary_bins(merged, nb, warp_totals, &s_bin, &s_above, remaining);
        bool whole_bucket = s_above + (int)merged[s_bin] == remaining;
        prefix |= ((unsigned int)s_bin) << shift;
        total_above += s_above;
        remaining -= s_above;
        consumed += bits;
        if (whole_bucket) break;
    }

    // non-spinning rendezvous: after this point every block has finished
    // reading the merged histograms, so the last arriver may re-zero them.
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        unsigned int t = atomicAdd(tail_arrive, 1u);
        s_flag = (t + 1u == (unsigned int)gridsz) ? 1 : 0;
    }
    __syncthreads();
    const bool last = s_flag != 0;

    const int final_shift = 32 - consumed;
    const unsigned int threshold = final_shift ? (prefix >> final_shift) : prefix;
    const int ties_needed = remaining;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        bool valid = j < 4 ? vok : tok;
        unsigned int key = j < 4 ? kv[j] : ktail;
        int i = j < 4 ? 4 * gtid + j : ptail;
        if (final_shift) key >>= final_shift;
        bool gt = valid && key > threshold;
        bool eq = valid && key == threshold;
        unsigned int gm = __ballot_sync(0xffffffffu, gt);
        unsigned int em = __ballot_sync(0xffffffffu, eq);
        if (gt) {
            int rank = __popc(gm & ((1u << lane) - 1));
            int leader = __ffs(gm) - 1;
            int base = 0;
            if (lane == leader) base = (int)atomicAdd(gt_write, (unsigned int)__popc(gm));
            base = __shfl_sync(gm, base, leader);
            out[base + rank] = i;
        }
        if (eq) {
            int rank = __popc(em & ((1u << lane) - 1));
            int leader = __ffs(em) - 1;
            int base = 0;
            if (lane == leader) base = (int)atomicAdd(tie_write, (unsigned int)__popc(em));
            base = __shfl_sync(em, base, leader);
            int slot = base + rank;
            if (slot < ties_needed) out[total_above + slot] = i;
        }
    }
    for (int it = 0; it < ov_iters; ++it) {
        int i4 = gstride + it * gstride + gtid;
        bool in = i4 < nv4;
        float4 f;
        if (in) f = in4[i4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int key = in ? fkey(j == 0 ? f.x : j == 1 ? f.y : j == 2 ? f.z : f.w) : 0u;
            int i = 4 * i4 + j;
            if (final_shift) key >>= final_shift;
            bool gt = in && key > threshold;
            bool eq = in && key == threshold;
            unsigned int gm = __ballot_sync(0xffffffffu, gt);
            unsigned int em = __ballot_sync(0xffffffffu, eq);
            if (gt) {
                int rank = __popc(gm & ((1u << lane) - 1));
                int leader = __ffs(gm) - 1;
                int base = 0;
                if (lane == leader) base = (int)atomicAdd(gt_write, (unsigned int)__popc(gm));
                base = __shfl_sync(gm, base, leader);
                out[base + rank] = i;
            }
            if (eq) {
                int rank = __popc(em & ((1u << lane) - 1));
                int leader = __ffs(em) - 1;
                int base = 0;
                if (lane == leader) base = (int)atomicAdd(tie_write, (unsigned int)__popc(em));
                base = __shfl_sync(em, base, leader);
                int slot = base + rank;
                if (slot < ties_needed) out[total_above + slot] = i;
            }
        }
    }
    if (last) {
        for (int i = tid; i < CNT; i += BLOCK) scratch[i] = 0u;
        if (tid == 0) atomicExch(tail_arrive, 0u);
    }
}

// ---------------------------------------------------------------------------
// Host launcher.
// ---------------------------------------------------------------------------

static unsigned int* g_scratch = nullptr;
static int g_blocks_cap = 0;
static int g_sms = 0;

// Block-size-generic variant of the descending boundary search (topk_mid
// runs 1024 threads; the 512-thread specialization above cannot be reused).
__device__ __forceinline__ void find_boundary_bins_generic(
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

// Single-CTA kernel for 2304 < n <= 16387: one 11-bit histogram pass over
// register-resident keys, then either (a) emit mandatory indices + compact the
// boundary bucket into shared memory and refine its 21 low bits over that tiny
// list, or (b) for oversized buckets, an in-CTA 11/11/10 ladder over the same
// register keys.  Replaces four 8-bit passes with one 11-bit pass + tiny work.
template<int KPT4>
__global__ __launch_bounds__(1024, 1)
void topk_mid(const float* __restrict__ logits, long rstride, int n, int k,
              int* __restrict__ out) {
    logits += (size_t)blockIdx.y * (size_t)rstride;
    out += (size_t)blockIdx.y * (size_t)k;
    const int tid = threadIdx.x;
    __shared__ unsigned int hist[2048];
    __shared__ unsigned int warp_totals[32];
    __shared__ int s_bin;
    __shared__ int s_above;
    __shared__ int s_gt;
    __shared__ int s_tie;
    __shared__ uint2 s_arena[2048];

    const int n4 = n >> 2;
    const int ntail = n - (n4 << 2);
    const float4* input4 = reinterpret_cast<const float4*>(logits);
    unsigned int keys[KPT4 * 4];
#pragma unroll
    for (int j = 0; j < KPT4; ++j) {
        int i4 = tid + (j << 10);
        if (i4 < n4) {
            float4 v = input4[i4];
            keys[j * 4 + 0] = fkey(v.x);
            keys[j * 4 + 1] = fkey(v.y);
            keys[j * 4 + 2] = fkey(v.z);
            keys[j * 4 + 3] = fkey(v.w);
        } else {
            keys[j * 4 + 0] = 0u;
            keys[j * 4 + 1] = 0u;
            keys[j * 4 + 2] = 0u;
            keys[j * 4 + 3] = 0u;
        }
    }
    unsigned int tail_key = 0u;
    if (tid < ntail) tail_key = fkey(logits[(n4 << 2) + tid]);

    for (int i = tid; i < 2048; i += 1024) hist[i] = 0;
    if (tid == 0) { s_gt = 0; s_tie = 0; }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < KPT4 * 4; ++j) {
        int i4 = tid + ((j >> 2) << 10);
        if (i4 < n4) atomicAdd(&hist[keys[j] >> 21], 1u);
    }
    if (tid < ntail) atomicAdd(&hist[tail_key >> 21], 1u);
    __syncthreads();
    find_boundary_bins_generic(hist, 2048, warp_totals, &s_bin, &s_above, k);
    const int boundary = s_bin;
    const int above = s_above;
    const int pop = (int)hist[boundary];
    const int remaining = k - above;
    const unsigned int ub = (unsigned int)boundary;
    __syncthreads();

    if (pop == remaining || pop <= 2048) {
        const bool direct = (pop == remaining);
#pragma unroll
        for (int j = 0; j < KPT4 * 4; ++j) {
            int i4 = tid + ((j >> 2) << 10);
            if (i4 >= n4) continue;
            unsigned int d = keys[j] >> 21;
            int idx = (i4 << 2) + (j & 3);
            if (d > ub) {
                out[atomicAdd(&s_gt, 1)] = idx;
            } else if (d == ub) {
                int s = atomicAdd(&s_tie, 1);
                if (direct) out[above + s] = idx;
                else s_arena[s] = make_uint2(keys[j], (unsigned int)idx);
            }
        }
        if (tid < ntail) {
            unsigned int d = tail_key >> 21;
            int idx = (n4 << 2) + tid;
            if (d > ub) {
                out[atomicAdd(&s_gt, 1)] = idx;
            } else if (d == ub) {
                int s = atomicAdd(&s_tie, 1);
                if (direct) out[above + s] = idx;
                else s_arena[s] = make_uint2(tail_key, (unsigned int)idx);
            }
        }
        if (direct) return;
        __syncthreads();

        // refine the bucket's 21 low key bits over the smem list
        int rem = remaining;
        int mand = 0;
        unsigned int thr;
        int shift;
        for (int i = tid; i < 2048; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < pop; i += 1024)
            atomicAdd(&hist[(s_arena[i].x & 0x1fffffu) >> 10], 1u);
        __syncthreads();
        find_boundary_bins_generic(hist, 2048, warp_totals, &s_bin, &s_above, rem);
        const int b1 = s_bin;
        const int a1 = s_above;
        if (a1 + (int)hist[b1] == rem) {
            thr = (unsigned int)b1;
            shift = 10;
            mand = a1;
            rem -= a1;
        } else {
            rem -= a1;
            __syncthreads();
            for (int i = tid; i < 1024; i += 1024) hist[i] = 0;
            __syncthreads();
            for (int i = tid; i < pop; i += 1024) {
                unsigned int k21 = s_arena[i].x & 0x1fffffu;
                if ((int)(k21 >> 10) == b1)
                    atomicAdd(&hist[k21 & 1023u], 1u);
            }
            __syncthreads();
            find_boundary_bins_generic(hist, 1024, warp_totals, &s_bin, &s_above, rem);
            thr = ((unsigned int)b1 << 10) | (unsigned int)s_bin;
            shift = 0;
            mand = a1 + s_above;
            rem -= s_above;
        }
        if (tid == 0) { s_gt = 0; s_tie = 0; }
        __syncthreads();
        for (int i = tid; i < pop; i += 1024) {
            uint2 p = s_arena[i];
            unsigned int kk = (p.x & 0x1fffffu) >> shift;
            if (kk > thr) {
                out[above + atomicAdd(&s_gt, 1)] = (int)p.y;
            } else if (kk == thr) {
                int s = atomicAdd(&s_tie, 1);
                if (s < rem) out[above + mand + s] = (int)p.y;
            }
        }
        return;
    }

    // oversized bucket: in-CTA 11/11/10 ladder over the register keys
    unsigned int prefix = ub << 21;
    int rem = remaining;
    int total_above = above;
    int consumed = 11;
#pragma unroll
    for (int pass = 1; pass < 3; ++pass) {
        const int shift = pass == 1 ? 10 : 0;
        const int bits = pass == 1 ? 11 : 10;
        const int nb = 1 << bits;
        const unsigned int digit_mask = nb - 1;
        const unsigned int high_mask = 0xffffffffu << (shift + bits);
        for (int i = tid; i < nb; i += 1024) hist[i] = 0;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < KPT4 * 4; ++j) {
            int i4 = tid + ((j >> 2) << 10);
            if (i4 < n4 && (keys[j] & high_mask) == (prefix & high_mask))
                atomicAdd(&hist[(keys[j] >> shift) & digit_mask], 1u);
        }
        if (tid < ntail && (tail_key & high_mask) == (prefix & high_mask))
            atomicAdd(&hist[(tail_key >> shift) & digit_mask], 1u);
        __syncthreads();
        find_boundary_bins_generic(hist, nb, warp_totals, &s_bin, &s_above, rem);
        bool whole_bucket = s_above + (int)hist[s_bin] == rem;
        prefix |= (unsigned int)s_bin << shift;
        total_above += s_above;
        rem -= s_above;
        consumed += bits;
        __syncthreads();
        if (whole_bucket) break;
    }

    const int final_shift = 32 - consumed;
    const unsigned int threshold = final_shift ? (prefix >> final_shift) : prefix;
    if (tid == 0) { s_gt = 0; s_tie = 0; }
    __syncthreads();
    const int ties_needed = rem;
#pragma unroll
    for (int j = 0; j < KPT4 * 4; ++j) {
        int i4 = tid + ((j >> 2) << 10);
        if (i4 >= n4) continue;
        unsigned int key = keys[j];
        if (final_shift) key >>= final_shift;
        int idx = (i4 << 2) + (j & 3);
        if (key > threshold) {
            out[atomicAdd(&s_gt, 1)] = idx;
        } else if (key == threshold) {
            int s = atomicAdd(&s_tie, 1);
            if (s < ties_needed) out[total_above + s] = idx;
        }
    }
    if (tid < ntail) {
        unsigned int key = tail_key;
        if (final_shift) key >>= final_shift;
        int idx = (n4 << 2) + tid;
        if (key > threshold) {
            out[atomicAdd(&s_gt, 1)] = idx;
        } else if (key == threshold) {
            int s = atomicAdd(&s_tie, 1);
            if (s < ties_needed) out[total_above + s] = idx;
        }
    }
}


void topk_launch(const float* logits, int n, int k, int* out,
                 cudaStream_t stream) {
    if (n - k == 3 && n <= 1536) {
        bottom3_kernel<<<1, 512, 0, stream>>>(logits, 0L, n, out);
        return;
    }
    if (n <= 1536) {
        topk_small<2><<<1, 768, 0, stream>>>(logits, 0L, n, k, out);
        return;
    }
    if (n <= 2304) {
        topk_small<3><<<1, 768, 0, stream>>>(logits, 0L, n, k, out);
        return;
    }
    // topk_mid pays off when the selection boundary sits in the distribution
    // tail (small boundary bucket); for k close to n the bucket is bulk-dense
    // and the byte-radix rungs win.
    const bool tail_sel = 4 * k <= n;
    // topk_mid<1> rung (n<=4099) measured a 5-11% regression vs the byte-radix
    // topk_small<6> on the real grid; gated out (see R3_LEDGER).
    if (n <= 4608) {
        topk_small<6><<<1, 768, 0, stream>>>(logits, 0L, n, k, out);
        return;
    }
    if (tail_sel && n <= 8195) {
        topk_mid<2><<<1, 1024, 0, stream>>>(logits, 0L, n, k, out);
        return;
    }
    if (n <= 8448) {
        topk_small<11><<<1, 768, 0, stream>>>(logits, 0L, n, k, out);
        return;
    }
    if (tail_sel && n <= 12291) {
        topk_mid<3><<<1, 1024, 0, stream>>>(logits, 0L, n, k, out);
        return;
    }
    if (tail_sel && n <= 16387) {
        topk_mid<4><<<1, 1024, 0, stream>>>(logits, 0L, n, k, out);
        return;
    }
    if (n <= 16896) {
        topk_small<17><<<1, 1024, 0, stream>>>(logits, 0L, n, k, out);
        return;
    }

    if (!g_scratch) {
        cudaMalloc(&g_scratch, SCRATCH_WORDS * sizeof(unsigned int));
        cudaMemset(g_scratch, 0, SCRATCH_WORDS * sizeof(unsigned int));
    }
    if (!g_blocks_cap) {
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_fast<1>, BLOCK, 0);
        if (active < 1) active = 1;
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_blocks_cap = active * g_sms;
    }
    int blocks = (n + 2047) / 2048;
    if (blocks < g_sms) blocks = g_sms;
    if (blocks > g_blocks_cap) blocks = g_blocks_cap;
    // Globally-monotonic generation: barrier sense tokens never collide
    // across launches, so the persistent arrive/release words need no reset.
    static unsigned int g_gen = 0;
    ++g_gen;
    topk_fast<1><<<blocks, BLOCK, 0, stream>>>(logits, 0L, n, k, g_scratch, out, g_gen, blocks);
}


// ---------------------------------------------------------------------------
// BS>1 extension (R3_LEDGER "BS>1 extension design analysis", steps A + B).
//   A: small/mid-n single-CTA tiers batch via grid.y (one launch, row per
//      blockIdx.y).
//   B: large-n topk_fast batches via row teams -- team = ceil(n/2048) CTAs
//      per row (register-resident constraint), per-row scratch slices, the
//      per-row grid barrier only spans that row's team.  Rows that fit the
//      co-residency cap run in ONE launch (single wave); larger BS is chunked
//      into ceil(BS/rows_per_wave) launches.  Slices are per-ROW (never
//      reused within a launch), so the fence-less L1 safety argument is
//      preserved: reuse only happens across launch boundaries.
// ---------------------------------------------------------------------------
static unsigned int* g_scratch_bs = nullptr;
static int g_scratch_bs_rows = 0;
static unsigned int g_gen_bs = 0;   // separate buffer => independent senses

void ext_caps(int* team_out, int* cap_out, int n) {
    if (!g_blocks_cap) {
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_fast<1>, BLOCK, 0);
        if (active < 1) active = 1;
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_blocks_cap = active * g_sms;
    }
    *team_out = (n + 2047) / 2048;
    *cap_out = g_blocks_cap;
}

void topk_launch_batched(const float* logits, long W, int n, int k, int* out,
                         int BS, cudaStream_t stream) {
    const dim3 gy(1, (unsigned int)BS);
    if (n - k == 3 && n <= 1536) {
        bottom3_kernel<<<gy, 512, 0, stream>>>(logits, W, n, out);
        return;
    }
    if (n <= 1536) {
        topk_small<2><<<gy, 768, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (n <= 2304) {
        topk_small<3><<<gy, 768, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    const bool tail_sel = 4 * k <= n;
    if (n <= 4608) {
        topk_small<6><<<gy, 768, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (tail_sel && n <= 8195) {
        topk_mid<2><<<gy, 1024, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (n <= 8448) {
        topk_small<11><<<gy, 768, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (tail_sel && n <= 12291) {
        topk_mid<3><<<gy, 1024, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (tail_sel && n <= 16387) {
        topk_mid<4><<<gy, 1024, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (n <= 16896) {
        topk_small<17><<<gy, 1024, 0, stream>>>(logits, W, n, k, out);
        return;
    }

    int team, cap;
    ext_caps(&team, &cap, n);
    if (team > cap) {
        // row wider than one wave: keep the shipped single-row path per row
        for (int r = 0; r < BS; ++r)
            topk_launch(logits + (size_t)r * (size_t)W, n, k,
                        out + (size_t)r * (size_t)k, stream);
        return;
    }
    if (g_scratch_bs_rows < BS) {
        if (g_scratch_bs) cudaFree(g_scratch_bs);
        cudaMalloc(&g_scratch_bs,
                   (size_t)BS * SCRATCH_WORDS * sizeof(unsigned int));
        cudaMemset(g_scratch_bs, 0,
                   (size_t)BS * SCRATCH_WORDS * sizeof(unsigned int));
        g_scratch_bs_rows = BS;
    }
    int rows_per_wave = cap / team;
    for (int r0 = 0; r0 < BS; r0 += rows_per_wave) {
        const int rows = BS - r0 < rows_per_wave ? BS - r0 : rows_per_wave;
        ++g_gen_bs;
        topk_fast<1><<<rows * team, BLOCK, 0, stream>>>(
            logits + (size_t)r0 * (size_t)W, W, n, k,
            g_scratch_bs + (size_t)r0 * (size_t)SCRATCH_WORDS,
            out + (size_t)r0 * (size_t)k, g_gen_bs, team);
    }
}

// ---------------------------------------------------------------------------
// Register-diet variants (R3_LEDGER BS>1 ext "next lever"): MINB in {1..4}
// via __launch_bounds__(BLOCK, MINB). Per-variant occupancy caches; smem
// carveout raised to MAX once per variant (static smem 40KB/block would
// otherwise cap active at the default carveout).
// ---------------------------------------------------------------------------
static int g_cap_v[5] = {0, 0, 0, 0, 0};

template<int MINB>
static void diet_caps(int* team_out, int* cap_out, int n) {
    if (!g_cap_v[MINB]) {
        cudaFuncSetAttribute(topk_fast<MINB>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_fast<MINB>, BLOCK, 0);
        if (active < 1) active = 1;
        if (!g_sms)
            cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_cap_v[MINB] = active * g_sms;
    }
    *team_out = (n + 2047) / 2048;
    *cap_out = g_cap_v[MINB];
}

template<int MINB>
static void launch_fast_teams(const float* logits, long W, int n, int k,
                              int* out, int BS, cudaStream_t stream) {
    int team, cap;
    diet_caps<MINB>(&team, &cap, n);
    if (team > cap) {
        for (int r = 0; r < BS; ++r)
            topk_launch(logits + (size_t)r * (size_t)W, n, k,
                        out + (size_t)r * (size_t)k, stream);
        return;
    }
    if (g_scratch_bs_rows < BS) {
        if (g_scratch_bs) cudaFree(g_scratch_bs);
        cudaMalloc(&g_scratch_bs,
                   (size_t)BS * SCRATCH_WORDS * sizeof(unsigned int));
        cudaMemset(g_scratch_bs, 0,
                   (size_t)BS * SCRATCH_WORDS * sizeof(unsigned int));
        g_scratch_bs_rows = BS;
    }
    const int rows_per_wave = cap / team;
    for (int r0 = 0; r0 < BS; r0 += rows_per_wave) {
        const int rows = BS - r0 < rows_per_wave ? BS - r0 : rows_per_wave;
        ++g_gen_bs;
        topk_fast<MINB><<<rows * team, BLOCK, 0, stream>>>(
            logits + (size_t)r0 * (size_t)W, W, n, k,
            g_scratch_bs + (size_t)r0 * (size_t)SCRATCH_WORDS,
            out + (size_t)r0 * (size_t)k, g_gen_bs, team);
    }
}

void launch_fast_teams_v(const float* logits, long W, int n, int k, int* out,
                         int BS, int minb, cudaStream_t stream) {
    switch (minb) {
        case 2: launch_fast_teams<2>(logits, W, n, k, out, BS, stream); break;
        case 3: launch_fast_teams<3>(logits, W, n, k, out, BS, stream); break;
        case 4: launch_fast_teams<4>(logits, W, n, k, out, BS, stream); break;
        default: launch_fast_teams<1>(logits, W, n, k, out, BS, stream); break;
    }
}

// numRegs, staticSmemBytes, localBytes, active_default, active_maxcarveout
template<int MINB>
static void fast_stats_t(int out5[5]) {
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, topk_fast<MINB>);
    out5[0] = a.numRegs;
    out5[1] = (int)a.sharedSizeBytes;
    out5[2] = (int)a.localSizeBytes;
    int act = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act, topk_fast<MINB>,
                                                  BLOCK, 0);
    out5[3] = act;
    cudaFuncSetAttribute(topk_fast<MINB>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    act = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act, topk_fast<MINB>,
                                                  BLOCK, 0);
    out5[4] = act;
}

void fast_stats_v(int minb, int out5[5]) {
    switch (minb) {
        case 2: fast_stats_t<2>(out5); break;
        case 3: fast_stats_t<3>(out5); break;
        case 4: fast_stats_t<4>(out5); break;
        default: fast_stats_t<1>(out5); break;
    }
}

// __ldcg (L2-only) twin of find_boundary_bins for the persistent-queue path:
// per-team scratch slices are reused across row iterations WITHIN one launch,
// so the "L1 invalidated at launch boundary" leg of the fence-less safety
// argument does not apply -- post-barrier global-histogram loads must bypass
// L1.
__device__ __forceinline__ void find_boundary_bins_cg(
        const unsigned int* __restrict__ hist, int nb,
        unsigned int* warp_totals, int* s_bin, int* s_above,
        int remaining) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int bins_per_thread = nb >> 9;   // nb / BLOCK, BLOCK == 512
    const int base = tid * bins_per_thread;
    unsigned int b[4];
    unsigned int local = 0;
    if (bins_per_thread == 4) {
        uint4 v = __ldcg(reinterpret_cast<const uint4*>(hist + base));
        b[0] = v.x; b[1] = v.y; b[2] = v.z; b[3] = v.w;
        local = v.x + v.y + v.z + v.w;
    } else {
        uint2 v = __ldcg(reinterpret_cast<const uint2*>(hist + base));
        b[0] = v.x; b[1] = v.y; b[2] = 0u; b[3] = 0u;
        local = v.x + v.y;
    }
    unsigned int suffix = local;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        unsigned int v = __shfl_down_sync(0xffffffffu, suffix, off);
        if (lane + off < 32) suffix += v;
    }
    if (lane == 0) warp_totals[warp] = suffix;
    __syncthreads();
    unsigned int higher_warps = 0;
#pragma unroll
    for (int w = 0; w < (BLOCK >> 5); ++w)
        if (w > warp) higher_warps += warp_totals[w];
    unsigned int higher = suffix - local + higher_warps;
    if ((int)higher < remaining && (int)(higher + local) >= remaining) {
        unsigned int cumulative = higher;
        int boundary = base;
        int above = (int)higher;
#pragma unroll
        for (int j = 3; j >= 0; --j) {
            if (j >= bins_per_thread) continue;
            unsigned int next = cumulative + b[j];
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

// B' persistent-queue kernel: one launch, nteams teams loop rows round-robin.
template<int MINB>
__global__ __launch_bounds__(BLOCK, MINB)
void topk_fast_pq(const float* __restrict__ logits, long rstride, int n, int k,
                  unsigned int* __restrict__ scratch, int* __restrict__ out,
                  unsigned int gen, int team, int BS) {
    const int nteams = gridDim.x / team;
    const int team_id = blockIdx.x / team;
    const int bx = blockIdx.x - team_id * team;
    scratch += (size_t)team_id * (size_t)SCRATCH_WORDS;
    const int gridsz = team;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int gtid = bx * BLOCK + tid;
    const int gstride = gridsz * BLOCK;
    unsigned int* hist0_g = scratch + HIST0;
    unsigned int* gt_write = scratch + CNT + 0;
    unsigned int* tie_write = scratch + CNT + 1;
    unsigned int* g_arrive = scratch + CNT + 2;
    unsigned int* g_release = scratch + CNT + 3;
    unsigned int* tail_arrive = scratch + CNT + 4;
    uint2* tiebuf = reinterpret_cast<uint2*>(scratch + TIEBUF);

    // shared: 2048-word histogram, reused after the first barrier as the
    // per-block compaction staging area (gt indices + tie pairs).
    __shared__ __align__(16) unsigned char smem_raw[2048 * 4 + CAP * 8];
    unsigned int* sh_hist = reinterpret_cast<unsigned int*>(smem_raw);
    int* st_gt = reinterpret_cast<int*>(smem_raw);
    uint2* st_tie = reinterpret_cast<uint2*>(smem_raw + 2048 * 4);

    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin;
    __shared__ int s_above;
    __shared__ int s_ngt, s_ntie, s_gtbase, s_tiebase, s_cnt, s_flag;
    __shared__ unsigned int s_eqk[32];
    __shared__ int s_eqv[32];


    for (int row = team_id, iter = 0; row < BS; row += nteams, ++iter) {
        const float* lgr = logits + (size_t)row * (size_t)rstride;
        int* outr = out + (size_t)row * (size_t)k;
        const unsigned int sense0 = gen * 8192u + (unsigned int)iter * 8u;
        [&]() {
            // ---- register-cached keys: one float4 + one tail scalar per thread ----
            const int n4 = n & ~3;
            const int nv4 = n4 >> 2;
            const float4* in4 = reinterpret_cast<const float4*>(lgr);
            unsigned int kv[4];
            const bool vok = gtid < nv4;
            if (vok) {
                float4 f = in4[gtid];
                kv[0] = fkey(f.x); kv[1] = fkey(f.y);
                kv[2] = fkey(f.z); kv[3] = fkey(f.w);
            } else {
                kv[0] = kv[1] = kv[2] = kv[3] = 0u;
            }
            const int ptail = n4 + gtid;
            const bool tok = ptail < n;
            const unsigned int ktail = tok ? fkey(lgr[ptail]) : 0u;
            // overflow region (n beyond register capacity); never taken for the
            // benchmark shapes but keeps the kernel correct for any n.
            const int ov_iters = nv4 > gstride
                ? (nv4 - gstride + gstride - 1) / gstride : 0;
        
            if (gtid == 0) { *gt_write = 0u; *tie_write = 0u; }
        
            // ---- pass 0: 11-bit MSB histogram ----
            for (int i = tid; i < 2048; i += BLOCK) sh_hist[i] = 0u;
            __syncthreads();
            if (vok) {
                atomicAdd(&sh_hist[kv[0] >> 21], 1u);
                atomicAdd(&sh_hist[kv[1] >> 21], 1u);
                atomicAdd(&sh_hist[kv[2] >> 21], 1u);
                atomicAdd(&sh_hist[kv[3] >> 21], 1u);
            }
            if (tok) atomicAdd(&sh_hist[ktail >> 21], 1u);
            for (int it = 0; it < ov_iters; ++it) {
                int i4 = gstride + it * gstride + gtid;
                if (i4 < nv4) {
                    float4 f = in4[i4];
                    atomicAdd(&sh_hist[fkey(f.x) >> 21], 1u);
                    atomicAdd(&sh_hist[fkey(f.y) >> 21], 1u);
                    atomicAdd(&sh_hist[fkey(f.z) >> 21], 1u);
                    atomicAdd(&sh_hist[fkey(f.w) >> 21], 1u);
                }
            }
            __syncthreads();
            // merge as packed 64-bit adds: per-bin totals < 2^20, no carry between
            // the two 32-bit halves; halves the global atomic count.
            {
                const uint2* sh2 = reinterpret_cast<const uint2*>(sh_hist);
                unsigned long long* g2 = reinterpret_cast<unsigned long long*>(hist0_g);
                for (int i = tid; i < 1024; i += BLOCK) {
                    uint2 c = sh2[i];
                    if (c.x | c.y)
                        atomicAdd(&g2[i], ((unsigned long long)c.y << 32) | c.x);
                }
            }
            global_barrier(g_arrive, g_release, gridsz, sense0);
        
            find_boundary_bins_cg(hist0_g, 2048, warp_totals, &s_bin, &s_above, k);
            const int b0 = s_bin;
            const int above0 = s_above;
            const int T = (int)__ldcg(&hist0_g[b0]);
            const int R = k - above0;
            const bool whole = T == R;
        
            if (whole || T <= CAP) {
                // ---- fast finishes: staged compaction, then arrive-and-exit ----
                if (tid == 0) { s_ngt = 0; s_ntie = 0; }
                __syncthreads();
        #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (vok) {
                        unsigned int d = kv[j] >> 21;
                        if ((int)d > b0) {
                            st_gt[atomicAdd(&s_ngt, 1)] = 4 * gtid + j;
                        } else if ((int)d == b0) {
                            st_tie[atomicAdd(&s_ntie, 1)] =
                                make_uint2(kv[j], (unsigned int)(4 * gtid + j));
                        }
                    }
                }
                if (tok) {
                    unsigned int d = ktail >> 21;
                    if ((int)d > b0) st_gt[atomicAdd(&s_ngt, 1)] = ptail;
                    else if ((int)d == b0)
                        st_tie[atomicAdd(&s_ntie, 1)] =
                            make_uint2(ktail, (unsigned int)ptail);
                }
                for (int it = 0; it < ov_iters; ++it) {
                    int i4 = gstride + it * gstride + gtid;
                    if (i4 < nv4) {
                        float4 f = in4[i4];
                        unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            unsigned int d = kk[j] >> 21;
                            if ((int)d > b0) st_gt[atomicAdd(&s_ngt, 1)] = 4 * i4 + j;
                            else if ((int)d == b0)
                                st_tie[atomicAdd(&s_ntie, 1)] =
                                    make_uint2(kk[j], (unsigned int)(4 * i4 + j));
                        }
                    }
                }
                __syncthreads();
                if (tid == 0) {
                    s_gtbase = s_ngt ? (int)atomicAdd(gt_write, (unsigned int)s_ngt) : 0;
                    s_tiebase = s_ntie ? (int)atomicAdd(tie_write, (unsigned int)s_ntie) : 0;
                }
                __syncthreads();
                for (int i = tid; i < s_ngt; i += BLOCK) outr[s_gtbase + i] = st_gt[i];
                if (whole) {
                    for (int i = tid; i < s_ntie; i += BLOCK)
                        outr[above0 + s_tiebase + i] = (int)st_tie[i].y;
                } else {
                    // L2-direct atomic stores: visible to the last arriver through the
                    // same atomic-chain contract the spinning barrier relies on.
                    unsigned long long* tb64 =
                        reinterpret_cast<unsigned long long*>(tiebuf + s_tiebase);
                    for (int i = tid; i < s_ntie; i += BLOCK) {
                        uint2 e = st_tie[i];
                        atomicExch(&tb64[i], ((unsigned long long)e.y << 32) | e.x);
                    }
                }
                __syncthreads();
                if (tid == 0) {
                    unsigned int t = atomicAdd(tail_arrive, 1u);
                    s_flag = (t + 1u == (unsigned int)gridsz) ? 1 : 0;
                }
                __syncthreads();
                if (!s_flag) return;
                // ---- last arriver: exclusive owner of scratch from here ----
                if (tid == 0) atomicExch(tail_arrive, 0u);
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
        
                if (!whole) {
                    // refine low 21 bits of the T (<= CAP) boundary-bucket pairs
                    unsigned int tk[PT];
                    int tv[PT];
        #pragma unroll
                    for (int j = 0; j < PT; ++j) {
                        if (j * BLOCK >= T) break;
                        int p = tid + j * BLOCK;
                        if (p < T) {
                            uint2 e = __ldcg(&tiebuf[p]);
                            tk[j] = e.x; tv[j] = (int)e.y;
                        } else { tk[j] = 0u; tv[j] = 0; }
                    }
                    __syncthreads();
                    // pass A: bits 20..10 (2048 bins)
                    for (int i = tid; i < 2048; i += BLOCK) sh_hist[i] = 0u;
                    __syncthreads();
        #pragma unroll
                    for (int j = 0; j < PT; ++j) {
                        if (j * BLOCK >= T) break;
                        if (tid + j * BLOCK < T)
                            atomicAdd(&sh_hist[(tk[j] >> 10) & 2047u], 1u);
                    }
                    __syncthreads();
                    find_boundary_bins(sh_hist, 2048, warp_totals, &s_bin, &s_above, R);
                    const int bA = s_bin;
                    const int aboveA = s_above;
                    const int TA = (int)sh_hist[bA];
                    const bool wholeA = aboveA + TA == R;
                    if (tid == 0) { s_cnt = 0; s_ntie = 0; }
                    __syncthreads();
                    // combined collect: pass-A winners straight to out; boundary-bin
                    // ties gathered into a 32-slot shared list for the warp finisher.
        #pragma unroll
                    for (int j = 0; j < PT; ++j) {
                        if (j * BLOCK >= T) break;
                        bool active = tid + j * BLOCK < T;
                        int binA = active ? (int)((tk[j] >> 10) & 2047u) : -1;
                        bool p = wholeA ? (binA >= bA) : (binA > bA);
                        unsigned int m = __ballot_sync(0xffffffffu, p);
                        if (p) {
                            int r = __popc(m & ((1u << lane) - 1));
                            int leader = __ffs(m) - 1;
                            int base = 0;
                            if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                            base = __shfl_sync(m, base, leader);
                            outr[above0 + base + r] = tv[j];
                        }
                        if (!wholeA && binA == bA) {
                            int q = atomicAdd(&s_ntie, 1);
                            if (q < 32) { s_eqk[q] = tk[j]; s_eqv[q] = tv[j]; }
                        }
                    }
                    __syncthreads();
                    if (!wholeA) {
                        const int R2 = R - aboveA;
                        if (TA <= 32) {
                            // warp-serial top-R2 of the <=32 boundary ties: R2 rounds
                            // of max-reduction over the low 10 bits (full-key ties at
                            // the k-th value may resolve either way).
                            if (tid < 32) {
                                unsigned int score =
                                    tid < TA ? ((s_eqk[tid] & 1023u) + 1u) : 0u;
                                for (int r = 0; r < R2; ++r) {
                                    unsigned int bs = score;
                                    int bl = tid;
        #pragma unroll
                                    for (int off = 16; off; off >>= 1) {
                                        unsigned int os =
                                            __shfl_down_sync(0xffffffffu, bs, off);
                                        int ol = __shfl_down_sync(0xffffffffu, bl, off);
                                        if (tid + off < 32 && os > bs) { bs = os; bl = ol; }
                                    }
                                    int winner = __shfl_sync(0xffffffffu, bl, 0);
                                    if (tid == winner) {
                                        outr[above0 + aboveA + r] = s_eqv[tid];
                                        score = 0u;
                                    }
                                }
                            }
                        } else {
                            // rare big-tie fallback: pass B over bits 9..0 (1024 bins)
                            __syncthreads();
                            for (int i = tid; i < 1024; i += BLOCK) sh_hist[i] = 0u;
                            __syncthreads();
        #pragma unroll
                            for (int j = 0; j < PT; ++j) {
                                if (j * BLOCK >= T) break;
                                if ((tid + j * BLOCK < T) &&
                                    (int)((tk[j] >> 10) & 2047u) == bA)
                                    atomicAdd(&sh_hist[tk[j] & 1023u], 1u);
                            }
                            __syncthreads();
                            find_boundary_bins(sh_hist, 1024, warp_totals,
                                               &s_bin, &s_above, R2);
                            const int bB = s_bin;
        #pragma unroll
                            for (int j = 0; j < PT; ++j) {
                                if (j * BLOCK >= T) break;
                                bool inA = (tid + j * BLOCK < T) &&
                                           (int)((tk[j] >> 10) & 2047u) == bA;
                                bool p = inA && (int)(tk[j] & 1023u) > bB;
                                unsigned int m = __ballot_sync(0xffffffffu, p);
                                if (p) {
                                    int r = __popc(m & ((1u << lane) - 1));
                                    int leader = __ffs(m) - 1;
                                    int base = 0;
                                    if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                                    base = __shfl_sync(m, base, leader);
                                    outr[above0 + base + r] = tv[j];
                                }
                            }
                            __syncthreads();
                            // full-key ties at the exact k-th value: any subset fills
        #pragma unroll
                            for (int j = 0; j < PT; ++j) {
                                if (j * BLOCK >= T) break;
                                bool inA = (tid + j * BLOCK < T) &&
                                           (int)((tk[j] >> 10) & 2047u) == bA;
                                bool p = inA && (int)(tk[j] & 1023u) == bB;
                                unsigned int m = __ballot_sync(0xffffffffu, p);
                                if (p) {
                                    int r = __popc(m & ((1u << lane) - 1));
                                    int leader = __ffs(m) - 1;
                                    int base = 0;
                                    if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                                    base = __shfl_sync(m, base, leader);
                                    int slot = base + r;
                                    if (slot < R) outr[above0 + slot] = tv[j];
                                }
                            }
                        }
                    }
                }
                // re-zero pass-0 histogram for the next launch (only block touching
                // scratch at this point; every other block has already arrived).
                for (int i = tid; i < 2048; i += BLOCK) hist0_g[i] = 0u;
                return;
            }
        
            // ---- fallback: boundary bucket too large; classic 11/11/10 ladder ----
            int remaining = R;
            int total_above = above0;
            unsigned int prefix = ((unsigned int)b0) << 21;
            int consumed = 11;
        #pragma unroll
            for (int pass = 1; pass < 3; ++pass) {
                const int shift = pass == 1 ? 10 : 0;
                const int bits = pass == 1 ? 11 : 10;
                const int nb = 1 << bits;
                const unsigned int digit_mask = (unsigned int)(nb - 1);
                const unsigned int high_mask = 0xffffffffu << (shift + bits);
                for (int i = tid; i < nb; i += BLOCK) sh_hist[i] = 0u;
                __syncthreads();
        #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (vok && (kv[j] & high_mask) == (prefix & high_mask))
                        atomicAdd(&sh_hist[(kv[j] >> shift) & digit_mask], 1u);
                if (tok && (ktail & high_mask) == (prefix & high_mask))
                    atomicAdd(&sh_hist[(ktail >> shift) & digit_mask], 1u);
                for (int it = 0; it < ov_iters; ++it) {
                    int i4 = gstride + it * gstride + gtid;
                    if (i4 < nv4) {
                        float4 f = in4[i4];
                        unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            if ((kk[j] & high_mask) == (prefix & high_mask))
                                atomicAdd(&sh_hist[(kk[j] >> shift) & digit_mask], 1u);
                    }
                }
                __syncthreads();
                unsigned int* merged = scratch + (pass == 1 ? HIST1 : HIST2);
                for (int i = tid; i < nb; i += BLOCK) {
                    unsigned int c = sh_hist[i];
                    if (c) atomicAdd(&merged[i], c);
                }
                global_barrier(g_arrive, g_release, gridsz,
                               sense0 + (unsigned int)pass);
                find_boundary_bins_cg(merged, nb, warp_totals, &s_bin,
                                      &s_above, remaining);
                bool whole_bucket = s_above + (int)__ldcg(&merged[s_bin]) == remaining;
                prefix |= ((unsigned int)s_bin) << shift;
                total_above += s_above;
                remaining -= s_above;
                consumed += bits;
                if (whole_bucket) break;
            }
        
            // non-spinning rendezvous: after this point every block has finished
            // reading the merged histograms, so the last arriver may re-zero them.
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                unsigned int t = atomicAdd(tail_arrive, 1u);
                s_flag = (t + 1u == (unsigned int)gridsz) ? 1 : 0;
            }
            __syncthreads();
            const bool last = s_flag != 0;
        
            const int final_shift = 32 - consumed;
            const unsigned int threshold = final_shift ? (prefix >> final_shift) : prefix;
            const int ties_needed = remaining;
        #pragma unroll
            for (int j = 0; j < 5; ++j) {
                bool valid = j < 4 ? vok : tok;
                unsigned int key = j < 4 ? kv[j] : ktail;
                int i = j < 4 ? 4 * gtid + j : ptail;
                if (final_shift) key >>= final_shift;
                bool gt = valid && key > threshold;
                bool eq = valid && key == threshold;
                unsigned int gm = __ballot_sync(0xffffffffu, gt);
                unsigned int em = __ballot_sync(0xffffffffu, eq);
                if (gt) {
                    int rank = __popc(gm & ((1u << lane) - 1));
                    int leader = __ffs(gm) - 1;
                    int base = 0;
                    if (lane == leader) base = (int)atomicAdd(gt_write, (unsigned int)__popc(gm));
                    base = __shfl_sync(gm, base, leader);
                    outr[base + rank] = i;
                }
                if (eq) {
                    int rank = __popc(em & ((1u << lane) - 1));
                    int leader = __ffs(em) - 1;
                    int base = 0;
                    if (lane == leader) base = (int)atomicAdd(tie_write, (unsigned int)__popc(em));
                    base = __shfl_sync(em, base, leader);
                    int slot = base + rank;
                    if (slot < ties_needed) outr[total_above + slot] = i;
                }
            }
            for (int it = 0; it < ov_iters; ++it) {
                int i4 = gstride + it * gstride + gtid;
                bool in = i4 < nv4;
                float4 f;
                if (in) f = in4[i4];
        #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    unsigned int key = in ? fkey(j == 0 ? f.x : j == 1 ? f.y : j == 2 ? f.z : f.w) : 0u;
                    int i = 4 * i4 + j;
                    if (final_shift) key >>= final_shift;
                    bool gt = in && key > threshold;
                    bool eq = in && key == threshold;
                    unsigned int gm = __ballot_sync(0xffffffffu, gt);
                    unsigned int em = __ballot_sync(0xffffffffu, eq);
                    if (gt) {
                        int rank = __popc(gm & ((1u << lane) - 1));
                        int leader = __ffs(gm) - 1;
                        int base = 0;
                        if (lane == leader) base = (int)atomicAdd(gt_write, (unsigned int)__popc(gm));
                        base = __shfl_sync(gm, base, leader);
                        outr[base + rank] = i;
                    }
                    if (eq) {
                        int rank = __popc(em & ((1u << lane) - 1));
                        int leader = __ffs(em) - 1;
                        int base = 0;
                        if (lane == leader) base = (int)atomicAdd(tie_write, (unsigned int)__popc(em));
                        base = __shfl_sync(em, base, leader);
                        int slot = base + rank;
                        if (slot < ties_needed) outr[total_above + slot] = i;
                    }
                }
            }
            if (last) {
                for (int i = tid; i < CNT; i += BLOCK) scratch[i] = 0u;
                if (tid == 0) atomicExch(tail_arrive, 0u);
            }
        
        }();
        // stand-in for the launch boundary: the last arriver's scratch
        // re-zero must be complete (and ordered) before this team's next row
        // starts merging into the same slice. Skipped when this team has no
        // further row (kernel exit is the boundary then).
        if (row + nteams < BS)
            global_barrier(g_arrive, g_release, gridsz, sense0 + 7u);
    }
}

static unsigned int* g_scratch_pq = nullptr;
static int g_pq_teams = 0;
static unsigned int g_gen_pq = 0;
static int g_cap_pq[5] = {0, 0, 0, 0, 0};

template<int MINB>
static void pq_caps(int* team_out, int* cap_out, int n) {
    if (!g_cap_pq[MINB]) {
        cudaFuncSetAttribute(topk_fast_pq<MINB>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_fast_pq<MINB>, BLOCK, 0);
        if (active < 1) active = 1;
        if (!g_sms)
            cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_cap_pq[MINB] = active * g_sms;
    }
    *team_out = (n + 2047) / 2048;
    *cap_out = g_cap_pq[MINB];
}

template<int MINB>
static void launch_fast_pq(const float* logits, long W, int n, int k,
                           int* out, int BS, cudaStream_t stream) {
    int team, cap;
    pq_caps<MINB>(&team, &cap, n);
    if (team > cap || BS > 8192) {   // iter bound 1023 needs BS/nteams small
        for (int r = 0; r < BS; ++r)
            topk_launch(logits + (size_t)r * (size_t)W, n, k,
                        out + (size_t)r * (size_t)k, stream);
        return;
    }
    int nteams = cap / team;
    if (nteams > BS) nteams = BS;
    if (g_pq_teams < nteams) {
        if (g_scratch_pq) cudaFree(g_scratch_pq);
        cudaMalloc(&g_scratch_pq,
                   (size_t)nteams * SCRATCH_WORDS * sizeof(unsigned int));
        cudaMemset(g_scratch_pq, 0,
                   (size_t)nteams * SCRATCH_WORDS * sizeof(unsigned int));
        g_pq_teams = nteams;
    }
    ++g_gen_pq;
    topk_fast_pq<MINB><<<nteams * team, BLOCK, 0, stream>>>(
        logits, W, n, k, g_scratch_pq, out, g_gen_pq, team, BS);
}

void launch_fast_pq_v(const float* logits, long W, int n, int k, int* out,
                      int BS, int minb, cudaStream_t stream) {
    switch (minb) {
        case 2: launch_fast_pq<2>(logits, W, n, k, out, BS, stream); break;
        case 3: launch_fast_pq<3>(logits, W, n, k, out, BS, stream); break;
        case 4: launch_fast_pq<4>(logits, W, n, k, out, BS, stream); break;
        default: launch_fast_pq<1>(logits, W, n, k, out, BS, stream); break;
    }
}

template<int MINB>
static void pq_stats_t(int out5[5]) {
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, topk_fast_pq<MINB>);
    out5[0] = a.numRegs;
    out5[1] = (int)a.sharedSizeBytes;
    out5[2] = (int)a.localSizeBytes;
    int act = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act, topk_fast_pq<MINB>,
                                                  BLOCK, 0);
    out5[3] = act;
    cudaFuncSetAttribute(topk_fast_pq<MINB>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    act = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act, topk_fast_pq<MINB>,
                                                  BLOCK, 0);
    out5[4] = act;
}

void pq_stats_v(int minb, int out5[5]) {
    switch (minb) {
        case 2: pq_stats_t<2>(out5); break;
        case 3: pq_stats_t<3>(out5); break;
        case 4: pq_stats_t<4>(out5); break;
        default: pq_stats_t<1>(out5); break;
    }
}

// ---------------------------------------------------------------------------
// D1 throughput arm (R3_LEDGER "BS>1 extension final picture" open item):
// barrier-free 3-kernel pipeline for the large-n high-BS regime.
// Per-row scratch: hist (2048 u32) in g_tp_hist; [gt_write, tie_write,
// pad, pad] + CAP uint2 tiebuf in g_tp_sc (16B-aligned slices).
// ---------------------------------------------------------------------------
#define TP_SCROW (4 + 2 * CAP)

__global__ __launch_bounds__(512, 1)
void tp_hist(const float* __restrict__ logits, long rstride, int n, int C,
             unsigned int* __restrict__ hist) {
    const int row = blockIdx.y;
    const int slice = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    unsigned int* hist_r = hist + (size_t)row * 2048;

    __shared__ unsigned int sh[2048];
    for (int i = tid; i < 2048; i += 512) sh[i] = 0u;
    __syncthreads();

    const int nv4 = n >> 2;
    const int q = nv4 / C, r = nv4 % C;
    const int beg = slice * q + min(slice, r);
    const int end = beg + q + (slice < r ? 1 : 0);
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
    for (int i = beg + tid; i < end; i += 512) {
        float4 f = __ldcs(in4 + i);
        atomicAdd(&sh[fkey(f.x) >> 21], 1u);
        atomicAdd(&sh[fkey(f.y) >> 21], 1u);
        atomicAdd(&sh[fkey(f.z) >> 21], 1u);
        atomicAdd(&sh[fkey(f.w) >> 21], 1u);
    }
    if (slice == C - 1) {
        for (int i = (nv4 << 2) + tid; i < n; i += 512)
            atomicAdd(&sh[fkey(__ldcs(lgr + i)) >> 21], 1u);
    }
    __syncthreads();
    const uint2* sh2 = reinterpret_cast<const uint2*>(sh);
    unsigned long long* g2 = reinterpret_cast<unsigned long long*>(hist_r);
    for (int i = tid; i < 1024; i += 512) {
        uint2 c = sh2[i];
        if (c.x | c.y)
            atomicAdd(&g2[i], ((unsigned long long)c.y << 32) | c.x);
    }
}

__global__ __launch_bounds__(512, 1)
void tp_collect(const float* __restrict__ logits, long rstride, int n, int k,
                int C, const unsigned int* __restrict__ hist,
                unsigned int* __restrict__ sc, int* __restrict__ out) {
    const int row = blockIdx.y;
    const int slice = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    const unsigned int* hist_r = hist + (size_t)row * 2048;
    unsigned int* sc_r = sc + (size_t)row * TP_SCROW;
    unsigned int* gt_write = sc_r + 0;
    unsigned int* tie_write = sc_r + 1;
    uint2* tiebuf = reinterpret_cast<uint2*>(sc_r + 4);
    int* outr = out + (size_t)row * (size_t)k;

    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin, s_above;
    find_boundary_bins(hist_r, 2048, warp_totals, &s_bin, &s_above, k);
    const int b0 = s_bin;
    const int above0 = s_above;
    const int T = (int)hist_r[b0];
    const bool whole = above0 + T == k;
    const bool store_ties = !whole && T <= CAP;

    const int nv4 = n >> 2;
    const int q = nv4 / C, r = nv4 % C;
    const int beg = slice * q + min(slice, r);
    const int end = beg + q + (slice < r ? 1 : 0);
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
    const int lane = tid & 31;
    for (int i = beg + tid; i < end; i += 512) {
        float4 f = __ldcs(in4 + i);
        unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
        // hits per row total only ~k + |boundary bucket| << n: plain
        // per-hit global atomics beat the per-element dual-ballot tax.
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = (int)(kk[j] >> 21);
            if (d > b0) {
                outr[atomicAdd(gt_write, 1u)] = 4 * i + j;
            } else if (d == b0) {
                const unsigned int s = atomicAdd(tie_write, 1u);
                if (whole) outr[above0 + s] = 4 * i + j;
                else if (store_ties)
                    tiebuf[s] = make_uint2(kk[j], (unsigned int)(4 * i + j));
            }
        }
    }
    if (slice == C - 1) {
        for (int i = (nv4 << 2) + tid; i < n; i += 512) {
            const unsigned int key = fkey(__ldcs(lgr + i));
            const int d = (int)(key >> 21);
            if (d > b0) {
                outr[atomicAdd(gt_write, 1u)] = i;
            } else if (d == b0) {
                const unsigned int s = atomicAdd(tie_write, 1u);
                if (whole) outr[above0 + s] = i;
                else if (store_ties)
                    tiebuf[s] = make_uint2(key, (unsigned int)i);
            }
        }
    }
}

__global__ __launch_bounds__(512, 1)
void tp_finish(const float* __restrict__ logits, long rstride, int n, int k,
               unsigned int* __restrict__ hist, unsigned int* __restrict__ sc,
               int* __restrict__ out) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    unsigned int* hist_r = hist + (size_t)row * 2048;
    unsigned int* sc_r = sc + (size_t)row * TP_SCROW;
    uint2* tiebuf = reinterpret_cast<uint2*>(sc_r + 4);
    int* outr = out + (size_t)row * (size_t)k;

    __shared__ __align__(16) unsigned int sh_hist[2048];
    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin, s_above, s_cnt, s_ntie;
    __shared__ unsigned int s_eqk[32];
    __shared__ int s_eqv[32];

    find_boundary_bins(hist_r, 2048, warp_totals, &s_bin, &s_above, k);
    const int b0 = s_bin;
    const int above0 = s_above;
    const int T = (int)hist_r[b0];
    const int R = k - above0;

    // ALL threads must finish their T load before anyone zeroes hist_r —
    // without this fence a fast thread zeroes hist_r[b0] while a slow one
    // is still loading T (observed: stochastic single-row wrong values at
    // BS=1024).
    __syncthreads();
    // re-zero hist + counters for the next call (this kernel is the last
    // reader; launch boundary orders it for the next batch).
    for (int i = tid; i < 2048; i += 512) hist_r[i] = 0u;
    if (tid == 0) { sc_r[0] = 0u; sc_r[1] = 0u; }

    if (T == R) return;                      // whole bucket: K2 finished it

    if (T <= CAP) {
        // ---- refine low 21 bits of the T boundary candidates (compB
        // fast-tail structure, single 512-thread CTA) ----
        unsigned int tk[PT];
        int tv[PT];
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            int p = tid + j * 512;
            if (p < T) {
                uint2 e = tiebuf[p];
                tk[j] = e.x; tv[j] = (int)e.y;
            } else { tk[j] = 0u; tv[j] = 0; }
        }
        __syncthreads();
        for (int i = tid; i < 2048; i += 512) sh_hist[i] = 0u;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            if (tid + j * 512 < T)
                atomicAdd(&sh_hist[(tk[j] >> 10) & 2047u], 1u);
        }
        __syncthreads();
        find_boundary_bins(sh_hist, 2048, warp_totals, &s_bin, &s_above, R);
        const int bA = s_bin;
        const int aboveA = s_above;
        const int TA = (int)sh_hist[bA];
        const bool wholeA = aboveA + TA == R;
        if (tid == 0) { s_cnt = 0; s_ntie = 0; }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            bool active = tid + j * 512 < T;
            int binA = active ? (int)((tk[j] >> 10) & 2047u) : -1;
            bool p = wholeA ? (binA >= bA) : (binA > bA);
            unsigned int m = __ballot_sync(0xffffffffu, p);
            if (p) {
                int rk = __popc(m & ((1u << lane) - 1));
                int leader = __ffs(m) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                base = __shfl_sync(m, base, leader);
                outr[above0 + base + rk] = tv[j];
            }
            if (!wholeA && binA == bA) {
                int qi = atomicAdd(&s_ntie, 1);
                if (qi < 32) { s_eqk[qi] = tk[j]; s_eqv[qi] = tv[j]; }
            }
        }
        __syncthreads();
        if (wholeA) return;
        const int R2 = R - aboveA;
        if (TA <= 32) {
            if (tid < 32) {
                unsigned int score =
                    tid < TA ? ((s_eqk[tid] & 1023u) + 1u) : 0u;
                for (int rr = 0; rr < R2; ++rr) {
                    unsigned int bs = score;
                    int bl = tid;
#pragma unroll
                    for (int off = 16; off; off >>= 1) {
                        unsigned int os = __shfl_down_sync(0xffffffffu, bs, off);
                        int ol = __shfl_down_sync(0xffffffffu, bl, off);
                        if (tid + off < 32 && os > bs) { bs = os; bl = ol; }
                    }
                    int winner = __shfl_sync(0xffffffffu, bl, 0);
                    if (tid == winner) {
                        outr[above0 + aboveA + rr] = s_eqv[tid];
                        score = 0u;
                    }
                }
            }
            return;
        }
        // rare big-tie fallback: pass B over bits 9..0
        __syncthreads();
        for (int i = tid; i < 1024; i += 512) sh_hist[i] = 0u;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            if ((tid + j * 512 < T) && (int)((tk[j] >> 10) & 2047u) == bA)
                atomicAdd(&sh_hist[tk[j] & 1023u], 1u);
        }
        __syncthreads();
        find_boundary_bins(sh_hist, 1024, warp_totals, &s_bin, &s_above, R2);
        const int bB = s_bin;
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            bool inA = (tid + j * 512 < T) &&
                       (int)((tk[j] >> 10) & 2047u) == bA;
            bool p = inA && (int)(tk[j] & 1023u) > bB;
            unsigned int m = __ballot_sync(0xffffffffu, p);
            if (p) {
                int rk = __popc(m & ((1u << lane) - 1));
                int leader = __ffs(m) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                base = __shfl_sync(m, base, leader);
                outr[above0 + base + rk] = tv[j];
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            bool inA = (tid + j * 512 < T) &&
                       (int)((tk[j] >> 10) & 2047u) == bA;
            bool p = inA && (int)(tk[j] & 1023u) == bB;
            unsigned int m = __ballot_sync(0xffffffffu, p);
            if (p) {
                int rk = __popc(m & ((1u << lane) - 1));
                int leader = __ffs(m) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                base = __shfl_sync(m, base, leader);
                int slot = base + rk;
                if (slot < R) outr[above0 + slot] = tv[j];
            }
        }
        return;
    }

    // ---- rare T > CAP: in-CTA 11/11/10 ladder over row re-reads ----
    const int nv4 = n >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
    int remaining = R;
    int total_above = above0;
    unsigned int prefix = ((unsigned int)b0) << 21;
    int consumed = 11;
#pragma unroll
    for (int pass = 1; pass < 3; ++pass) {
        const int shift = pass == 1 ? 10 : 0;
        const int bits = pass == 1 ? 11 : 10;
        const int nb = 1 << bits;
        const unsigned int digit_mask = (unsigned int)(nb - 1);
        const unsigned int high_mask = 0xffffffffu << (shift + bits);
        for (int i = tid; i < nb; i += 512) sh_hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < nv4; i += 512) {
            float4 f = __ldcs(in4 + i);
            unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if ((kk[j] & high_mask) == (prefix & high_mask))
                    atomicAdd(&sh_hist[(kk[j] >> shift) & digit_mask], 1u);
        }
        for (int i = (nv4 << 2) + tid; i < n; i += 512) {
            unsigned int key = fkey(__ldcs(lgr + i));
            if ((key & high_mask) == (prefix & high_mask))
                atomicAdd(&sh_hist[(key >> shift) & digit_mask], 1u);
        }
        __syncthreads();
        find_boundary_bins(sh_hist, nb, warp_totals, &s_bin, &s_above,
                           remaining);
        bool whole_bucket = s_above + (int)sh_hist[s_bin] == remaining;
        prefix |= ((unsigned int)s_bin) << shift;
        total_above += s_above;
        remaining -= s_above;
        consumed += bits;
        if (whole_bucket) break;
        __syncthreads();
    }
    const int final_shift = 32 - consumed;
    const unsigned int threshold =
        final_shift ? (prefix >> final_shift) : prefix;
    if (tid == 0) { s_cnt = 0; s_ntie = 0; }
    __syncthreads();
    // final collect: winners strictly above threshold from bucket b0, plus
    // any-subset fill of full-key ties at the k-th value.
    for (int i = tid; i < nv4; i += 512) {
        float4 f = __ldcs(in4 + i);
        unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int key = kk[j];
            if ((int)(key >> 21) != b0) continue;
            if (final_shift) key >>= final_shift;
            const bool gt = key > threshold;
            const bool eq = key == threshold;
            const unsigned int gm = __ballot_sync(0xffffffffu, gt);
            const unsigned int em = __ballot_sync(0xffffffffu, eq);
            if (gt) {
                int rk = __popc(gm & ((1u << lane) - 1));
                int leader = __ffs(gm) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_cnt, __popc(gm));
                base = __shfl_sync(gm, base, leader);
                outr[above0 + base + rk] = 4 * i + j;
            }
            if (eq) {
                int rk = __popc(em & ((1u << lane) - 1));
                int leader = __ffs(em) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_ntie, __popc(em));
                base = __shfl_sync(em, base, leader);
                int slot = base + rk;
                if (slot < remaining) outr[total_above + slot] = 4 * i + j;
            }
        }
    }
    for (int i = (nv4 << 2) + tid; i < n; i += 512) {
        unsigned int key = fkey(__ldcs(lgr + i));
        if ((int)(key >> 21) != b0) continue;
        if (final_shift) key >>= final_shift;
        if (key > threshold) outr[above0 + atomicAdd(&s_cnt, 1)] = i;
        else if (key == threshold) {
            int slot = atomicAdd(&s_ntie, 1);
            if (slot < remaining) outr[total_above + slot] = i;
        }
    }
}

static unsigned int* g_tp_hist = nullptr;
static unsigned int* g_tp_sc = nullptr;
static int g_tp_rows = 0;

void tp_launch(const float* logits, long W, int n, int k, int* out, int BS,
               cudaStream_t stream) {
    if (g_tp_rows < BS) {
        if (g_tp_hist) cudaFree(g_tp_hist);
        if (g_tp_sc) cudaFree(g_tp_sc);
        cudaMalloc(&g_tp_hist, (size_t)BS * 2048 * sizeof(unsigned int));
        cudaMalloc(&g_tp_sc, (size_t)BS * TP_SCROW * sizeof(unsigned int));
        cudaMemset(g_tp_hist, 0, (size_t)BS * 2048 * sizeof(unsigned int));
        cudaMemset(g_tp_sc, 0, (size_t)BS * TP_SCROW * sizeof(unsigned int));
        g_tp_rows = BS;
    }
    if (!g_sms)
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
    int C = (4 * g_sms + BS - 1) / BS;      // target ~4 CTAs per SM
    if (C > 32) C = 32;
    if (C < 1) C = 1;
    const dim3 grid((unsigned int)C, (unsigned int)BS);
    tp_hist<<<grid, 512, 0, stream>>>(logits, W, n, C, g_tp_hist);
    tp_collect<<<grid, 512, 0, stream>>>(logits, W, n, k, C, g_tp_hist,
                                         g_tp_sc, out);
    tp_finish<<<BS, 512, 0, stream>>>(logits, W, n, k, g_tp_hist, g_tp_sc,
                                      out);
}
} // namespace aefm

namespace v30 {

namespace cg = cooperative_groups;

// Hand-rolled inter-block barrier for a regular (non-cooperative) launch.
// Safe only when all blocks are co-resident (we size the grid to residency,
// exactly as the cooperative launch requires).  Avoids the cooperative-launch
// runtime premium while giving the same all-block rendezvous.
// Grid barrier for a regular (non-cooperative) launch.  `arrive` counts blocks
// reaching the current barrier; the last arriver resets it and publishes the
// barrier's unique `sense` token to `release`.  `sense` is globally monotonic
// (host generation counter * 8 + barrier index), so a leftover `release` value
// from a previous launch can never be mistaken for the current barrier -> no
// per-launch reset needed.  Safe as long as all `gridsz` blocks are co-resident.
__device__ __forceinline__ void global_barrier(unsigned int* arrive,
                                                unsigned int* release,
                                                int gridsz, unsigned int sense) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int a = atomicAdd(arrive, 1u) + 1u;
        if (a == (unsigned int)gridsz) {
            atomicExch(arrive, 0u);
            atomicExch(release, sense);
        } else {
            while (atomicAdd(release, 0u) != sense) { }
        }
    }
    __syncthreads();
}

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

// scratch layout (u32): hist[6144], gt, tie, arrive, release.
template<bool COOP>
__global__ __launch_bounds__(512, 4)
void topk_coop_t(const float* __restrict__ logits, int n, int k,
               unsigned int* __restrict__ scratch, int* __restrict__ out,
               unsigned int gen) {
    cg::grid_group grid = cg::this_grid();
    int gridsz = gridDim.x;
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    int gstride = gridDim.x * blockDim.x;
    unsigned int* global_hist = scratch;
    int* gt_write = (int*)(scratch + 3 * 4096);
    int* tie_write = (int*)(scratch + 3 * 4096 + 1);
    unsigned int* g_arrive = scratch + 3 * 4096 + 2;
    unsigned int* g_release = scratch + 3 * 4096 + 3;


    __shared__ unsigned int hist[4096];
    __shared__ unsigned int warp_totals[16];
    __shared__ int boundary;
    __shared__ int above;
    __shared__ int vec_begin;
    __shared__ int vec_end;
    __shared__ int scalar_begin;
    __shared__ int scalar_end;

    if (gtid == 0) {
        *gt_write = 0;
        *tie_write = 0;
    }
    if (tid == 0) {
        const int nvec = (n & ~3) >> 2;
        const int vq = nvec / gridsz;
        const int vr = nvec - vq * gridsz;
        vec_begin = blockIdx.x * vq + min((int)blockIdx.x, vr);
        vec_end = vec_begin + vq + ((int)blockIdx.x < vr);

        const int sq = n / gridsz;
        const int sr = n - sq * gridsz;
        scalar_begin = blockIdx.x * sq + min((int)blockIdx.x, sr);
        scalar_end = scalar_begin + sq + ((int)blockIdx.x < sr);
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
        for (int i4 = vec_begin + tid; i4 < vec_end; i4 += blockDim.x) {
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
        if (COOP) { grid.sync(); }
        else { global_barrier(g_arrive, g_release, gridsz, gen * 8u + (unsigned int)pass); }
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
    int iters = (scalar_end - scalar_begin + blockDim.x - 1) / blockDim.x;
    int ties_needed = remaining;
    for (int it = 0; it < iters; ++it) {
        int i = scalar_begin + it * blockDim.x + tid;
        unsigned int key = i < scalar_end ? fkey(logits[i]) : 0u;
        if (final_shift) key >>= final_shift;
        bool gt = i < scalar_end && key > threshold;
        bool eq = i < scalar_end && key == threshold;
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
static unsigned int* g_scratch2 = nullptr;
static int g_max_coop_blocks = 0;
static int g_max_coop_blocks2 = 0;
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
        cudaMalloc(&g_scratch, (3 * 4096 + 4) * sizeof(unsigned int));
        cudaMemset(g_scratch, 0, (3 * 4096 + 4) * sizeof(unsigned int));
    }
    if (!g_max_coop_blocks) {
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_coop_t<false>, 512, 0);
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_max_coop_blocks = active * g_sms;
    }
    int blocks = (n + 2047) / 2048;
    if (blocks < g_sms) blocks = g_sms;
    if (blocks > g_max_coop_blocks) blocks = g_max_coop_blocks;
    // Globally-monotonic generation so barrier sense tokens never collide across
    // launches -> the persistent arrive/release words need no per-launch reset.
    static unsigned int g_gen = 0;
    ++g_gen;
#ifdef USE_COOP
    unsigned int gen = g_gen;
    void* args[] = {(void*)&logits, (void*)&n, (void*)&k,
                    (void*)&g_scratch, (void*)&out, (void*)&gen};
    cudaLaunchCooperativeKernel((void*)topk_coop_t<true>, dim3(blocks), dim3(512),
                                args, 0, stream);
#else
    // Regular launch + hand-rolled global barrier (blocks are co-resident since
    // grid is sized to residency).  Avoids cooperative-launch runtime premium.
    topk_coop_t<false><<<blocks, 512, 0, stream>>>(logits, n, k, g_scratch, out, g_gen);
#endif
}

} // namespace v30

// Engineer composite dispatch (r3_compB):
// K=2048 mid-n prefers the contiguous-slice 3-barrier ladder (30e79029);
// everything else on aef33fac (becd fast-tail + topk_mid rungs, mid<1> gated out).
void topk_launch(const float* logits, int n, int k, int* out,
                 cudaStream_t stream) {
    if (k == 2048 && n > 16896 && n <= 140000) {
        v30::topk_launch(logits, n, k, out, stream);
    } else {
        aefm::topk_launch(logits, n, k, out, stream);
    }
}

// Extension dispatch: aefm batched paths; the v30 range (K=2048 mid-n) is out
// of the minimal experiment's scope and falls back to a sequential row loop.
void topk_launch_ext(const float* logits, long W, int n, int k, int* out,
                     int BS, cudaStream_t stream) {
    if (k == 2048 && n > 16896 && n <= 140000) {
        for (int r = 0; r < BS; ++r)
            v30::topk_launch(logits + (size_t)r * (size_t)W, n, k,
                             out + (size_t)r * (size_t)k, stream);
        return;
    }
    aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
}

// (path, team, cap, rows_per_wave, waves) for a given (n, k, BS).
// path: 0 = small-tier grid.y batch, 1 = row-team, 2 = v30 sequential
// fallback, 3 = row-wider-than-wave sequential fallback.
void topk_ext_info(int n, int k, int BS, int info[5]) {
    if (k == 2048 && n > 16896 && n <= 140000) {
        info[0] = 2; info[1] = 0; info[2] = 0; info[3] = 1; info[4] = BS;
        return;
    }
    if (n <= 16896) {
        info[0] = 0; info[1] = 1; info[2] = 0; info[3] = BS; info[4] = 1;
        return;
    }
    int team, cap;
    aefm::ext_caps(&team, &cap, n);
    if (team > cap) {
        info[0] = 3; info[1] = team; info[2] = cap; info[3] = 1; info[4] = BS;
        return;
    }
    const int rpw = cap / team;
    info[0] = 1; info[1] = team; info[2] = cap; info[3] = rpw;
    info[4] = (BS + rpw - 1) / rpw;
}

// Diet-variant entries. topk_launch_ext_v(minb=1) == topk_launch_ext for
// large-n modulo the smem-carveout attribute (cap may exceed active=2).
void topk_launch_ext_v(const float* logits, long W, int n, int k, int* out,
                       int BS, int minb, cudaStream_t stream) {
    if (k == 2048 && n > 16896 && n <= 140000) {
        for (int r = 0; r < BS; ++r)
            v30::topk_launch(logits + (size_t)r * (size_t)W, n, k,
                             out + (size_t)r * (size_t)k, stream);
        return;
    }
    if (n <= 16896) {
        aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
        return;
    }
    aefm::launch_fast_teams_v(logits, W, n, k, out, BS, minb, stream);
}

void topk_fast_stats(int minb, int out5[5]) { aefm::fast_stats_v(minb, out5); }

// B' persistent-queue entries.
void topk_launch_pq_v(const float* logits, long W, int n, int k, int* out,
                      int BS, int minb, cudaStream_t stream) {
    if (k == 2048 && n > 16896 && n <= 140000) {
        for (int r = 0; r < BS; ++r)
            v30::topk_launch(logits + (size_t)r * (size_t)W, n, k,
                             out + (size_t)r * (size_t)k, stream);
        return;
    }
    if (n <= 16896) {
        aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
        return;
    }
    aefm::launch_fast_pq_v(logits, W, n, k, out, BS, minb, stream);
}

void topk_pq_stats(int minb, int out5[5]) { aefm::pq_stats_v(minb, out5); }

// D1 throughput-arm entry (large-n only; small-n keeps grid.y batching).
void topk_launch_tp(const float* logits, long W, int n, int k, int* out,
                    int BS, cudaStream_t stream) {
    if (n <= 16896) {
        aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
        return;
    }
    aefm::tp_launch(logits, W, n, k, out, BS, stream);
}
