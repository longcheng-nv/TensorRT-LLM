#include <cuda_runtime.h>
#include <cstdint>

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

// ---------------------------------------------------------------------------
// Large-n kernel.
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(BLOCK, 1)
void topk_fast(const float* __restrict__ logits, int n, int k,
               unsigned int* __restrict__ scratch, int* __restrict__ out,
               unsigned int gen) {
    const int gridsz = gridDim.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int gtid = blockIdx.x * BLOCK + tid;
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
        cudaMalloc(&g_scratch, SCRATCH_WORDS * sizeof(unsigned int));
        cudaMemset(g_scratch, 0, SCRATCH_WORDS * sizeof(unsigned int));
    }
    if (!g_blocks_cap) {
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_fast, BLOCK, 0);
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
    topk_fast<<<blocks, BLOCK, 0, stream>>>(logits, n, k, g_scratch, out, g_gen);
}
