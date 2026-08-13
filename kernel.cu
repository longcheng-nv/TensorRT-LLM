/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include "kernel.h"

namespace cg = cooperative_groups;

#define FULLM 0xffffffffu
#define NB   1024
#define MAXC 160         /* max rows on the multi-CTA path */
#define GCAP 16384
#define GVR_MAX_DEV 64            /* per-device attribute latch (B1c) */
/* crossing-bin size below which the O(mc^2) rank beats histogram narrowing */
#define QUADC 96
/* r3: clus-only QUAD gate.  The 512k-band crossing bin measures ~250, just
   past QUADC, so every row pays the multi-level narrowing (clear+bin+scan+3
   barriers per level) on rank 0 while the rest of the grid has drained --
   pure wall-time tail.  The O(mc^2) shared-broadcast rank at mc<=288 is ~290
   iterations of 4 inst on 9 warps, no barriers.  Streaming/reg paths keep 96
   (measured there). */
#define QUADC_CLUS 288
#define IDXB 22                  /* index bits packed alongside the bin */
#define IDXM ((1u << IDXB) - 1u)

/* Multi-CTA row split publishes its candidates into one global SLAB and hands
   the whole selection to whichever CTA arrives last -- no grid spin barrier, no
   cross-CTA histogram merge, no read-back round trip.
   B2 (prod hardening): the slab lives in a CALLER-PROVIDED workspace instead of
   __device__ globals, so two streams can run concurrently with two workspaces.
   Layout (offsets in bytes; workspace must be zero-initialised once, 8B-aligned;
   the kernel restores the zeros it consumes, so one zeroing per buffer suffices):
     [0,               MAXC*8)  g_don  (arrivals << 32) | candidates
     [GVR_WS_OFF_OFF,  +MAXC*4) g_off  slab write cursor
     [GVR_WS_BUF_OFF,  +MAXC*GCAP*8) g_buf (value bits, index) per row */
#define GVR_WS_OFF_OFF (MAXC * 8)
#define GVR_WS_BUF_OFF 2048
size_t gvr_topk_workspace_bytes() {
    return GVR_WS_BUF_OFF + (size_t)MAXC * GCAP * sizeof(int2);
}

/* B3 (prod hardening): PDL hooks. Compile with -DGVR_ENABLE_PDL and launch via
   cudaLaunchKernelEx + cudaLaunchAttributeProgrammaticStreamSerialization to
   overlap with the producer kernel; default build compiles to nothing. The
   entry-side wait is the correctness-mandatory half; the explicit
   launch_dependents trigger is left to the in-tree port (kernel completion
   implicitly releases dependents). */
#if defined(GVR_ENABLE_PDL)
#define GVR_GDC_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#else
#define GVR_GDC_WAIT() do { } while (0)
#endif
/* redux.sync: a SINGLE instruction for the whole 32-lane min/max, replacing a
   5-deep dependent __shfl_down chain (SM80+). */
__device__ __forceinline__ uint32_t warp_min_u32(uint32_t v) { return __reduce_min_sync(0xffffffffu, v); }
__device__ __forceinline__ uint32_t warp_max_u32(uint32_t v) { return __reduce_max_sync(0xffffffffu, v); }

__device__ __forceinline__ float invkey(uint32_t K) {
    uint32_t u = (K & 0x80000000u) ? (K ^ 0x80000000u) : ~K;
    return __uint_as_float(u);
}
__device__ __forceinline__ uint32_t fkey(float x) {
    uint32_t u = __float_as_uint(x);
    return u ^ ((uint32_t)((int32_t)u >> 31) | 0x80000000u);
}

/* highest bin B with sum_{j>=B} hist[j] >= target; also total, m=hist[B],
   above=sum_{j>B}.  Warp-parallel, bank-conflict free. */
template <int NB_ = 1024>
__device__ __forceinline__ void find_cross(const uint32_t* __restrict__ hist, int target,
                                           int tid, int lane,
                                           int* s_B, int* s_m, int* s_above, int* s_tot) {
    constexpr int BPL_ = NB_ / 32;          /* bins per lane */
    if (tid < 32) {
        uint32_t part = 0;
#pragma unroll
        for (int j = 0; j < BPL_; j++) part += hist[lane * BPL_ + ((j + lane) & (BPL_ - 1))];
        uint32_t v = part;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            uint32_t z = __shfl_down_sync(FULLM, v, o);
            if (lane + o < 32) v += z;
        }
        if (lane == 0) *s_tot = (int)v;
        unsigned msk = __ballot_sync(FULLM, v >= (uint32_t)target);
        int L = msk ? (31 - __clz(msk)) : 0;
        uint32_t aboveL = __shfl_sync(FULLM, v - part, L);
        uint32_t h = (lane < BPL_) ? hist[L * BPL_ + lane] : 0u;
        uint32_t w = h;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            uint32_t z = __shfl_down_sync(FULLM, w, o);
            if (lane + o < 32) w += z;
        }
        unsigned msk2 = __ballot_sync(FULLM, (aboveL + w) >= (uint32_t)target);
        int J = msk2 ? (31 - __clz(msk2)) : 0;
        if (lane == J) {
            *s_B = L * BPL_ + J;
            *s_m = (int)h;
            *s_above = (int)(aboveL + (w - h));
        }
    }
}

/* r5 (peer ab6a7302): warp0-fused cluster merge + suffix scan.  The
   256-thread DSMEM merge loop (mrg/hoff materialisation) and its publishing
   __syncthreads are folded into the warp-0 scan: each lane reads its 8-bin
   span from EVERY rank's hist via DSMEM uint4 loads, sums the cluster totals
   (and the r<rank prefix that biases this rank's cursors) in registers, runs
   the usual suffix scan and writes the biased output cursors straight into
   mrg.  One barrier (the post-scan publish the caller already pays) instead
   of two, and no hoff[] array at all. */
template <int NB_, int CS>
__device__ __forceinline__ void merge_scan0(uint32_t* __restrict__ hist,
                                            uint32_t* __restrict__ mrg,
                                            cg::cluster_group& clus, int rank,
                                            int target, int tid, int lane,
                                            int* s_B, int* s_m, int* s_above, int* s_tot) {
    constexpr int BPT = NB_ / 32;
    constexpr int NV  = BPT / 4;
    static_assert(NV >= 1 && BPT % 4 == 0, "NB_ must be a multiple of 128");
    if (tid < 32) {
        uint4 tot[NV], pre[NV];
        uint32_t sm = 0u;
#pragma unroll
        for (int q = 0; q < NV; q++) {
            uint4 t = make_uint4(0u,0u,0u,0u), p = make_uint4(0u,0u,0u,0u);
#pragma unroll
            for (int r = 0; r < CS; r++) {
                const uint4* src = (const uint4*)clus.map_shared_rank(hist + lane * BPT + 4 * q, r);
                uint4 v = *src;
                t.x += v.x; t.y += v.y; t.z += v.z; t.w += v.w;
                if (r < rank) { p.x += v.x; p.y += v.y; p.z += v.z; p.w += v.w; }
            }
            tot[q] = t; pre[q] = p;
            sm += t.x + t.y + t.z + t.w;
        }
        uint32_t w = sm;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { uint32_t z = __shfl_up_sync(FULLM, w, o); if (lane >= o) w += z; }
        const uint32_t tt = __shfl_sync(FULLM, w, 31);
        uint32_t after = tt - w;
        if (lane == 0) *s_tot = (int)tt;
        const int base = lane * BPT;
#pragma unroll
        for (int q = NV - 1; q >= 0; q--) {
            const uint32_t c[4]  = { tot[q].x, tot[q].y, tot[q].z, tot[q].w };
            const uint32_t pr[4] = { pre[q].x, pre[q].y, pre[q].z, pre[q].w };
            uint32_t o4[4];
#pragma unroll
            for (int j = 3; j >= 0; j--) {
                o4[j] = after + pr[j];
                const int gb = base + 4 * q + j;
                if ((int)after < target && ((int)(after + c[j]) >= target || gb == 0)) {
                    *s_B = gb; *s_above = (int)after; *s_m = (int)c[j];
                }
                after += c[j];
            }
            ((uint4*)mrg)[lane * NV + q] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
    }
}

#define SNB 256   /* streaming-path bin count: 1k-8k candidates need no more.
                     MEASURED r7: 1024 bins cost +5..+13% on every streaming
                     workload -- the warp-0 suffix scan parks the whole CTA at a
                     barrier, so its cost scales with bins and swamps the finer
                     rung and smaller crossing bin it buys.
                     r5 RE-MEASURED post-scan_cross0: 512 bins STILL cost +5.8%
                     on pro_512k_L26 (16.66 vs 15.74us) -- the bin count's cost
                     is not the scan barrier; keep 256. */

/* Block-parallel suffix scan over NB_ (<= BLK) bins.  Leaves hist[j] = number
   of elements in bins > j -- the per-bin OUTPUT CURSOR -- and pins the highest
   bin B with sum_{j>=B} >= target.  Warps that hold no bin skip the whole body,
   so the cost scales with NB_ and not with the block size (the classic
   warp-0-only form instead parks every other warp at a barrier). */
template <int BLK, int NB_, bool TWO = false>
__device__ __forceinline__ void scan_cross(uint32_t* __restrict__ hist,
                                           uint32_t* __restrict__ ws, int target,
                                           int tid, int lane,
                                           int* s_B, int* s_m, int* s_above, int* s_tot,
                                           int target2 = 0, int* s_B2 = nullptr) {
    constexpr int NWU = NB_ / 32;
    const int wid = tid >> 5;
    uint32_t c = 0u, w = 0u;
    if (tid < NB_) {
        c = hist[tid]; w = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { uint32_t z = __shfl_up_sync(FULLM, w, o); if (lane >= o) w += z; }
        if (lane == 31) ws[wid] = w;
    }
    __syncthreads();
    if (tid < NB_) {
        uint32_t v2 = (lane < NWU) ? ws[lane] : 0u, pre = v2;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { uint32_t z = __shfl_up_sync(FULLM, pre, o); if (lane >= o) pre += z; }
        const uint32_t tot = __shfl_sync(FULLM, pre, 31);
        const uint32_t off = __shfl_sync(FULLM, pre - v2, wid);
        const uint32_t after = tot - (off + w);
        if (tid == 0) *s_tot = (int)tot;
        hist[tid] = after;
        if ((int)after < target && ((int)(after + c) >= target || tid == 0)) {
            *s_B = tid; *s_above = (int)after; *s_m = (int)c;
        }
        if constexpr (TWO) {
            if ((int)after < target2 && ((int)(after + c) >= target2 || tid == 0)) *s_B2 = tid;
        }
    }
}

/* Warp-0-only suffix scan for the streaming path's NB_=256 bins, reading and
   writing each lane's private 8-bin span as two 16B vectors.  Nothing here
   crosses a warp, so the caller pays ONE block barrier instead of the two the
   block-parallel form needs (its internal ws[] handoff is a barrier that all
   BLK/32 warps must reach, and on a 32-warp CTA that is the dominant cost of
   the phase).  ZERO=true leaves the bins CLEARED instead of leaving output
   cursors, which folds the next phase's histogram clear -- and its barrier --
   into this one. */
template <int NB_, bool ZERO, bool TWO = false, bool THREE = false, bool ADD = false>
__device__ __forceinline__ void scan_cross0(uint32_t* __restrict__ hist, int target,
                                            int tid, int lane,
                                            int* s_B, int* s_m, int* s_above, int* s_tot,
                                            int target2 = 0, int* s_B2 = nullptr,
                                            int target3 = 0, int* s_B3 = nullptr,
                                            const uint32_t* __restrict__ addv = nullptr) {
    constexpr int BPT = NB_ / 32;          /* bins per lane            */
    constexpr int NV  = BPT / 4;           /* 16B vectors per lane     */
    static_assert(BPT % 4 == 0, "NB_ must be a multiple of 128");
    // Holding all NV vectors across the shuffle scan costs 4*NV registers in a
    // kernel that is already pinned at the 64-register wall by __launch_bounds__;
    // past two vectors that is a spill, so the wide-bin instantiations re-READ
    // their span (NV extra shared loads, no barrier) instead of holding it.
    constexpr bool HOLD = (NV <= 2);
    if (tid < 32) {
        uint4 v[HOLD ? NV : 1];
        uint32_t sm = 0u;
#pragma unroll
        for (int q = 0; q < NV; q++) {
            uint4 t = ((const uint4*)hist)[lane * NV + q];
            if constexpr (HOLD) v[q] = t;
            sm += t.x + t.y + t.z + t.w;
        }
        uint32_t w = sm;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { uint32_t z = __shfl_up_sync(FULLM, w, o); if (lane >= o) w += z; }
        const uint32_t tot = __shfl_sync(FULLM, w, 31);
        uint32_t after = tot - w;          /* bins strictly above my span */
        if (lane == 0) *s_tot = (int)tot;
        const int base = lane * BPT;
#pragma unroll
        for (int q = NV - 1; q >= 0; q--) {
            uint4 vv;
            if constexpr (HOLD) vv = v[q < (HOLD ? NV : 1) ? q : 0];
            else                vv = ((const uint4*)hist)[lane * NV + q];
            const uint32_t c[4] = { vv.x, vv.y, vv.z, vv.w };
            uint32_t o4[4];
#pragma unroll
            for (int j = 3; j >= 0; j--) {
                o4[j] = ZERO ? 0u : after;
                const int gb = base + 4 * q + j;
                if ((int)after < target && ((int)(after + c[j]) >= target || gb == 0)) {
                    *s_B = gb; *s_above = (int)after; *s_m = (int)c[j];
                }
                if constexpr (TWO) {
                    if ((int)after < target2 && ((int)(after + c[j]) >= target2 || gb == 0)) *s_B2 = gb;
                }
                if constexpr (THREE) {
                    if ((int)after < target3 && ((int)(after + c[j]) >= target3 || gb == 0)) *s_B3 = gb;
                }
                after += c[j];
            }
            if constexpr (ADD) {   /* fold the per-rank bin offset into the cursor */
                uint4 av = ((const uint4*)addv)[lane * NV + q];
                o4[0] += av.x; o4[1] += av.y; o4[2] += av.z; o4[3] += av.w;
            }
            ((uint4*)hist)[lane * NV + q] = make_uint4(o4[0], o4[1], o4[2], o4[3]);
        }
    }
}

/* Same idea for NB_ >= BLK (the register path bins at 1024): every thread owns
   a private contiguous span of bins, so its read->write needs no barrier. */
template <int BLK, int NB_>
__device__ __forceinline__ void scan_cross_w(uint32_t* __restrict__ hist,
                                             uint32_t* __restrict__ ws, int target,
                                             int tid, int lane,
                                             int* s_B, int* s_m, int* s_above, int* s_tot) {
    constexpr int BPT = NB_ / BLK;
    constexpr int NW  = BLK / 32;
    uint32_t loc[BPT], sm = 0;
    const int base = tid * BPT;
#pragma unroll
    for (int i = 0; i < BPT; i++) { loc[i] = hist[base + i]; sm += loc[i]; }
    uint32_t w = sm;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { uint32_t z = __shfl_up_sync(FULLM, w, o); if (lane >= o) w += z; }
    const int wid = tid >> 5;
    if (lane == 31) ws[wid] = w;
    __syncthreads();
    // Every thread of a warp wants the SAME two numbers -- the block total and
    // the sum of the warps BELOW it -- so this stage is two REDUCTIONS, not a
    // scan.  redux.sync computes each in one instruction and broadcasts it,
    // replacing a 10-deep DEPENDENT shfl_up chain plus two shuffle broadcasts
    // (14 instructions -> 5).  wid is warp-uniform, so the masked operand is
    // uniform inside the warp and the reduction stays convergent.
    const uint32_t vv  = (lane < NW) ? ws[lane] : 0u;
    const uint32_t tot = __reduce_add_sync(FULLM, vv);
    const uint32_t off = __reduce_add_sync(FULLM, (lane < wid) ? vv : 0u);
    uint32_t after = tot - (off + w);
    if (tid == 0) *s_tot = (int)tot;
#pragma unroll
    for (int i = BPT - 1; i >= 0; i--) {
        const uint32_t c = loc[i];
        hist[base + i] = after;                 // per-bin OUTPUT CURSOR
        if ((int)after < target && ((int)(after + c) >= target || (base + i) == 0)) {
            *s_B = base + i; *s_above = (int)after; *s_m = (int)c;
        }
        after += c;
    }
}

/* LAZY HINT GATHER: block-wide (min,max) of logits[pre_idx[j]] over all k hint
   slots.  Runs the two dependent round trips (k coalesced pre_idx words, then
   k scattered 4B gathers) IN PLACE, so it is only called off the hot path: on
   an attempt failure or a degenerate sample.  Healthy rows land attempt 0 on
   the sample rung alone and never pay it.  Contains barriers -- call sites
   must be block-uniform.  GM_/GX_ are written by every thread (identical). */
#define GVR_GATHER_HINT(GM_, GX_, KPTV)                                          \
    do {                                                                         \
        uint32_t glmin = 0xffffffffu, glmax = 0u;                                \
        _Pragma("unroll")                                                        \
        for (int t = 0; t < (KPTV); t++) {                                       \
            int j = tid + t * BLK;                                               \
            int p = (j < k) ? P[j] : -1;                                         \
            if ((unsigned)p < (unsigned)n) {                                     \
                uint32_t u2 = fkey(__ldg(X + p));                                \
                glmin = min(glmin, u2); glmax = max(glmax, u2);                  \
            }                                                                    \
        }                                                                        \
        glmin = warp_min_u32(glmin); glmax = warp_max_u32(glmax);                \
        if (lane == 0) { wmn[tid >> 5] = glmin; wmx[tid >> 5] = glmax; }         \
        __syncthreads();                                                         \
        {                                                                        \
            constexpr int NW_ = BLK / 32;                                        \
            uint32_t a2 = (lane < NW_) ? wmn[lane] : 0xffffffffu;                \
            uint32_t c2 = (lane < NW_) ? wmx[lane] : 0u;                         \
            GM_ = invkey(warp_min_u32(a2)); GX_ = invkey(warp_max_u32(c2));      \
        }                                                                        \
        if (!(GM_ < GX_)) { GM_ = -3.0e38f; GX_ = 3.0e38f; }                     \
        __syncthreads();                                                         \
    } while (0)

/* ---------------------------------------------------------------------------
   Streaming GVR kernel.  R CTAs per row (R==1 for the common case).
   P1  gather logits[pre_idx] -> gmin (count(>=gmin) >= k guaranteed), gmax;
       the quantile sample is PREFETCHED in the same issue window so it rides
       under the two dependent hint round trips.
   P2  guess: quantile rung T from a small sample read as free 32B float4 pairs,
       aimed low inside a very wide accept window so one attempt always lands.
   P3  ONE row pass: a 32-element predicate MASK per thread amortises the slot
       reservation (one shared atomic per thread, no warp prefix scan) and the
       survivor walk over 32 elements, so the ~1% of elements above T pay the
       expensive path only ~3 times per 32 slots instead of ~17.
   P4  scan_cross turns the histogram into per-bin OUTPUT CURSORS and pins B.
   P5  emit: one atomicAdd on the bin cursor per candidate routes it to the
       output (bins > B) or to the crossing-bin buffer.  On collect overflow the
       row is re-swept with the same cursors -- never a second histogram pass.
   P6  the compacted crossing bin is refined in exact key space.
--------------------------------------------------------------------------- */
template <int BLK, int U, int MINB, int NBS, int KPT, bool SPLIT>
__global__ void __launch_bounds__(BLK, MINB)
gvr_main(const float* __restrict__ logits, const int* __restrict__ pre_idx,
         int* __restrict__ out, int n, int npad, int k,
         int SCAP_, int CMP_, int R, int SMP, int TGT, int Q, int SS2, int TGT2,
         void* __restrict__ slabws) {
    GVR_GDC_WAIT();
    /* B2: slab views into the caller workspace (aliases keep the body diff-free;
       only SPLIT instantiations ever dereference them). */
    unsigned*           g_off = (unsigned*)((char*)slabws + GVR_WS_OFF_OFF);
    unsigned long long* g_don = (unsigned long long*)slabws;
    int2 (*g_buf)[GCAP]       = (int2 (*)[GCAP])((char*)slabws + GVR_WS_BUF_OFF);
    (void)g_off; (void)g_don; (void)g_buf;
    // The ROW histogram needs only NBS bins (a few thousand candidates), but the
    // SAMPLE rung is quantised by ITS bin width: find_cross is conservative, so a
    // coarse sample bin never undershoots - it OVERSHOOTS, and every extra
    // candidate is paid for in the survivor walk.  Give the sample 4x the bins.
    constexpr int HB = NBS;
    // SCPB/CMPB are pure functions of `big`, which is exactly (BLK == 1024) on
    // every dispatch, so make them COMPILE-TIME.  SASS: with a runtime SCPB the
    // ck64 base is an IMAD chain the register allocator rematerialises -- NCU
    // charges 1.9% of the whole kernel to that one pointer expression.  As a
    // constant it folds into the LDS/STS immediate offset.  The histogram is
    // NBS words of compile-time size too, so it moves to STATIC shared (same
    // total per CTA) and its address loses the dynamic-window IADD pair.
    // Non-split 1024-thread staging doubled to 16K words: the accept window's
    // upper edge is what turns a LOW rung into a full-row P5 re-sweep, and at
    // 1 CTA/SM the extra 32KB of dynamic shared is free (80KB of 100KB budget).
    // KBIG marks the k>1024 (v32 K=2048) instantiations: KPT*BLK >= 2048 with
    // KPT >= 2 holds exactly for (1024,2), (512,4), (256,8) -- every dispatch
    // that reaches them has k > 1024, and no k <= 1024 dispatch does, so the
    // V4-domain shapes keep their tuned buffers bit-identically.  At k=2048
    // the k-th value sits ~2x deeper into the dense part of the histogram, so
    // the crossing bin and the staged-candidate population both scale ~2x:
    // without 2x CMPB/SCPB nearly every row overflows into the whole-row
    // key-space narrowing (measured 2.7-3.2x vs baseline on v32_256k bs>=256).
    constexpr bool KBIG = (KPT >= 2) && (KPT * BLK >= 2048);
    // iter3b: at k=2048 the old SCPB=4096 clamps aim to exactly k -- a ZERO
    // accept margin, so rows cascade att0 -> TSH -> GMIN and detonate
    // (measured 0.31-0.37x on v32_256k bs>=256).  SCPB=8192 restores the
    // 1.375x margin (aim 11k/8).  CMPB stays 1024 at BLK <= 512: doubling it
    // pushed the BLK=256 CTA to 65.6KB shared, 4 CTAs/SM = 262KB > 227KB, and
    // the occupancy drop cost 20-25% on healthy 128k cells (iter2), while
    // measured crossing bins at k=2048 stay ~100 (iter3a: CMPB x2 alone moved
    // nothing).  8192*4B + 1025*8B = 41KB keeps 4 CTAs/SM.
    constexpr int SCPB = (BLK >= 1024) ? (SPLIT ? 8192 : 16384)
                                       : (KBIG ? 8192 : 4096);
    constexpr int CMPB  = (BLK >= 1024) ? (KBIG ? 4096 : 2048) : 1024;
    // RUNG LADDER: the non-split 1024-thread variant retries a failed attempt 0
    // at TSH -- the sample's rank-(2*TGT) floor, available for free from the
    // same scan -- before collapsing to GMIN.  A sample-bias overshoot at the
    // rung (count(>=T) < k) is 3-5x smaller at twice the depth, so the TSH
    // retry lands with a BOUNDED emit (~2*aim) instead of the 110k-candidate
    // GMIN flood + full-row P5 re-sweep that made these rows 0.28-0.42x.
    // (A per-element shadow histogram of the [TSH,T) band was measured at +40%
    // on every ARMED row -- the scattered walk reloads sit naked on the
    // latency-critical iteration path at 1 CTA/SM.  The ladder is free.)
    // r5 (a005): extended to ALL non-split streaming variants -- the floor
    // lives in one shared word (no register cost at any BLK), and a midband /
    // wide-batch rung miss otherwise floods straight to GMIN.  With the
    // bounded TSH retry in place, the !big aim floor can afford a trim.
    constexpr bool SHD = !SPLIT;
    // r4 VSTG: staged candidates carry their VALUE (int2, the SPLIT-slab form)
    // on every variant whose occupancy affords the doubled staging: the P5
    // crossing-bin emit and -- critically for the pro_128k degen row -- the
    // 8-level staged-source narrowing lose their per-candidate scattered
    // X[id] re-gathers.  256-thread CTAs at 4/SM keep the packed form (the
    // doubled staging would blow the carveout).
    constexpr bool VSTG = SPLIT || (BLK >= 512);
    __shared__ __align__(16) uint32_t hist[HB];   // scan_cross0 reads it as uint4
    extern __shared__ __align__(16) unsigned char smraw[];
    int*      cbuf = (int*)smraw;               // SCPB+4  (packed bin|idx)
    int2*     cbuf2 = (int2*)smraw;             // SCPB+4  (value,idx) on split
    // Crossing-bin candidates live as ONE 64-bit word (key<<32 | index): the
    // O(mc^2) exact rank then needs a single unsigned compare per pair instead
    // of the (v>u)||(v==u&&j<i) triple -- indices are unique, so the packed word
    // is already a strict total order and the tie-break is free.  A ulonglong2
    // read brings two candidates per 16B shared access.  Same 8 bytes/candidate.
    unsigned long long* ck64 =
        (unsigned long long*)(cbuf + (VSTG ? 2 * (SCPB + 4) : (SCPB + 4)));  // CMPB+1

    __shared__ uint32_t ws[BLK / 32], wmn[BLK / 32], wmx[BLK / 32];
    __shared__ unsigned s_bufn, s_o1, s_o2, s_base;
    __shared__ unsigned long long s_pk;
    __shared__ int s_B, s_m, s_above, s_tot, s_B2, s_B3;
    __shared__ float s_TSH;
    __shared__ uint32_t s_kmin, s_kmax;

    // 2D grid: (part, row).  A runtime bid/R + bid%R pair cost ~50 integer
    // instructions per thread -- 10% of the whole fixed stream at small b.
    const int row  = (int)blockIdx.y;
    const int part = SPLIT ? (int)blockIdx.x : 0;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    const float*  X  = logits + (size_t)row * (size_t)npad;
    const float4* X4 = (const float4*)X;
    const int*    P  = pre_idx + (size_t)row * (size_t)k;
    int*          O  = out + (size_t)row * (size_t)k;

    const int n4 = n >> 2;
    const int c0 = SPLIT ? (part * Q) : 0;
    int c1 = SPLIT ? (c0 + Q) : n4; if (c1 > n4) c1 = n4;
    const int tail0 = n4 << 2;
    const int tailn = (part == 0) ? (n - tail0) : 0;

    if (tid == 0) { s_bufn = 0u; s_B2 = -1; s_B3 = -1; }
    // NBS <= BLK on every streaming dispatch, so the clear is ONE predicated
    // store, not a loop with its compare, its add and its backward branch.
    if constexpr (HB <= BLK) { if (tid < HB) hist[tid] = 0u; }
    else                     { for (int i = tid; i < HB; i += BLK) hist[i] = 0u; }

    // ---------------- P1: sample prefetch (hint gather is LAZY) -------------
    // LAZY HINT (both paths): on rows that land attempt 0 -- every healthy row
    // under the wide-margin rung -- GMIN/GMAX are never consumed: the accept
    // test (tot >= k) is exact on its own and the histogram window comes from
    // the sample (HIC).  The two dependent hint round trips leave the hot
    // path entirely and are paid only on an attempt failure or a degenerate
    // sample (GVR_GATHER_HINT).  r3: the SPLIT path drops its eager gather
    // too -- at R=64 the hop-2 sectors were 2x the whole row's bytes.  SPLIT
    // has no retry, but it does not need one for exactness: the last CTA
    // verifies tot >= k against the merged slab and falls back to the exact
    // full-row key-space narrowing (degen) when the rung overshot; the rung's
    // 3.5x aim margin makes that a many-sigma event, and a degenerate sample
    // still gathers the hint up front and runs at the GMIN floor.
    float4 sa, sb;
    const bool shas = (tid < SMP);
    if (shas) { int p4 = tid * SS2 * 2; sa = __ldg(X4 + p4); sb = __ldg(X4 + p4 + 1); }

    // The row pass loads depend only on the SLICE, never on the rung, so
    // iteration 0 is hoisted here: it issues alongside the hint's first hop and
    // lands under the two dependent hint round trips plus the whole sample
    // phase.  Only the MINB==1 variants have the register budget to hold it.
    // __launch_bounds__(1024,1) still caps the thread at 64 registers, of which
    // the row pass already needs 4*U for its in-flight float4, so cap the
    // prefetch DEPTH at 4 (16 registers) instead of refusing it whenever the
    // slice is deep.  At U<=2 that hoists the WHOLE row pass; at U=8 it hoists
    // half of iteration 0, which then rides under both dependent hint round
    // trips and the entire quantile-sample phase.  The 4-CTA/SM wide-batch
    // variant has the same 64 registers spread over 4x the threads and measured
    // neutral-to-negative at depth 2, so it keeps no prefetch.
    // r4: the 512-thread midband variant (MINB=2) has the same 64-register
    // budget per thread as the 1024-thread variant -- give it the same
    // prefetch depth.  The 4-CTA/SM 256-thread variant keeps none (measured
    // neutral-to-negative there).
    constexpr int PFD = (MINB <= 2) ? (U < 4 ? U : 4) : 0;
    constexpr bool PF = (PFD > 0);
    const int lim4 = (npad >> 2) - 1;
    float4 pf[PF ? PFD : 1];
    // r4 PRIME-LATE (see gvr_clus): priming here puts 64KB/CTA of streaming
    // row-buffer hits in front of the scattered rung-critical sample in the
    // machine-wide DRAM queue.  The prime moves below the sample-consuming
    // reduce barrier; DRAM stays idle until every sample has landed.

    // ---------------- P2: quantile rung from the sample ---------------------
    // The sample supplies its OWN histogram range.  [GMIN,GMAX] is finer when
    // the hint is HOT, but on a COLD hint (hit rates in these rows go down to
    // 0.02) GMIN already collapses to ~the row minimum -- i.e. exactly this
    // range -- and those rows are handled fine today, so the resolution this
    // gives up is inside the regime the kernel is already tuned for.  What it
    // buys is that the rung stops depending on the hint at all.
    float smn = INFINITY, smx = -INFINITY;
    if (shas) {
        float e[8] = { sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w };
#pragma unroll
        for (int t = 0; t < 8; t++) { smn = fminf(smn, e[t]); smx = fmaxf(smx, e[t]); }
    }
    for (int j = tid + BLK; j < SMP; j += BLK) {
        int p4 = j * SS2 * 2;
        float4 u1 = __ldg(X4 + p4), u2 = __ldg(X4 + p4 + 1);
        float e[8] = { u1.x, u1.y, u1.z, u1.w, u2.x, u2.y, u2.z, u2.w };
#pragma unroll
        for (int t = 0; t < 8; t++) { smn = fminf(smn, e[t]); smx = fmaxf(smx, e[t]); }
    }
    {
        const uint32_t a0 = warp_min_u32(fkey(smn)), c0 = warp_max_u32(fkey(smx));
        if (lane == 0) { wmn[tid >> 5] = a0; wmx[tid >> 5] = c0; }
    }
    __syncthreads();                 // also publishes the hist clear + s_bufn
    if constexpr (PF) {
        if ((c1 - c0) >= BLK * U) {
#pragma unroll
            for (int u = 0; u < PFD; u++) pf[u] = X4[c0 + tid + u * BLK];
        } else {
#pragma unroll
            for (int u = 0; u < PFD; u++) { int i = c0 + tid + u * BLK; pf[u] = X4[(i < c1) ? i : lim4]; }
        }
        // r5 (a005): L2-prefetch iteration-0's u >= PFD slots -- the first
        // naked loads consumed right after the rung lands (gvr_clus has had
        // this deep-gated since r4; gvr_main never did).  Gate at a slice of
        // >= 2 full chunks so the fill is amortized by a long row pass, AND
        // at a FAT sample (SMP >= 160): the ~64KB/CTA fill only pays when the
        // head's random-page sample service is long enough to hide it
        // (measured: flash_256k SMP=256 15.07->14.18, pro_128k SMP=186 helps,
        // pro_256k_bs128 SMP=128 15.62->15.71 -- the fill spills into the
        // row pass on short heads).  A formula of measured sample statistics.
        if ((c1 - c0) >= 2 * BLK * U && SMP >= 160) {
#pragma unroll
            for (int u = PFD; u < U; u++)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(X4 + c0 + tid + u * BLK));
        }
    } else if constexpr (!SPLIT) {
        // r5 (a005): the PFD==0 wide-batch variant (BLK=256, 4 CTAs/SM) has NO
        // register prefetch at all -- its whole first iteration sits naked
        // behind the T-chain.  A pure L2 hint here costs no registers; the
        // fill turns iteration-0's DRAM misses into L2 hits so the row pass's
        // MSHRs recycle ~3x faster (the flash_256k mechanism).  Issued at the
        // prime-late position (after the sample-consume barrier) so it cannot
        // delay the rung-critical sample randoms.  Full-iteration slices only.
        // (Extending to iteration 1 measured 17.95 vs 17.23us on pro_64k --
        // the doubled fill floods the DRAM queue in front of the OTHER
        // co-resident CTAs' still-pending sample randoms.  it0 only.)
        if ((c1 - c0) >= BLK * U) {
#pragma unroll
            for (int u = 0; u < U; u++)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(X4 + c0 + tid + u * BLK));
        } else if (SMP > 0) {
            // knife4-L1: PARTIAL slice on a SAMPLED-RUNG dispatch (SMP > 0 --
            // i.e. exactly the small_dense k>1024 corner: n4 ~ 1027 < BLK*U;
            // large-n sampled rows are never partial at this variant).  Here
            // the whole row read used to sit naked behind the T-chain
            // (sample randoms + reduce barriers + warp-0 rung scan) because
            // the full-iteration gate above never fires.  The fill is
            // bounded by the slice itself (<= 16KB at 4k) -- it is exactly
            // this CTA's own iteration-0 loads, so the pro_64k "doubled
            // fill floods the queue" hazard does not apply.  Clamped
            // addressing mirrors the register-prefetch tail idiom.  SMP==0
            // (eager-hint small rows, all k<=1024 V4 dispatches) keeps the
            // old no-prefetch behavior: their head is the hint gather, not
            // the sample chain, and the measured effect there was a small
            // regression (flash_4k +1-3%) -- zero-harm gate.
            const int lim4p = (npad >> 2) - 1;
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int i = c0 + tid + u * BLK;
                asm volatile("prefetch.global.L2 [%0];"
                             :: "l"(X4 + ((i < c1) ? i : lim4p)));
            }
        }
    }
    float SMIN, SMAX;
    {
        constexpr int NW = BLK / 32;
        uint32_t a = (lane < NW) ? wmn[lane] : 0xffffffffu;
        uint32_t c = (lane < NW) ? wmx[lane] : 0u;
        SMIN = invkey(warp_min_u32(a)); SMAX = invkey(warp_max_u32(c));
    }

    // Sentinel bracket until a gather runs: the non-split histogram top then
    // rests on HIC (which fires on every healthy row); an untightened +3e38
    // window is only ever LIVE on degenerate rows, whose crossing bin goes
    // through the exact refine/degen machinery anyway.
    float GMIN = -3.0e38f, GMAX = 3.0e38f;
    float T = -INFINITY;
    float HIC = -INFINITY;          // tightened top of the row-histogram range
    float w = 0.f;
    const bool sok = (SMP > 0) && (SMAX > SMIN);
    {
        if (sok) {
            w = (SMAX - SMIN) * (1.0f / (float)NBS);
            const float sc = 1.0f / w;
            // every sample value is >= SMIN by construction, so the sign test
            // the [GMIN,GMAX] form needed disappears with it.
            if (shas) {
                float e[8] = { sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w };
#pragma unroll
                for (int t = 0; t < 8; t++)
                    atomicAdd(&hist[min(__float2int_rz((e[t] - SMIN) * sc), NBS - 1)], 1u);
            }
            for (int j = tid + BLK; j < SMP; j += BLK) {
                int p4 = j * SS2 * 2;
                float4 u1 = __ldg(X4 + p4), u2 = __ldg(X4 + p4 + 1);
                float e[8] = { u1.x, u1.y, u1.z, u1.w, u2.x, u2.y, u2.z, u2.w };
#pragma unroll
                for (int t = 0; t < 8; t++)
                    atomicAdd(&hist[min(__float2int_rz((e[t] - SMIN) * sc), NBS - 1)], 1u);
            }
        }
        __syncthreads();
        // ZERO=true: the sample's cursors are never used, so leaving the bins
        // cleared hands the row pass a zeroed histogram for free -- one clear
        // pass and one barrier removed from the hot path.
        // knife5: SPLIT needs s_B3 too (the TSH-floor staging below), but ONLY
        // on the veto fall-through band -- compiling the third crossing into
        // every SPLIT dispatch cost a REAL 6-12% at 1024k/v32 BS1/2 (64-reg
        // wall: same REG count, fatter live set, inflated T-chain; confirmed
        // by arm-order swap, self-vs-self probe clean).  The gate is grid-
        // uniform, so branch between two instantiations: the ungated branch
        // is the exact pre-knife5 scan.
        if (SPLIT && gridDim.y > 15 && k <= 1024 && (n >> 2) <= 32768)
            scan_cross0<NBS, true, true, true>(hist, TGT, tid, lane, &s_B, &s_m, &s_above, &s_tot,
                                          TGT2, &s_B2, 2 * TGT, &s_B3);
        else
            scan_cross0<NBS, true, true, SHD>(hist, TGT, tid, lane, &s_B, &s_m, &s_above, &s_tot,
                                          TGT2, &s_B2, 2 * TGT, &s_B3);
        // The eager-gather consume block used to carry the barrier that
        // publishes warp 0's scan (s_B/s_tot/s_B2/s_B3); without it every
        // thread outside warp 0 races the scan.  Pay it explicitly.
        __syncthreads();
        if (sok && s_tot >= TGT) T = fmaf((float)s_B, w, SMIN);
        const float Trung = T;   // sample rung BEFORE the hint floor
        if (!(T > -INFINITY)) {
            // Degenerate sample (or sample skipped): the rung never formed, so
            // pay the hint round trips NOW -- rare, and exactness needs a
            // finite T with count(>=T) >= k, which only GMIN guarantees here.
            // (On SPLIT every CTA computes the identical GMIN, so the rung
            // stays row-consistent.)
            GVR_GATHER_HINT(GMIN, GMAX, KPT);
            T = GMIN;
        }
        // The row histogram used to span [T, GMAX], but GMAX is the MAXIMUM hint
        // value -- one outlier that stretches the range so far that all ~aim
        // candidates collapse into the bottom handful of bins.  NCU on
        // pro_64k bs512 charged 12% of ALL executed instructions to the O(mc^2)
        // crossing-bin rank for exactly that reason (mc pinned at the QUADC=96
        // cap).  The SAME sample scan that produced the rung also gives, for
        // free, the bin where the count reaches k -- i.e. an estimate of the
        // k-th value.  Spanning the row histogram over [T, T + 3*(Tk - T)]
        // instead puts rank k at bin ~85 of NBS with everything above the top
        // (all of it strictly above the k-th value, hence a definite winner)
        // clamped into the last bin.  Bins stay monotone in value, so the
        // cursor/emit/refine logic is untouched and remains exact.
        if (sok && s_tot >= TGT && s_B2 >= 0) {
            float Tk = fmaf((float)s_B2, w, SMIN);
            // 4x the (rung -> rank-k) distance puts rank k at bin ~NBS/4 with
            // room for the sample's own error; never tighter than 8 sample bins.
            // ANCHOR ON THE SAMPLE RUNG, not the hint-floored T: when GMIN lifts
            // T above the rung, (Tk - T) collapses and the 4x window tops out
            // BELOW the true rank-k value -- the winners then pile into the
            // clamp bin, m > CMPB, and the full-row degen runs (measured 0.25x
            // on pro_128k bs256).  anc == T whenever the floor did not fire, so
            // every healthy dispatch is bit-identical.  The hi-tighten guard
            // below already ignores HIC when it lands at or below T.
            // Width from the sample scale, base at T: bins below T hold no
            // candidates (the emit test is x >= T), so basing the window at the
            // rung wastes (GMIN - rung)*SC bins and coarsens the live range --
            // measured +32% on flash_256k_L06 (hot hint, GMIN >> rung).
            const float anc = SPLIT ? T : fminf(T, Trung);
            // fire even when Tk <= anc: rank-aim and rank-k in the SAME sample
            // bin is the DENSEST case, where leaving hi at GMAX quantises the
            // whole candidate set into a handful of bins and pushes the
            // crossing bin past CMPB into the degenerate fallback.
            HIC = fmaxf(fmaf(4.0f, fmaxf(Tk - anc, 0.0f), T), fmaf(8.0f, w, T));
        }
    }

    // ---- ladder floor: sample rank-(2*TGT) estimate, clamped to GMIN --------
    // Armed only when the sample genuinely crossed 2*TGT, so TSH is never the
    // bin-0 fallback that would put the floor at the row minimum.
    // r4 (a000): register-free -- the floor lives in ONE shared word instead
    // of a float+bool pair held across the whole row pass at the 64-reg wall.
    // -INFINITY means "not armed"; read only on the cold retry path.
    // knife5: SPLIT arms it too -- not for a retry (SPLIT has none) but for
    // the TSH-floor staging below.
    if constexpr (SHD || SPLIT) {
        if (!SPLIT || (gridDim.y > 15 && k <= 1024 && (n >> 2) <= 32768)) {
            if (tid == 0) {
                float t5 = -INFINITY;
                if (sok && s_tot >= 2 * TGT && s_B3 >= 0 && T > GMIN) {
                    float T3 = fmaf((float)s_B3, w, SMIN);
                    if (T3 < T) t5 = T3;
                }
                s_TSH = t5;
            }
        }
    }
    // knife5 TSH-FLOOR STAGING (SPLIT only): SPLIT has no retry ladder, so a
    // rung overshoot (count(>=T) < k) used to hand the LAST CTA a single-CTA
    // whole-row key-space narrowing -- 3.7-3.9x on the retry-heavy rows that
    // killed cand_L2v (pro_512k_L46/L52 x BS16: 84us vs 22.5us).  Pay the
    // ladder's insurance in SPACE instead of a second pass: lower the staging
    // threshold from the rung T to the sample's rank-(2*TGT) floor TSH.  The
    // staged population goes ~aim -> ~2*aim (the same bound the non-split TSH
    // retry's "bounded emit" relies on) -- ~+11KB of slab traffic against a
    // 512KB row read -- and the merged histogram then contains the k-crossing
    // whenever count(>=TSH) >= k, which is exactly the event the non-split
    // ladder's second rung catches.  TSH misses too -> GMIN/degen backstop
    // unchanged (rare^2).  The window base, mask predicate, histogram, cursor
    // and refine all derive from T, so one lowering does the whole job; the
    // machinery is threshold-agnostic (the GMIN-floor flood runs this same
    // shape today).
    // Scope: ONLY the veto fall-through population (b > 15 slab = the 512k
    // b16 family; every b <= 15 / KBIG / 1M-deep slab user predates the veto,
    // was full-grid green in wbp3, and measured a real always-on tax here
    // (v32_256k 0.64-0.80: doubled staging overflows SCPB into per-slice
    // re-sweeps; 1M BS1/8 0.86-0.94).  gridDim.y == b: shape-blind gate.
    // ... and the fall-through population is exactly "would have fit the
    // clustered register path" (k <= BLKC && n4 <= 8*BLKC*4) at b > 15:
    // 1M rows (n4 > 32768, no cluster option ever) and KBIG v32 rows
    // (k > 1024) are pre-veto slab natives, wbp3-green unarmed, and
    // measured the always-on tax hardest (v32 b16/32 0.64-0.76: doubled
    // staging overflows SCPB into per-slice re-sweeps).  The gate is
    // grid-uniform, so the barrier lives inside it: ungated dispatches
    // execute nothing at all here.
    if constexpr (SPLIT) {
        if (gridDim.y > 15 && k <= 1024 && (n >> 2) <= 32768) {
            __syncthreads();
            const float t5s = s_TSH;
            if (t5s > -INFINITY && t5s < T) T = t5s;
        }
    }

    // ================= attempt loop (guess -> verify) =========================
    int listN = 0, above = 0, m = 0, need = 0, B = 0;
    float SC = 1.f, TF = T;
    bool complete = false, valid = false, fromg = false;
    constexpr int NATT = SPLIT ? 1 : 3;
#pragma unroll 1
    for (int att = 0; att < NATT; ++att) {
        // att 0 inherits a cleared histogram and s_bufn from the guess phase.
        if (att) {
            // EXACTNESS: Re-prime pf[] -- it holds stale data from the previous
            // attempt's roll; a retry classifying iteration 0 against it silently
            // drops true winners. Re-prime; att 0 hot path untouched.
            if constexpr (PF) {
                if ((c1 - c0) >= BLK * U) {
#pragma unroll
                    for (int u = 0; u < PFD; u++) pf[u] = X4[c0 + tid + u * BLK];
                } else {
#pragma unroll
                    for (int u = 0; u < PFD; u++) { int i = c0 + tid + u * BLK; pf[u] = X4[(i < c1) ? i : lim4]; }
                }
            }
            if constexpr (NBS <= BLK) { if (tid < NBS) hist[tid] = 0u; }
            else                      { for (int i = tid; i < NBS; i += BLK) hist[i] = 0u; }
            if (tid == 0) s_bufn = 0u;
            __syncthreads();
        }

        TF = T;
        float hi = fmaxf(GMAX, T);
        if (HIC > T && HIC < hi) hi = HIC;
        float WD = (hi - T) * (1.0f / (float)NBS);
        if (!(WD > 0.f)) WD = 1e-30f;
        SC = 1.0f / WD;

        // ---- row pass.  Every load is UNCONDITIONAL (out-of-slice lanes read
        // the padded tail float4 and get an INFINITY threshold instead), so all
        // U loads of an iteration issue before any is consumed -- the row pass
        // is long-scoreboard bound, so memory-level parallelism beats
        // instruction count.  The U*4-bit predicate mask then amortises the slot
        // reservation (one shared atomic per thread) and the __ffs walk.
        {
            // A slice whose length is an exact multiple of BLK*U needs NO
            // in-slice test at all: peel the (at most one) partial iteration out
            // so the FULL iterations lose the per-float4 bounds ISETP, the
            // clamped-address SEL and the INFINITY-threshold SEL -- three
            // instructions per float4 on the two lines NCU charged 13% of the
            // streaming kernel to.  Every R==1 shape lands exactly here.
            const int span = c1 - c0, step = BLK * U;
            const int nFull = (span > 0) ? (span / step) : 0;
            const int rem   = (span > 0) ? (span - nFull * step) : 0;
            const int nIt   = nFull + (rem > 0 ? 1 : 0);
            for (int it = 0; it < nIt; ++it) {
                const int i0 = c0 + it * step + tid;
                unsigned M = 0u;
                if (it < nFull) {
#pragma unroll
                    for (int u = 0; u < U; u++) {
                        const int i = i0 + u * BLK;
                        float4 v;
                        if constexpr (PF) v = (u < PFD) ? pf[u < PFD ? u : 0] : X4[i];
                        else              v = X4[i];
                        if (v.x >= TF) M |= (1u << (u * 4 + 0));
                        if (v.y >= TF) M |= (1u << (u * 4 + 1));
                        if (v.z >= TF) M |= (1u << (u * 4 + 2));
                        if (v.w >= TF) M |= (1u << (u * 4 + 3));

                    }
                } else {
#pragma unroll
                for (int u = 0; u < U; u++) {
                    const int i = i0 + u * BLK;
                    const bool ok = (i < c1);
                    float4 v;
                    if constexpr (PF) v = (u < PFD) ? pf[u < PFD ? u : 0] : X4[ok ? i : lim4];
                    else              v = X4[ok ? i : lim4];
                    /* prod-fix: the old TFI=INFINITY guard let a +inf PADDING
                       value at the clamp slot pass `>=INFINITY` and stage an
                       out-of-slice index (lim4 points into [n, npad)).  Gate
                       the mask bits on ok directly; partial iteration only. */
                    if (ok) {
                        if (v.x >= TF) M |= (1u << (u * 4 + 0));
                        if (v.y >= TF) M |= (1u << (u * 4 + 1));
                        if (v.z >= TF) M |= (1u << (u * 4 + 2));
                        if (v.w >= TF) M |= (1u << (u * 4 + 3));
                    }
                }
                }
                // ROLL THE PREFETCH FORWARD.  The head of iteration it+1 does
                // not depend on anything this iteration produces, so issue it
                // before the slot reservation and the survivor walk -- the walk
                // is a chain of scattered dependent reloads that NCU charges
                // 18.7% of all stalls to, and it now covers a full DRAM round
                // trip for the next iteration instead of leaving it exposed.
                // On the LAST iteration every rolled load clamps out of slice
                // and is thrown away, but it still pays the full (VIADD, ISETP,
                // SEL, IMAD.WIDE, LDG) quad per float4 -- NCU charges 4.2% of
                // the whole kernel to this line, and on the nFull==2 shapes that
                // dominate the streaming path HALF of it is this dead roll.
                if constexpr (PF) if (it + 1 < nIt) {
                    const int j0 = i0 + step;
                    // SASS: the (j < c1) clamp turns ONE IMAD.WIDE + PFD LDG.128
                    // with immediate offsets into PFD x (VIADD, ISETP, SEL,
                    // IMAD.WIDE, LDG) -- 19 instructions instead of 5, and NCU
                    // charges 4.1% of the whole kernel to this single line.  The
                    // NEXT iteration is fully in-slice whenever it+1 < nFull, and
                    // that test is warp-uniform, so peel it.
                    if (it + 1 < nFull) {
#pragma unroll
                        for (int u = 0; u < PFD; u++) pf[u] = X4[j0 + u * BLK];
                    } else {
#pragma unroll
                        for (int u = 0; u < PFD; u++) { const int j = j0 + u * BLK; pf[u] = X4[(j < c1) ? j : lim4]; }
                    }
                }
                // Warp-aggregated slot reservation.  Every thread of every CTA
                // hammers the SAME shared word, and shared atomics to one
                // address serialise -- at U=8 roughly half the lanes of a warp
                // have a survivor, so this was ~15 serialised RMWs per warp per
                // iteration.  A 5-step shuffle scan turns it into one.
                const int cnt = __popc(M);
                int inc = cnt;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) { int z = __shfl_up_sync(FULLM, inc, o); if (lane >= o) inc += z; }
                unsigned bpos = 0u;
                if (lane == 31 && inc) bpos = atomicAdd(&s_bufn, (unsigned)inc);
                unsigned pos = __shfl_sync(FULLM, bpos, 31) + (unsigned)(inc - cnt);
                // MEASURED: holding the U float4 live through the walk so the
                // value comes straight from the mask's own register costs +18%
                // -- every streaming variant is already at the 64-register wall
                // that __launch_bounds__ implies, and the array spills.  The
                // reload hits L1, so keep it; just store the VALUE rather than
                // the bin on the split path, which is free here and saves the
                // last CTA a scattered re-gather of the whole crossing bin.
                // Software-pipelined: the NEXT survivor's scattered reload is
                // issued before the current one is consumed.  NCU puts 42% of
                // all stalls on long-scoreboard and this walk was a serial chain
                // of dependent L1/L2 loads -- one iteration of lookahead turns it
                // into two in flight for two extra registers.  The slot store is
                // branchless (a trash slot at cbuf[SCPB]) so the overflow test
                // costs an IMNMX instead of a BSSY/BRA/BSYNC triple.
#define GVR_EMITC(xv_, idx_)                                                       \
                do {                                                               \
                    if constexpr (!SPLIT) {                                        \
                        unsigned bn = min(__float2uint_rz(((xv_) - TF) * SC),      \
                                          (unsigned)(NBS - 1));                    \
                        atomicAdd(&hist[bn], 1u);                                  \
                        if constexpr (!VSTG)                                       \
                            cbuf[min(pos, (unsigned)SCPB)] =                       \
                                (int)((bn << IDXB) | (unsigned)(idx_));            \
                    }                                                              \
                    if constexpr (VSTG)                                            \
                        cbuf2[min(pos, (unsigned)SCPB)] =                          \
                            make_int2(__float_as_int(xv_), (idx_));                \
                    pos++;                                                         \
                } while (0)
                if (M) {
                    int bp = __ffs(M) - 1;
                    M &= (M - 1u);
                    int idx = ((i0 + (bp >> 2) * BLK) << 2) + (bp & 3);
                    float xv = X[idx];
                    while (M) {
                        int bp2 = __ffs(M) - 1;
                        M &= (M - 1u);
                        int idx2 = ((i0 + (bp2 >> 2) * BLK) << 2) + (bp2 & 3);
                        float xv2 = X[idx2];
                        GVR_EMITC(xv, idx);
                        idx = idx2; xv = xv2;
                    }
                    GVR_EMITC(xv, idx);
                }
            }
            for (int i = tid; i < tailn; i += BLK) {
                float x = X[tail0 + i];
                if (x >= TF) {
                    unsigned pos = atomicAdd(&s_bufn, 1u);
                    GVR_EMITC(x, tail0 + i);
                }
            }
#undef GVR_EMITC
        }
        __syncthreads();

        const int myn = (int)s_bufn;

        if constexpr (SPLIT) {
            // ---- SLAB HAND-OFF ------------------------------------------------
            // Every CTA of the row derives the IDENTICAL rung T (the sample spans
            // the whole row), so its bin index is globally meaningful.  Each CTA
            // therefore just appends its packed (bin,index) words to one global
            // slab and drops out; the CTA that takes the last arrival ticket owns
            // the whole selection.  That replaces a global histogram merge, a
            // release/acquire grid spin (whose ld.acquire emits CCTL.IVALL on
            // every poll, flushing the SM's L1) and a merged-histogram read-back
            // with two global atomics and one coalesced slab read.
            if (tid == 0) s_base = atomicAdd(&g_off[row], (unsigned)myn);
            __syncthreads();
            const unsigned base = s_base;
            if (myn <= SCPB) {
                for (int i = tid; i < myn; i += BLK) {
                    unsigned p = base + (unsigned)i;
                    if (p < (unsigned)GCAP) g_buf[row][p] = cbuf2[i];
                }
            } else {
                // collect overflowed: re-sweep this CTA's own slice straight into
                // the slab (rare; the shared staging buffer is 8x k).
                if (tid == 0) s_bufn = 0u;
                __syncthreads();
                // EXACTNESS: the tail scalars live at [tail0, tail0+tailn), not
                // at c1*4 (part 0's slice end is NOT the row end when R > 1);
                // the old (c1<<2)+tailn bound double-staged the partner's first
                // elements and missed the true tail.
                const int lo2 = c0 << 2, hi2 = (c1 << 2);
                for (int i = lo2 + tid; i < hi2; i += BLK) {
                    float x = X[i];
                    if (x >= TF) {
                        unsigned p = base + atomicAdd(&s_bufn, 1u);
                        if (p < (unsigned)GCAP) g_buf[row][p] = make_int2(__float_as_int(x), i);
                    }
                }
                for (int i = tid; i < tailn; i += BLK) {
                    float x = X[tail0 + i];
                    if (x >= TF) {
                        unsigned p = base + atomicAdd(&s_bufn, 1u);
                        if (p < (unsigned)GCAP) g_buf[row][p] = make_int2(__float_as_int(x), tail0 + i);
                    }
                }
            }
            __syncthreads();
            // one packed RMW carries both the arrival ticket and the running
            // candidate total, so the last CTA learns the total for free.
            if (tid == 0) { __threadfence();
                            s_pk = atomicAdd(&g_don[row], (1ull << 32) | (unsigned long long)(unsigned)myn); }
            __syncthreads();
            const unsigned long long pk = s_pk;
            if ((unsigned)(pk >> 32) != (unsigned)(R - 1)) return;
            /* B2b: acquire side of the slab handoff (pairs with the
               publish-side __threadfence above).  Practically covered by
               __ldcg (L2-direct) + the barrier, but the PTX memory model
               wants an explicit fence; one per last CTA is free. */
            __threadfence();
            if (tid == 0) { g_off[row] = 0u; g_don[row] = 0ull; }
            const unsigned total = (unsigned)(pk & 0xffffffffull) + (unsigned)myn;
            if (total <= (unsigned)GCAP) {
                listN = (int)total;
                fromg = (total > (unsigned)SCPB);
                // one pass: the slab word is histogrammed as it lands and staged
                // in shared for the emit.  hist is still the zeros the guess
                // phase's ZERO scan left behind, so no clear and no barrier.
                for (int i = tid; i < listN; i += BLK) {
                    int2 w = __ldcg(&g_buf[row][i]);
                    if (!fromg) cbuf2[i] = w;
                    unsigned bn = (unsigned)min(__float2int_rz((__int_as_float(w.x) - TF) * SC), NBS - 1);
                    atomicAdd(&hist[bn], 1u);
                }
                __syncthreads();
                scan_cross0<NBS, false>(hist, k, tid, lane, &s_B, &s_m, &s_above, &s_tot);
                __syncthreads();
                if (s_tot >= k) { valid = true; complete = true;
                                  above = s_above; m = s_m; need = k - s_above; B = s_B; }
            }
            break;
        } else {
            scan_cross0<NBS, false>(hist, k, tid, lane, &s_B, &s_m, &s_above, &s_tot);
            __syncthreads();
            const int tot = s_tot;
            if (tot >= k) {
                valid = true; complete = (myn <= SCPB); listN = myn;
                above = s_above; m = s_m; need = k - s_above; B = s_B;
                break;
            }
            if (att == NATT - 1) break;                  // ladder exhausted
            // r4 (a000) ladder: att0 miss -> TSH with NO hint gather at all --
            // if TSH <= GMIN the accept is guaranteed (count(>=GMIN) >= k by
            // construction, so count(>=TSH) >= k), and a TSH miss pays the
            // gather on the way to the GMIN backstop.  HIC is armed whenever
            // s_TSH is, so the retry window stays [TSH, HIC).
            if constexpr (SHD) {
                if (att == 0) {
                    float T5 = s_TSH;
                    if (T5 > -INFINITY && T5 < TF) { T = T5; __syncthreads(); continue; }
                }
            }
            // LAZY GATHER: pays the two hint round trips here, off the hot
            // path.  GMIN floors the descent and is the exactness backstop
            // (count(>=GMIN) >= k by construction).
            if (GMIN == -3.0e38f) GVR_GATHER_HINT(GMIN, GMAX, KPT);
            if (!(T > GMIN)) break;                      // rung floor reached
            T = GMIN;
            __syncthreads();
        }
    }

    const bool whole = valid && (need >= m);
    const int  lim1  = whole ? (above + m) : above;
    const bool degen = (!valid) || (m > CMPB);
    const int  mc    = degen ? 0 : m;

    // ---------------- P5: place candidates through the bin cursors ----------
    if (!degen) {
        if (complete) {
            for (int i = tid; i < listN; i += BLK) {
                int id, bn; float xv = 0.f;
                if constexpr (VSTG) {
                    int2 w = (SPLIT && fromg) ? __ldcg(&g_buf[row][i]) : cbuf2[i];
                    xv = __int_as_float(w.x); id = w.y;
                    bn = (int)min(__float2int_rz((xv - TF) * SC), NBS - 1);
                } else {
                    unsigned wpk = (unsigned)cbuf[i];
                    id = (int)(wpk & IDXM); bn = (int)(wpk >> IDXB);
                }
                if (bn >= B) {
                    unsigned p = atomicAdd(&hist[bn], 1u);
                    if (p < (unsigned)lim1) O[p] = id;
                    else if (!whole) { unsigned q2 = p - (unsigned)above;
                                       if (q2 < (unsigned)CMPB) {
                                           // value came with the candidate: no scattered re-gather
                                           ck64[q2] = ((unsigned long long)fkey(VSTG ? xv : X[id]) << 32)
                                                      | (unsigned long long)(unsigned)id; } }
                }
            }
        } else {
            // this CTA's collect overflowed: one direct sweep of ITS OWN slice
            // through the same cursors -- no histogram, no second guess.
            // rare path: keep it scalar so it costs no extra live registers
            // (tail remap: tail scalars live at tail0, not at c1*4 -- see the
            // clus resweep note; only reachable non-SPLIT where c1==n4, but
            // keep the mapping exact anyway)
            const int lo2 = c0 << 2, hi2 = (c1 << 2);
            for (int i0_ = lo2 + tid; i0_ < hi2 + tailn; i0_ += BLK) {
                const int i = (i0_ < hi2) ? i0_ : (tail0 + (i0_ - hi2));
                float x = X[i];
                if (x >= TF) {
                    int bn = (int)min(__float2int_rz((x - TF) * SC), NBS - 1);
                    if (bn >= B) {
                        unsigned p = atomicAdd(&hist[bn], 1u);
                        if (p < (unsigned)lim1) O[p] = i;
                        else if (!whole) { unsigned q2 = p - (unsigned)above;
                                           if (q2 < (unsigned)CMPB) { ck64[q2] = ((unsigned long long)fkey(x) << 32) | (unsigned long long)(unsigned)i; } }
                    }
                }
            }
        }
    }

    if (!degen) {
        if (whole) return;
        __syncthreads();

        if (mc <= QUADC_CLUS) {   /* r3: same 288 gate as clus -- measured positive there */
            // exact selection by counting strictly-greater keys (index-tie-broken)
            // two candidates per 16B shared read: the index tie-break rides
            // in the low half of the packed key so ONE unsigned compare decides.
            const int mc2 = mc & ~1;
            for (int i = tid; i < mc; i += BLK) {
                unsigned long long u = ck64[i];
                int r = 0;
                for (int j = 0; j < mc2; j += 2) {
                    ulonglong2 v = *(const ulonglong2*)(ck64 + j);
                    r += (v.x > u) + (v.y > u);
                }
                if (mc2 < mc) r += (ck64[mc2] > u);
                if (r < need) O[above + r] = (int)(unsigned)u;
            }
            return;
        }
        if (tid == 0) { s_kmin = 0xffffffffu; s_kmax = 0u; }
        // cleared ONCE: every narrowing level's scan leaves the bins zeroed
        for (int i = tid; i < NBS; i += BLK) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < mc; i += BLK) { uint32_t kk = (uint32_t)(ck64[i] >> 32);
                                             atomicMin(&s_kmin, kk); atomicMax(&s_kmax, kk); }
        __syncthreads();
        uint32_t rlo = s_kmin, rhi = s_kmax;
        long long ethr = (long long)rlo; int aboveC = 0, needC = need, mm = mc;
        for (int lev = 0; ; ++lev) {
            if (needC == mm) { ethr = (long long)rlo - 1LL; aboveC += mm; needC = 0; break; }
            if (rlo >= rhi)  { ethr = (long long)rlo; break; }
            if (lev >= 6)    { ethr = (long long)rlo; break; }
            uint32_t d2 = rhi - rlo;
            int b2 = 32 - __clz(d2 | 1u);
            int lb = 0; { int t2 = NBS; while (t2 > 1) { t2 >>= 1; lb++; } }
            int sh2 = (b2 > lb) ? (b2 - lb) : 0;
            for (int i = tid; i < mc; i += BLK) {
                uint32_t u = (uint32_t)(ck64[i] >> 32);
                if (u >= rlo && u <= rhi)
                    atomicAdd(&hist[min((unsigned)((u - rlo) >> sh2), (unsigned)(NBS - 1))], 1u);
            }
            __syncthreads();
            scan_cross0<NBS, true>(hist, needC, tid, lane, &s_B, &s_m, &s_above, &s_tot);
            __syncthreads();
            aboveC += s_above; needC -= s_above; mm = s_m;
            uint32_t nlo = rlo + ((uint32_t)s_B << sh2);
            rhi = (s_B == NBS - 1) ? rhi : (nlo + ((1u << sh2) - 1u));
            rlo = nlo;
        }
        if (tid == 0) { s_o1 = 0u; s_o2 = 0u; }
        __syncthreads();
        int it2 = (mc + BLK - 1) / BLK;
        for (int it = 0; it < it2; ++it) {
            int i = it * BLK + tid;
            bool val = i < mc;
            unsigned long long w64 = val ? ck64[i] : 0ull;
            uint32_t u = (uint32_t)(w64 >> 32);
            int id = (int)(unsigned)w64;
            bool q1 = val && ((long long)u > ethr);
            bool q2 = val && ((long long)u == ethr);
            unsigned n1 = __ballot_sync(FULLM, q1);
            unsigned n2 = __ballot_sync(FULLM, q2);
            unsigned b1 = 0, b2 = 0;
            if (lane == 0) {
                if (n1) b1 = atomicAdd(&s_o1, (unsigned)__popc(n1));
                if (n2) b2 = atomicAdd(&s_o2, (unsigned)__popc(n2));
            }
            b1 = __shfl_sync(FULLM, b1, 0);
            b2 = __shfl_sync(FULLM, b2, 0);
            if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u)); if (p < (unsigned)aboveC) O[above + p] = id; }
            if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u)); if (p < (unsigned)needC) O[above + aboveC + p] = id; }
        }
        return;
    }

    // ---- degenerate: exact key-space narrowing over the STAGED CANDIDATES --
    // A huge crossing bin (m > CMPB) with a healthy accept means the top-k is
    // fully inside cbuf (count(>=T) = tot >= k, all staged): narrowing over the
    // listN L1-hot candidates costs ~listN/n of the full-row passes below
    // (measured 0.25x on pro_128k bs256 where m=1056 > CMPB ran 8 full-row
    // levels plus a full-row emit).
    if (valid && complete) {
        uint32_t rlo = 0u, rhi = 0xffffffffu;
        int above2 = 0, need2 = k, m2 = listN;
        long long ethr = 0; bool tieM = true;
        int lb = 0; { int t2 = NBS; while (t2 > 1) { t2 >>= 1; lb++; } }
        for (int i = tid; i < NBS; i += BLK) hist[i] = 0u;
        __syncthreads();
        for (int lev = 0; ; ++lev) {
            if (need2 == m2) { ethr = (long long)rlo - 1LL; above2 += m2; need2 = 0; tieM = false; break; }
            if (rlo >= rhi)  { ethr = (long long)rlo; break; }
            if (lev >= 8)    { ethr = (long long)rlo; break; }
            uint32_t d2 = rhi - rlo;
            int b2 = 32 - __clz(d2 | 1u);
            int sh2 = (b2 > lb) ? (b2 - lb) : 0;
            for (int i = tid; i < listN; i += BLK) {
                uint32_t u;
                if constexpr (VSTG) { int2 wq = (SPLIT && fromg) ? __ldcg(&g_buf[row][i]) : cbuf2[i]; u = fkey(__int_as_float(wq.x)); }
                else                 { u = fkey(X[(unsigned)cbuf[i] & IDXM]); }
                if (u >= rlo && u <= rhi)
                    atomicAdd(&hist[min((unsigned)((u - rlo) >> sh2), (unsigned)(NBS - 1))], 1u);
            }
            __syncthreads();
            scan_cross0<NBS, true>(hist, need2, tid, lane, &s_B, &s_m, &s_above, &s_tot);
            __syncthreads();
            above2 += s_above; need2 -= s_above; m2 = s_m;
            uint32_t nlo = rlo + ((uint32_t)s_B << sh2);
            rhi = (s_B == NBS - 1) ? rhi : (nlo + ((1u << sh2) - 1u));
            rlo = nlo;
        }
        if (tid == 0) { s_o1 = 0u; s_o2 = 0u; }
        __syncthreads();
        int nA = tieM ? above2 : k, nT = tieM ? need2 : 0;
        int it2 = (listN + BLK - 1) / BLK;
        for (int it = 0; it < it2; ++it) {
            int i = it * BLK + tid;
            bool val = i < listN;
            int id = 0; uint32_t u = 0u;
            if (val) {
                if constexpr (VSTG) { int2 wq = (SPLIT && fromg) ? __ldcg(&g_buf[row][i]) : cbuf2[i]; u = fkey(__int_as_float(wq.x)); id = wq.y; }
                else                 { id = (int)((unsigned)cbuf[i] & IDXM); u = fkey(X[id]); }
            }
            bool q1 = val && ((long long)u > ethr);
            bool q2 = val && tieM && ((long long)u == ethr);
            unsigned n1 = __ballot_sync(FULLM, q1);
            unsigned n2 = __ballot_sync(FULLM, q2);
            unsigned b1 = 0, b2 = 0;
            if (lane == 0) {
                if (n1) b1 = atomicAdd(&s_o1, (unsigned)__popc(n1));
                if (n2) b2 = atomicAdd(&s_o2, (unsigned)__popc(n2));
            }
            b1 = __shfl_sync(FULLM, b1, 0);
            b2 = __shfl_sync(FULLM, b2, 0);
            if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u)); if (p < (unsigned)nA) O[p] = id; }
            if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u)); if (p < (unsigned)nT) O[nA + p] = id; }
        }
        return;
    }

    // ---- degenerate: exact key-space narrowing over the whole row ----------
    {
        uint32_t rlo = 0u, rhi = 0xffffffffu;
        int above2 = 0, need2 = k, m2 = n;
        long long ethr = 0; bool tieM = true;
        int lb = 0; { int t2 = NBS; while (t2 > 1) { t2 >>= 1; lb++; } }
        for (int i = tid; i < NBS; i += BLK) hist[i] = 0u;
        __syncthreads();
        for (int lev = 0; ; ++lev) {
            if (need2 == m2) { ethr = (long long)rlo - 1LL; above2 += m2; need2 = 0; tieM = false; break; }
            if (rlo >= rhi)  { ethr = (long long)rlo; break; }
            if (lev >= 8)    { ethr = (long long)rlo; break; }
            uint32_t d2 = rhi - rlo;
            int b2 = 32 - __clz(d2 | 1u);
            int sh2 = (b2 > lb) ? (b2 - lb) : 0;
            for (int i = tid; i < n; i += BLK) {
                uint32_t u = fkey(X[i]);
                if (u >= rlo && u <= rhi)
                    atomicAdd(&hist[min((unsigned)((u - rlo) >> sh2), (unsigned)(NBS - 1))], 1u);
            }
            __syncthreads();
            scan_cross0<NBS, true>(hist, need2, tid, lane, &s_B, &s_m, &s_above, &s_tot);
            __syncthreads();
            above2 += s_above; need2 -= s_above; m2 = s_m;
            uint32_t nlo = rlo + ((uint32_t)s_B << sh2);
            rhi = (s_B == NBS - 1) ? rhi : (nlo + ((1u << sh2) - 1u));
            rlo = nlo;
        }
        if (tid == 0) { s_o1 = 0u; s_o2 = 0u; }
        __syncthreads();
        int nA = tieM ? above2 : k, nT = tieM ? need2 : 0;
        int it2 = (n + BLK - 1) / BLK;
        for (int it = 0; it < it2; ++it) {
            int i = it * BLK + tid;
            bool val = i < n;
            uint32_t u = val ? fkey(X[i]) : 0u;
            bool q1 = val && ((long long)u > ethr);
            bool q2 = val && tieM && ((long long)u == ethr);
            unsigned n1 = __ballot_sync(FULLM, q1);
            unsigned n2 = __ballot_sync(FULLM, q2);
            unsigned b1 = 0, b2 = 0;
            if (lane == 0) {
                if (n1) b1 = atomicAdd(&s_o1, (unsigned)__popc(n1));
                if (n2) b2 = atomicAdd(&s_o2, (unsigned)__popc(n2));
            }
            b1 = __shfl_sync(FULLM, b1, 0);
            b2 = __shfl_sync(FULLM, b2, 0);
            if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u)); if (p < (unsigned)nA) O[p] = i; }
            if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u)); if (p < (unsigned)nT) O[nA + p] = i; }
        }
    }
}

#define LNB 10

/* ---------------------------------------------------------------------------
   Register-resident GVR variant: the whole row lives in registers as RAW
   floats, so after the single global read every phase runs out of registers
   and the histogram bins in FLOAT space (uniform key-space bins collapse the
   whole top-k region into one or two bins on real indexer logits).
--------------------------------------------------------------------------- */
/* DEG: the GUESS STAGE IS PROVABLY USELESS here and is compiled out.  On the
   register path the row is already resident, so the pre_idx prior's only
   economic job is to shrink the histogram -- T=GMIN(hint) bins count(>=GMIN)
   elements instead of all n, i.e. it removes at most n-k bin increments.  Its
   own price is k pre_idx words of DRAM plus k scattered gathers (or, in the
   bitmap form, k shared atomicOr onto (n+31)/32 words -- at n=1027,k=1024 that
   is 1024 atomics serialised onto 34 words).  So once n is already inside the
   candidate budget the guess CANNOT shrink the candidate set below that budget
   and is pure loss.  Then T = the ROW MINIMUM, taken from the values already
   sitting in registers: count(>=T) = n >= k, exact by construction and
   independent of hint quality, and nothing downstream changes.  This is the
   degenerate n<=~k fast path the operator spec allows for. */
template <int BLK, int VPT, int MINB, int KPT, bool CUR, bool DEG, bool IMGF = false, int NBH = NB>
__global__ void __launch_bounds__(BLK, MINB)
gvr_topk_reg(const float* __restrict__ logits, const int* __restrict__ pre_idx,
             int* __restrict__ out, int n, int npad, int k, int CMP, int IMGOFF, int QC) {
    GVR_GDC_WAIT();
    constexpr int S = VPT * 4;
    // log2(NBH) for the key-space narrowing shift
    constexpr int LNBH = (NBH == 256) ? 8 : (NBH == 512) ? 9 : (NBH == 2048) ? 11 : 10;
    extern __shared__ __align__(16) unsigned char smr[];
    uint32_t* hist = (uint32_t*)smr;    // NBH
    uint32_t* ck   = hist + NBH;         // CMP
    int*      ci   = (int*)(ck + CMP);  // CMP
    float*    img  = (float*)((uint32_t*)smr + IMGOFF);   // transient row image

    __shared__ uint32_t ws[BLK / 32], wmn[BLK / 32], wmx[BLK / 32];
    __shared__ int s_B, s_m, s_above, s_tot;
    __shared__ unsigned s_o1, s_oc;

    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const float*  X  = logits + (size_t)row * (size_t)npad;
    const float4* X4 = (const float4*)X;
    const int*    P  = pre_idx + (size_t)row * (size_t)k;
    int*          O  = out + (size_t)row * (size_t)k;

    const int n4 = n >> 2;
    const int ntail = n - (n4 << 2);

    // Hop 1 of the hint chain first: all pre_idx loads (coalesced) are issued
    // before any dependent gather, so both round trips overlap the row load.
    // DEG reads no hint at all, so the whole two-hop chain -- and pre_idx's DRAM
    // traffic -- disappears from the kernel.
    int pv[DEG ? 1 : KPT];
    if constexpr (!DEG) {
#pragma unroll
        for (int t = 0; t < KPT; t++) { int j = tid + t * BLK; pv[t] = (j < k) ? P[j] : -1; }
    }

    // Padding slots are -inf, never -FLT_MAX: that makes the sign of the
    // quantised value a complete validity test, so no per-slot mask is needed.
    float val[S];
    // The dispatcher picks BLK*VPT as the SMALLEST power of two that covers n4,
    // so the common case is an EXACT fit and every resident slot is a real
    // float4.  That test is block-uniform, so peeling it costs one branch and
    // removes a per-float4 bounds ISETP plus the four -INFINITY moves of the
    // else arm -- six instructions per resident float4 on a kernel NCU measures
    // at 482-589 warp-instructions total.
    if (n4 >= BLK * VPT) {
#pragma unroll
        for (int u = 0; u < VPT; u++) {
            float4 v = X4[tid + u * BLK];
            val[4 * u + 0] = v.x; val[4 * u + 1] = v.y;
            val[4 * u + 2] = v.z; val[4 * u + 3] = v.w;
        }
    } else {
#pragma unroll
    for (int u = 0; u < VPT; u++) {
        int i = tid + u * BLK;
        float4 v;
        if (i < n4) v = X4[i];
        else { v.x = -INFINITY; v.y = -INFINITY; v.z = -INFINITY; v.w = -INFINITY; }
        val[4 * u + 0] = v.x; val[4 * u + 1] = v.y;
        val[4 * u + 2] = v.z; val[4 * u + 3] = v.w;
    }
    }
    const int  tidx = (n4 << 2) + tid;
    const float tval = (tid < ntail) ? X[tidx] : -INFINITY;

    // Two ways to bracket the hint.  The direct gather costs KPT scattered 4B
    // loads per thread; the bit-set costs one shared word per float4 plus S
    // masked min/max but no gather at all.  Measured crossover: the bit set
    // wins only when a thread owns few row slots AND several hint slots, i.e.
    // KPT>=2 with one float4 per thread (at S=16 it was ~8% slower).
    constexpr bool USE_BM  = (!DEG && !IMGF && KPT >= 2 && VPT == 1);
    // With several float4 per thread the bit set costs more masked min/max
    // than it saves; publishing the already-loaded row as a transient shared
    // IMAGE instead turns hop 2 into a bank-spread shared gather and removes
    // the whole dependent global round trip (and its k scattered sectors).
    // SHARED ROW IMAGE FOR HOP 2.  The hint costs TWO dependent global round
    // trips per row: load pre_idx[j], then use it as an address into the row.
    // NCU on the streaming kernel charges 22% of ALL warp stalls to exactly that
    // second hop, and on the b<=148 register path there is at most ONE CTA per
    // SM -- no other warp exists to cover it.  But the row is already being
    // loaded into registers in the same issue window, so publishing it as a
    // transient shared IMAGE turns hop 2 into a bank-spread SHARED gather: the
    // chain becomes (row load) -> barrier -> shared gather instead of
    // (pre_idx load) -> global gather, i.e. one DRAM round trip instead of two.
    // Cost is VPT float4 shared stores plus two barriers.  It is a LOSS at wide
    // batch (4 CTAs/SM already cover the round trip and the image evicts the
    // occupancy), so it is enabled only for the MINB<=2 wide-batch variants.
    // MEASURED: at VPT>=2 the image LOSES (+2 to +7%) -- the barrier then has to
    // wait for VPT float4 per thread instead of the single pre_idx word, the
    // shared write set is VPT x larger, and the scattered shared gather runs
    // over VPT x more banks.  At VPT==1 it WINS (-5 to -6% at b=1/64).
    // r8: the gate is the IMAGE SIZE, not the batch.  NCU at b=1024 charges
    // 597k global sectors -- 56% of them EXCESSIVE -- to the hint's second hop
    // (k scattered 4B loads per row, each its own 32B sector), against 1.05M
    // fully-coalesced sectors for the whole row.  A warp's 32 scattered lanes
    // cost up to 32 L1 wavefronts but only ~4 SHARED ones (32 banks, not 32
    // cache lines), so publishing the already-resident row as a transient
    // shared image is an ~8x cut on that traffic at ANY occupancy.  It loses at
    // VPT=4 (a 64KB image, and the barrier then waits on four float4 per
    // thread instead of one pre_idx word), so the gate stays VPT<=2.
    constexpr bool USE_IMG = IMGF && (VPT == 1);
    uint32_t lmin = 0xffffffffu, lmax = 0u;
    if (tid == 0) { s_o1 = 0u; s_oc = 0u; }
    for (int i = tid; i < NBH; i += BLK) hist[i] = 0u;
    if constexpr (USE_IMG) {
#pragma unroll
        for (int u = 0; u < VPT; u++) {
            int i = tid + u * BLK;
            if (i < n4) ((float4*)img)[i] = make_float4(val[4*u+0], val[4*u+1], val[4*u+2], val[4*u+3]);
        }
        if (tid < ntail) img[tidx] = tval;
        __syncthreads();
#pragma unroll
        for (int t = 0; t < KPT; t++) {
            int p = pv[t];
            if ((unsigned)p < (unsigned)n) { uint32_t u = fkey(img[p]); lmin = min(lmin, u); lmax = max(lmax, u); }
        }
        __syncthreads();                         // img dies here; ck is reused later
    } else if constexpr (USE_BM) {
        const int NBW = (n + 31) >> 5;
        uint32_t* bm = ck;                       // crossing-bin buffer, not live yet
        for (int i = tid; i < NBW; i += BLK) bm[i] = 0u;
        __syncthreads();
#pragma unroll
        for (int t = 0; t < KPT; t++) {
            int p = pv[t];
            if ((unsigned)p < (unsigned)n) atomicOr(&bm[p >> 5], 1u << (p & 31));
        }
        __syncthreads();
        float lmn = INFINITY, lmx = -INFINITY;
#pragma unroll
        for (int u = 0; u < VPT; u++) {
            int base = (tid + u * BLK) << 2;     // multiple of 4: its 4 bits share a word
            unsigned w = ((unsigned)base < (unsigned)n) ? (bm[base >> 5] >> (base & 31)) : 0u;
#pragma unroll
            for (int c = 0; c < 4; c++)
                if (w & (1u << c)) { lmn = fminf(lmn, val[4 * u + c]); lmx = fmaxf(lmx, val[4 * u + c]); }
        }
        if (tid < ntail && ((bm[tidx >> 5] >> (tidx & 31)) & 1u)) { lmn = fminf(lmn, tval); lmx = fmaxf(lmx, tval); }
        lmin = fkey(lmn); lmax = fkey(lmx);      // monotone, so it commutes with min/max
        __syncthreads();                         // bm dies here; ck is reused later
    } else if constexpr (DEG) {
        // Row min/max straight out of the registers.  Padded slots hold
        // -INFINITY (never a real value in [0,n)), so one compare per slot
        // separates them; the max needs no guard because -INFINITY cannot win.
        float lmn = INFINITY, lmx = -INFINITY;
#pragma unroll
        for (int s2 = 0; s2 < S; s2++) {
            float v = val[s2];
            if (v > -INFINITY) { lmn = fminf(lmn, v); lmx = fmaxf(lmx, v); }
        }
        if (tid < ntail) { lmn = fminf(lmn, tval); lmx = fmaxf(lmx, tval); }
        lmin = fkey(lmn); lmax = fkey(lmx);   // monotone: commutes with min/max
    } else {
#pragma unroll
        for (int t = 0; t < KPT; t++) {
            int p = pv[t];
            if ((unsigned)p < (unsigned)n) {
                uint32_t u = fkey(__ldg(X + p));
                lmin = min(lmin, u); lmax = max(lmax, u);
            }
        }
    }
    // Block min/max in ONE barrier -- see the note in gvr_main.  This barrier
    // also publishes the histogram clear.
    lmin = warp_min_u32(lmin); lmax = warp_max_u32(lmax);
    if (lane == 0) { wmn[tid >> 5] = lmin; wmx[tid >> 5] = lmax; }
    __syncthreads();
    {
        constexpr int NW = BLK / 32;
        uint32_t a = (lane < NW) ? wmn[lane] : 0xffffffffu;
        uint32_t c = (lane < NW) ? wmx[lane] : 0u;
        lmin = warp_min_u32(a); lmax = warp_max_u32(c);
    }
    float T = invkey(lmin), GMAX = invkey(lmax);
    // The one-FFMA fold below evaluates val*SC - T*SC, so it needs SC to be a
    // sane scale: if GMAX-T collapses, the reciprocal explodes to 1e30 and
    // val*SC can overflow to +-inf (and inf-inf to NaN, which would saturate
    // EVERY element into the trash bin and break the count(bins>=1) >= k
    // invariant).  A collapsed range is treated as degenerate exactly as an
    // empty one already was: every real value then lands in one middle bin, the
    // crossing bin is the whole row, and the exact key-space narrowing takes it.
    if (!(T < GMAX) || !((GMAX - T) > 1e-30f)) { T = -3.0e38f; GMAX = 3.0e38f; }
    // BIN 0 IS A TRASH BIN under the branchless classify: q = fma(val-T, SC, 1)
    // is >= 1 for exactly the elements >= T (val-T is exactly signed and SC > 0,
    // so the fma of a non-negative product with 1 can only round to >= 1), and
    // the float->UNSIGNED convert saturates every negative q -- and the
    // -INFINITY padding -- to 0.  That replaces the per-slot `if (q >= 0)`
    // guard, which NCU charges ~30 warp-instructions of pure BSSY/BRA/BSYNC to,
    // with two clamps folded into the address.  Bins stay MONOTONE in val, and
    // count(bins >= 1) >= count(>= T) >= k, so the trash bin can never be the
    // crossing bin and nothing downstream changes.
    // GATED: it costs one extra live float across the whole unrolled classify,
    // and the 512-thread/VPT=2/MINB=4 variant runs at exactly 32 registers with
    // EIGHT resident values -- there it measured +11 to +15%.  Enable it only
    // where the thread either has the 64-register budget or holds one float4.
    constexpr bool BRL = (MINB * BLK <= 1024) || (VPT == 1);
    const float OFF = BRL ? 1.0f : 0.0f;
    const float WD = (GMAX - T) * (1.0f / (float)(NBH - (BRL ? 2 : 0)));
    const float SC = 1.0f / (WD > 0.f ? WD : 1e-30f);
    // q = (val - T) * SC is >= 0 for exactly the elements >= T (exact in fp:
    // val-T is exactly signed, SC > 0), so its sign is both the range test and
    // the validity test, and bin = trunc(min(q, NBH-1)).
    const float QCAP = (float)(NBH - 1);
    // ONE FFMA per slot instead of FADD+FFMA: (val-T)*SC + OFF is algebraically
    // fma(val, SC, OFF - T*SC), and SASS on the BRL variants shows the compiler
    // otherwise holding every (val-T) live across the whole unrolled classify
    // purely so the emit can re-multiply it.  The fold moves WHERE the rounding
    // happens, so the trash-bin invariant -- q(val) >= OFF for every val >= T,
    // which is what guarantees count(bins >= 1) >= count(>= T) >= k and hence
    // that the crossing bin never swallows the -INFINITY padding -- is restored
    // explicitly: CQ is nudged UP by ~1e-6*(|CQ|+1), which is >10x the 0.5-ulp
    // rounding error of T*SC and still under 1e-6 of a bin.  Nudging UP can only
    // pull elements slightly BELOW T into the bottom real bin, and that stays
    // exact because they rank below the k-th value and the crossing-bin select
    // is exact in key space.  Classify and emit evaluate the IDENTICAL fma, so
    // the bin an element is counted in is the bin it is emitted from, bit for
    // bit.  NOT applied on the !BRL variant: it runs at exactly 32 registers
    // with eight resident values, and once the two sites spell q identically
    // nvcc CSEs them and holds eight q's live across the barrier and the whole
    // 1024-bin scan (measured +1.3 to +2.5% there by r9-a001).
    const float CQ0 = OFF - T * SC;
    const float CQ  = CQ0 + 1e-6f * (fabsf(CQ0) + 1.0f);

    if constexpr (BRL) {
#pragma unroll
        for (int s = 0; s < S; s++)
            atomicAdd(&hist[min(__float2uint_rz(fmaf(val[s], SC, CQ)), (unsigned)(NBH - 1))], 1u);
        atomicAdd(&hist[min(__float2uint_rz(fmaf(tval, SC, CQ)), (unsigned)(NBH - 1))], 1u);
    } else {
#pragma unroll
        for (int s = 0; s < S; s++) {
            float q = (val[s] - T) * SC;
            if (q >= 0.f) atomicAdd(&hist[(int)fminf(q, QCAP)], 1u);
        }
        { float q = (tval - T) * SC;
          if (q >= 0.f) atomicAdd(&hist[(int)fminf(q, QCAP)], 1u); }
    }
    __syncthreads();
    // the block-parallel scan only earns its keep when the CURSORS it leaves
    // behind are what the emit runs on; otherwise warp-0 find_cross is cheaper.
    // find_cross's second level gives one bin per lane, so it only spans
    // NB_/32 <= 32 bins; past 1024 bins the block-parallel scan is the only
    // correct form.
    if constexpr (CUR || NBH > 1024) scan_cross_w<BLK, NBH>(hist, ws, k, tid, lane, &s_B, &s_m, &s_above, &s_tot);
    else                             find_cross<NBH>(hist, k, tid, lane, &s_B, &s_m, &s_above, &s_tot);
    __syncthreads();
    const int above = s_above, m = s_m, need = k - s_above;
    const int B = s_B;
    const bool whole = (need >= m);

    /* prod-fix ESCAPE: the crossing-bin compaction buffer holds CMP slots but
       the crossing bin can hold up to n candidates (>2560 DISTINCT values
       packed into one histogram bin -- battery case regbin_dense_trunc); the
       old code silently dropped the overflow (`q2 < CMP` below) and ranked a
       truncated set.  Never observed on real indexer logits (crossing bin
       ~250), so this path is correctness-only: an exact 32-step key-space
       bisection over the REGISTER-RESIDENT row finds the k-th key, then a
       two-predicate ballot emit writes the exact tie-aware top-k. */
    if (!whole && m > CMP) {
        if (tid == 0) { s_o1 = 0u; s_oc = 0u; }
        __syncthreads();
        uint32_t klo = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t kt = klo | (1u << bit);
            int cnt = 0;
#pragma unroll
            for (int s = 0; s < S; s++) {
                const int ix = ((tid + (s >> 2) * BLK) << 2) + (s & 3);
                cnt += (ix < n && fkey(val[s]) >= kt);
            }
            cnt += (tid < ntail && fkey(tval) >= kt);
            cnt = __reduce_add_sync(FULLM, cnt);
            if (lane == 0 && cnt) atomicAdd(&s_o1, (unsigned)cnt);
            __syncthreads();
            if ((int)s_o1 >= k) klo = kt;
            __syncthreads();
            if (tid == 0) s_o1 = 0u;
            __syncthreads();
        }
        /* klo = k-th largest key: count(>=klo) >= k, count(>klo) < k. */
        const long long ethr = (long long)klo;
        int abv = 0;
#pragma unroll
        for (int s = 0; s < S; s++) {
            const int ix = ((tid + (s >> 2) * BLK) << 2) + (s & 3);
            abv += (ix < n && (long long)fkey(val[s]) > ethr);
        }
        abv += (tid < ntail && (long long)fkey(tval) > ethr);
        abv = __reduce_add_sync(FULLM, abv);
        if (lane == 0 && abv) atomicAdd(&s_oc, (unsigned)abv);
        __syncthreads();
        const int nA = (int)s_oc, nT = k - nA;
        if (tid == 0) { s_o1 = 0u; s_oc = 0u; }
        __syncthreads();
#pragma unroll
        for (int s = 0; s < S; s++) {
            const int ixv = ((tid + (s >> 2) * BLK) << 2) + (s & 3);
            const long long u = (ixv < n) ? (long long)fkey(val[s]) : -1ll;
            const bool q1 = (u > ethr), q2 = (u == ethr);
            unsigned n1 = __ballot_sync(FULLM, q1), n2 = __ballot_sync(FULLM, q2);
            unsigned b1 = 0, b2 = 0;
            if (lane == 0) {
                if (n1) b1 = atomicAdd(&s_o1, (unsigned)__popc(n1));
                if (n2) b2 = atomicAdd(&s_oc, (unsigned)__popc(n2));
            }
            b1 = __shfl_sync(FULLM, b1, 0); b2 = __shfl_sync(FULLM, b2, 0);
            if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u));
                      if (p < (unsigned)nA) O[p] = ixv; }
            if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u));
                      if (p < (unsigned)nT) O[nA + p] = ixv; }
        }
        {
            const long long u = (tid < ntail) ? (long long)fkey(tval) : -1ll;
            const bool q1 = (u > ethr), q2 = (u == ethr);
            unsigned n1 = __ballot_sync(FULLM, q1), n2 = __ballot_sync(FULLM, q2);
            unsigned b1 = 0, b2 = 0;
            if (lane == 0) {
                if (n1) b1 = atomicAdd(&s_o1, (unsigned)__popc(n1));
                if (n2) b2 = atomicAdd(&s_oc, (unsigned)__popc(n2));
            }
            b1 = __shfl_sync(FULLM, b1, 0); b2 = __shfl_sync(FULLM, b2, 0);
            if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u));
                      if (p < (unsigned)nA) O[p] = tidx; }
            if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u));
                      if (p < (unsigned)nT) O[nA + p] = tidx; }
        }
        return;
    }

    if constexpr (CUR) {
    // ---- one register sweep through the per-bin OUTPUT CURSORS: a single
    // shared atomicAdd on an element's own bin both classifies it (bin > B is a
    // winner, bin == B lands past `above`) and reserves its output slot.  That
    // deletes the two 5-step warp prefix scans, the two ballots and the __ffs
    // walk the two-mask ballot emit needed.
    {
        const float LOQ = (float)B;                 // bin >= B  <=>  q >= B
        const int   lim1 = whole ? (above + m) : above;
#pragma unroll
        for (int s = 0; s < S; s++) {
            float q = (BRL ? fmaf(val[s], SC, CQ) : fmaf(val[s] - T, SC, OFF));
            if (q >= LOQ) {
                unsigned bn = min(__float2uint_rz(q), (unsigned)(NBH - 1));
                unsigned p = atomicAdd(&hist[bn], 1u);
                int idx = ((tid + (s >> 2) * BLK) << 2) + (s & 3);
                if (p < (unsigned)lim1) O[p] = idx;
                else if (!whole) { unsigned q2 = p - (unsigned)above;
                                   if (q2 < (unsigned)CMP) { ck[q2] = fkey(val[s]); ci[q2] = idx; } }
            }
        }
        { float q = (BRL ? fmaf(tval, SC, CQ) : fmaf(tval - T, SC, OFF));
          if (q >= LOQ) {
              unsigned bn = min(__float2uint_rz(q), (unsigned)(NBH - 1));
              unsigned p = atomicAdd(&hist[bn], 1u);
              if (p < (unsigned)lim1) O[p] = tidx;
              else if (!whole) { unsigned q2 = p - (unsigned)above;
                                 if (q2 < (unsigned)CMP) { ck[q2] = fkey(tval); ci[q2] = tidx; } }
          } }
        if (whole) return;
    }
    } else {
    // ---- one register sweep: emit definite winners, compact the crossing bin.
    // The bin index is never rebuilt: the "bin > B" / "bin == B" tests collapse
    // to two float compares against the bin boundaries in quantised space.
    {
        const float HI = whole ? (float)B : ((B >= NBH - 1) ? INFINITY : (float)(B + 1));
        const float LO = whole ? INFINITY : (float)B;
        unsigned m1 = 0u, m2 = 0u;
        bool t1 = false, t2 = false;
#pragma unroll
        for (int s = 0; s < S; s++) {
            float q = (BRL ? fmaf(val[s], SC, CQ) : fmaf(val[s] - T, SC, OFF));
            if (q >= HI)      m1 |= (1u << s);
            else if (q >= LO) m2 |= (1u << s);
        }
        { float q = (BRL ? fmaf(tval, SC, CQ) : fmaf(tval - T, SC, OFF));
          t1 = (q >= HI); t2 = !t1 && (q >= LO); }
        int c1 = __popc(m1) + (t1 ? 1 : 0), c2 = __popc(m2) + (t2 ? 1 : 0);
        int s1 = c1, s2 = c2;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int z1 = __shfl_up_sync(FULLM, s1, o); if (lane >= o) s1 += z1;
            int z2 = __shfl_up_sync(FULLM, s2, o); if (lane >= o) s2 += z2;
        }
        unsigned b1 = 0, b2 = 0;
        if (lane == 31) { b1 = atomicAdd(&s_o1, (unsigned)s1); b2 = atomicAdd(&s_oc, (unsigned)s2); }
        b1 = __shfl_sync(FULLM, b1, 31); b2 = __shfl_sync(FULLM, b2, 31);
        unsigned p1 = b1 + (unsigned)(s1 - c1), p2 = b2 + (unsigned)(s2 - c2);
        const int lim1 = whole ? k : above;
        // Winners are sparse (~k of n), so walk set bits instead of all S slots.
        for (unsigned w = m1; w; w &= w - 1u) {
            int s = __ffs(w) - 1;
            int idx = ((tid + (s >> 2) * BLK) << 2) + (s & 3);
            if (p1 < (unsigned)lim1) O[p1] = idx;
            p1++;
        }
        if (t1) { if (p1 < (unsigned)lim1) O[p1] = tidx; p1++; }
        // The crossing-bin slots need val[s] by static index, so they stay
        // unrolled -- but the whole block is skipped when the mask is empty.
        if (m2) {
#pragma unroll
            for (int s = 0; s < S; s++) {
                if (m2 & (1u << s)) {
                    int idx = ((tid + (s >> 2) * BLK) << 2) + (s & 3);
                    if (p2 < (unsigned)CMP) { ck[p2] = fkey(val[s]); ci[p2] = idx; }
                    p2++;
                }
            }
        }
        if (t2) { if (p2 < (unsigned)CMP) { ck[p2] = fkey(tval); ci[p2] = tidx; } p2++; }
        if (whole) return;
    }
    }
    __syncthreads();
    const int mc = CUR ? ((m < CMP) ? m : CMP) : (int)min(s_oc, (unsigned)CMP);

    if (mc >= m && mc <= QC) {
        // exact selection by counting strictly-greater keys (index-tie-broken)
        for (int i = tid; i < mc; i += BLK) {
            uint32_t u = ck[i];
            int r = 0;
            for (int j = 0; j < mc; j++) {
                uint32_t v = ck[j];
                r += (v > u) || (v == u && j < i);
            }
            if (r < need) O[above + r] = ci[i];
        }
        return;
    }

    // ---- fallback: exact key-space narrowing of the compacted crossing bin
    __shared__ uint32_t s_kmin, s_kmax;
    __shared__ unsigned s_e1, s_e2;
    if (tid == 0) { s_kmin = 0xffffffffu; s_kmax = 0u; }
    __syncthreads();
    for (int i = tid; i < mc; i += BLK) { atomicMin(&s_kmin, ck[i]); atomicMax(&s_kmax, ck[i]); }
    __syncthreads();
    uint32_t rlo = s_kmin, rhi = s_kmax;
    long long ethr = (long long)rlo; int aboveC = 0, needC = need, mm = mc;
    for (int lev = 0; ; ++lev) {
        if (needC == mm) { ethr = (long long)rlo - 1LL; aboveC += mm; needC = 0; break; }
        if (rlo >= rhi)  { ethr = (long long)rlo; break; }
        if (lev >= 6)    { ethr = (long long)rlo; break; }
        uint32_t d2 = rhi - rlo;
        int b2 = 32 - __clz(d2 | 1u);
        int sh2 = (b2 > LNBH) ? (b2 - LNBH) : 0;
        for (int i = tid; i < NBH; i += BLK) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < mc; i += BLK) {
            uint32_t u = ck[i];
            if (u >= rlo && u <= rhi)
                atomicAdd(&hist[min((unsigned)((u - rlo) >> sh2), (unsigned)(NBH - 1))], 1u);
        }
        __syncthreads();
        if constexpr (NBH > 1024) scan_cross_w<BLK, NBH>(hist, ws, needC, tid, lane, &s_B, &s_m, &s_above, &s_tot);
        else                      find_cross<NBH>(hist, needC, tid, lane, &s_B, &s_m, &s_above, &s_tot);
        __syncthreads();
        aboveC += s_above; needC -= s_above; mm = s_m;
        uint32_t nlo = rlo + ((uint32_t)s_B << sh2);
        rhi = (s_B == NBH - 1) ? rhi : (nlo + ((1u << sh2) - 1u));
        rlo = nlo;
    }
    if (tid == 0) { s_e1 = 0u; s_e2 = 0u; }
    __syncthreads();
    {
        int it2 = (mc + BLK - 1) / BLK;
        for (int it = 0; it < it2; ++it) {
            int i = it * BLK + tid;
            bool v = i < mc;
            uint32_t u = v ? ck[i] : 0u;
            int id = v ? ci[i] : 0;
            bool q1 = v && ((long long)u > ethr);
            bool q2 = v && ((long long)u == ethr);
            unsigned n1 = __ballot_sync(FULLM, q1);
            unsigned n2 = __ballot_sync(FULLM, q2);
            unsigned b1 = 0, b2 = 0;
            if (lane == 0) {
                if (n1) b1 = atomicAdd(&s_e1, (unsigned)__popc(n1));
                if (n2) b2 = atomicAdd(&s_e2, (unsigned)__popc(n2));
            }
            b1 = __shfl_sync(FULLM, b1, 0);
            b2 = __shfl_sync(FULLM, b2, 0);
            if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u)); if (p < (unsigned)aboveC) O[above + p] = id; }
            if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u)); if (p < (unsigned)needC) O[above + aboveC + p] = id; }
        }
    }
}


/* ---------------------------------------------------------------------------
   CLUSTERED streaming GVR kernel.  Identical algorithm to gvr_main, but the
   row is split across a THREAD-BLOCK CLUSTER of CS CTAs instead of CS
   independent CTAs joined by a global spin barrier.

   That replaces, per row:
     - NBS global atomicAdds (per-CTA bin offsets)
     - a red.release.gpu.global + ld.acquire.gpu.global spin loop
       (every acquire poll emits CCTL.IVALL -> whole-L1 invalidate)
     - NBS global loads to read the merged histogram back
     - a __threadfence + global RMW ticket, a global candidate slab, and
       2*NBS global stores to reset the scratch for the next launch
   with TWO hardware cluster barriers and CS distributed-shared-memory reads
   per bin.  Prior NCU on this path charged ~34% of all warp stalls to the
   global-barrier machinery at b=1; DSMEM keeps the merge on-chip.
--------------------------------------------------------------------------- */
template <int BLK, int U, int MINB, int NBS, int CS>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(BLK, MINB)
gvr_clus(const float* __restrict__ logits, const int* __restrict__ pre_idx,
         int* __restrict__ out, int n, int npad, int k,
         int SCAP, int CMP, int SMP, int TGT, int Q, int SS2, int TGT2) {
    GVR_GDC_WAIT();
    constexpr int HB = NBS;
    extern __shared__ __align__(16) unsigned char smraw[];
    uint32_t* hist = (uint32_t*)smraw;          // HB   (this CTA's raw counts)
    // r4: staged candidates carry their VALUE (int2 = value bits, index), the
    // SPLIT-slab pattern: the crossing-bin emit then needs no scattered X[id]
    // re-gather on the pre-exit critical chain, and the packed-bin IDXB cap
    // disappears.  +32KB dynamic shared at MINB=1 (85KB of the 100KB budget).
    int2*     cbuf = (int2*)(hist + HB);        // SCAP+4 (value bits, index)
    // r4 (a005): crossing-bin candidates land in rank 0 as ONE packed 64-bit
    // word (key<<32 | idx, the gvr_main P6 form): one DSMEM store instead of
    // two, a single unsigned compare per pair in the QUAD rank, and
    // ulonglong2 reads two candidates per 16B.  Same bytes as ckey+cidx.
    unsigned long long* ck64c = (unsigned long long*)(cbuf + SCAP + 4);   // CMP (key<<32|idx)
    uint32_t* mrg  = (uint32_t*)(ck64c + CMP);  // NBS  (cluster totals -> cursors)
    // r5: hoff[] folded into merge_scan0's registers -- no array at all.

    __shared__ uint32_t ws[BLK / 32], wmn[BLK / 32], wmx[BLK / 32];
    __shared__ unsigned s_bufn, s_o1, s_o2;
    __shared__ int s_B, s_m, s_above, s_tot, s_B2, s_B3;
    __shared__ float s_TSH;
    __shared__ uint32_t s_kmin, s_kmax;

    cg::cluster_group clus = cg::this_cluster();
    const int rank = (int)blockIdx.x;
    const int row  = (int)blockIdx.y;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    const float*  X  = logits + (size_t)row * (size_t)npad;
    const float4* X4 = (const float4*)X;
    const int*    P  = pre_idx + (size_t)row * (size_t)k;
    int*          O  = out + (size_t)row * (size_t)k;

    const int n4 = n >> 2;
    // INTERLEAVED chunk ownership: CTA `rank` owns chunks rank, rank+CS, ... of
    // BLK*U float4 each.  Captured decode rows have contiguous hot spans, so a
    // contiguous half-row slice puts nearly all survivors -- their walk, their
    // scattered reloads and their staging -- on ONE CTA of the cluster, and the
    // partner then parks at the merge barrier (NCU: ~17% of stalls on cluster
    // barrier arrive).  Interleaving 32KB+ chunks balances the emit work at
    // zero cost and stays DRAM-friendly.
    constexpr int STEPC = BLK * U;
    const int nCh = (n4 + STEPC - 1) / STEPC;
    const int nFullG = n4 / STEPC;
    const int tail0 = n4 << 2;
    const int tailn = (rank == 0) ? (n - tail0) : 0;

    if (tid == 0) { s_B2 = -1; s_B3 = -1; s_bufn = 0u; }
    for (int i = tid; i < HB; i += BLK) hist[i] = 0u;

    // ---------------- P1: sample prefetch (hint gather is LAZY) -------------
    // Same lazy-hint scheme as gvr_main: healthy rows land attempt 0 on the
    // sample rung alone -- the accept test (tot >= k) is exact by itself and
    // the histogram window comes from the sample (HIC).  GMIN/GMAX only
    // materialize on an attempt failure or a degenerate sample
    // (GVR_GATHER_HINT), computed identically on every rank of the cluster.
    float4 sa, sb;
    const bool shas = (tid < 2 * SMP);
    // r4: PAIRED sample -- one 64B line per location halves the random pages
    // vs the declustered form (sample service 2.1 -> ~1.1us).
    // r5 (peer ab6a7302) QUAD sample: one location is now a full 64B-aligned
    // line of FOUR float4 (16 elements), served by TWO threads (tid / tid+SMP
    // take the lower / upper pair).  Same element count and per-thread
    // register profile, HALF the page activations again (170 -> 85 locations
    // at 512k); the wide [k, SCAP] accept window keeps the rung margin
    // multi-sigma.  Clus-only: the same form measured +0.14..+0.75us drift
    // across gvr_main cells and stays out of that kernel.
    if (shas) { int p4 = (tid < SMP) ? (tid * SS2 * 4) : ((tid - SMP) * SS2 * 4 + 2);
                sa = __ldg(X4 + p4); sb = __ldg(X4 + p4 + 1); }
    // r4 PRIME-LATE: the pf prime used to issue here, "riding under the
    // sample phase" -- but platform phase timers show the sample then lands
    // at 2.27us instead of ~0.95us: the DRAM scheduler (FR-FCFS) prefers the
    // 8.4MB machine-wide streaming burst's row-buffer hits over the sample's
    // ~43k scattered sectors, and an in-CTA issue-order barrier cannot fix a
    // machine-wide queue.  Keep DRAM IDLE until every CTA's sample has landed
    // (the post-reduce barrier below), then prime -- the burst streams under
    // the ~800ns bin+scan chain instead of in front of the rung-critical
    // sample.  Measured on pro_512k_L26: sample-land 2272ns -> target ~950.
    constexpr int PFD = (U < 4 ? U : 4);
    const int lim4p = (npad >> 2) - 1;
    float4 pf[PFD];

    // ---------------- P2: quantile rung (redundant per CTA: it costs no
    // barrier and every CTA needs the identical rung anyway) -----------------
    float smn = INFINITY, smx = -INFINITY;
    if (shas) {
        float e[8] = { sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w };
#pragma unroll
        for (int t = 0; t < 8; t++) { smn = fminf(smn, e[t]); smx = fmaxf(smx, e[t]); }
    }
    for (int j = tid + BLK; j < 2 * SMP; j += BLK) {
        int p4 = (j < SMP) ? (j * SS2 * 4) : ((j - SMP) * SS2 * 4 + 2);
        float4 u1 = __ldg(X4 + p4), u2 = __ldg(X4 + p4 + 1);
        float e[8] = { u1.x, u1.y, u1.z, u1.w, u2.x, u2.y, u2.z, u2.w };
#pragma unroll
        for (int t = 0; t < 8; t++) { smn = fminf(smn, e[t]); smx = fmaxf(smx, e[t]); }
    }
    {
        const uint32_t a0 = warp_min_u32(fkey(smn)), c0 = warp_max_u32(fkey(smx));
        if (lane == 0) { wmn[tid >> 5] = a0; wmx[tid >> 5] = c0; }
    }
    __syncthreads();                 // one barrier (see the note in gvr_main)
    // every rank's sample has landed and been consumed; DRAM is idle.  Prime
    // the row-pass prefetch NOW so it streams under the bin+scan chain.
#pragma unroll
    for (int u = 0; u < PFD; u++) { int i = rank * STEPC + tid + u * BLK; pf[u] = X4[(i < n4) ? i : lim4p]; }
    // r4 (a001): L2-prefetch chunk 0's remaining u >= PFD slots -- the first
    // naked loads after the rung lands; no registers, pure hint.  DEEP rows
    // only (n4 >= 32768, each CTA owns >= 2 full chunks): on the 256k shapes
    // the same hint measured +1.4-3.2% -- the extra 64KB/CTA of L2 fill is
    // not amortized by a short row pass.  512k band measured -1 to -2.5%.
    if (n4 >= 32768 && (rank + 1) * STEPC <= n4) {
#pragma unroll
        for (int u = PFD; u < U; u++)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(X4 + rank * STEPC + tid + u * BLK));
    }
    float SMIN, SMAX;
    {
        constexpr int NW = BLK / 32;
        uint32_t a = (lane < NW) ? wmn[lane] : 0xffffffffu;
        uint32_t c = (lane < NW) ? wmx[lane] : 0u;
        SMIN = invkey(warp_min_u32(a)); SMAX = invkey(warp_max_u32(c));
    }

    float GMIN = -3.0e38f, GMAX = 3.0e38f;
    float T = -INFINITY;
    float HIC = -INFINITY;
    float w = 0.f;
    const bool sok = (SMP > 0) && (SMAX > SMIN);
    {
        if (sok) {
            w = (SMAX - SMIN) * (1.0f / (float)NBS);
            const float sc = 1.0f / w;
            if (shas) {
                float e[8] = { sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w };
#pragma unroll
                for (int t = 0; t < 8; t++)
                    atomicAdd(&hist[min(__float2int_rz((e[t] - SMIN) * sc), NBS - 1)], 1u);
            }
            for (int j = tid + BLK; j < 2 * SMP; j += BLK) {
                int p4 = (j < SMP) ? (j * SS2 * 4) : ((j - SMP) * SS2 * 4 + 2);
                float4 u1 = __ldg(X4 + p4), u2 = __ldg(X4 + p4 + 1);
                float e[8] = { u1.x, u1.y, u1.z, u1.w, u2.x, u2.y, u2.z, u2.w };
#pragma unroll
                for (int t = 0; t < 8; t++)
                    atomicAdd(&hist[min(__float2int_rz((e[t] - SMIN) * sc), NBS - 1)], 1u);
            }
        }
        __syncthreads();
        // r3: warp0-only suffix scan (scan_cross0) -- the block-parallel form
        // pays an internal ws[] handoff barrier that all 32 warps must reach;
        // NCU charged ~10% of all stall samples to this phase's barriers.
        // ZERO=true leaves the bins cleared for the row pass.
        scan_cross0<NBS, true, true, true>(hist, TGT, tid, lane, &s_B, &s_m, &s_above, &s_tot,
                                           TGT2, &s_B2, 2 * TGT, &s_B3);
        __syncthreads();               // publish warp 0's scan
        if (sok && s_tot >= TGT) T = fmaf((float)s_B, w, SMIN);
        if (!(T > -INFINITY)) {
            // Degenerate sample: pay the hint round trips now (identical on
            // every rank of the cluster, so the rung stays consistent).
            GVR_GATHER_HINT(GMIN, GMAX, 1);
            T = GMIN;
        }
        if (sok && s_tot >= TGT && s_B2 >= 0) {
            float Tk = fmaf((float)s_B2, w, SMIN);
            // fire even when Tk == T (dense rows), matching gvr_main.
            float up = fmaxf(Tk - T, 0.0f);
            // r5 (peer ab6a7302): the rank-k estimate rests on ~TGT2 (~10)
            // sample hits; on a heavy upper tail it lands far ABOVE truth and
            // the 4x stretch quantises the live range into a handful of
            // coarse bins (L60 probe: HIC-T 10.6 vs needed ~0.04 -> crossing
            // bin m=262 at B=0, +2.8us in the fat-bin tail).  Cap the stretch
            // by the locally measured density scale T - T3 (rank-TGT to
            // rank-2*TGT distance, ~3x Tk's evidence).  A too-tight cap only
            // piles winners into the clamp bin, which the staged narrowing
            // already bounds; the 8-bin floor below still applies.
            if (s_tot >= 2 * TGT && s_B3 >= 0) {
                float T3 = fmaf((float)s_B3, w, SMIN);
                if (T > T3) up = fminf(up, 2.0f * (T - T3));
            }
            HIC = fmaxf(fmaf(4.0f, up, T), fmaf(8.0f, w, T));
        }
        // r4 (a000): ladder floor -- register-free port of gvr_main's TSH
        // rung.  The sample's rank-(2*TGT) estimate goes to SHARED (one
        // word), so the hot path holds no extra registers (the register form
        // spilled this kernel at the 64-reg wall, +3% in r3).  Read only on
        // the cold retry path; -INFINITY means "not armed".  Cluster-
        // consistent: every rank derives it from the identical sample.
        if (tid == 0) {
            float t5 = -INFINITY;
            if (sok && s_tot >= 2 * TGT && s_B3 >= 0 && T > GMIN) {
                float T3 = fmaf((float)s_B3, w, SMIN);
                if (T3 < T) t5 = T3;
            }
            s_TSH = t5;
        }
    }

    // ================= attempt loop (guess -> verify) =========================
    int listN = 0, above = 0, m = 0, need = 0, B = 0;
    float SC = 1.f, TF = T;
    bool complete = false, valid = false;
#pragma unroll 1
    for (int att = 0; att < 3; ++att) {
        if (att) {
            // EXACTNESS: pf[] holds stale data from the previous attempt's
            // roll; a retry classifying iteration 0 against it silently drops
            // true winners.  Re-prime; att 0 hot path untouched.
            if (rank < nFullG) {
#pragma unroll
                for (int u = 0; u < PFD; u++) pf[u] = X4[rank * STEPC + tid + u * BLK];
            } else {
#pragma unroll
                for (int u = 0; u < PFD; u++) { int i = rank * STEPC + tid + u * BLK; pf[u] = X4[(i < n4) ? i : lim4p]; }
            }
            clus.sync();                  // everyone done reading hist
            // r3: attempt 0 needs no clear -- the sample scan ran with
            // ZERO=true (bins left zeroed, published by the scan barrier) and
            // s_bufn was cleared at kernel entry (published by the first
            // barrier).  Only the retry pays the clear and its barrier.
            for (int i = tid; i < NBS; i += BLK) hist[i] = 0u;
            if (tid == 0) s_bufn = 0u;
            __syncthreads();
        }

        TF = T;
        float hi = fmaxf(GMAX, T);
        if (HIC > T && HIC < hi) hi = HIC;
        float WD = (hi - T) * (1.0f / (float)NBS);
        if (!(WD > 0.f)) WD = 1e-30f;
        SC = 1.0f / WD;

        {
            // Full iterations peeled from the (at most one) partial one: they
            // lose the bounds ISETP, the clamped-address SEL and the INFINITY
            // threshold SEL -- three instructions per float4.
            const int lim4 = (npad >> 2) - 1;
            for (int g = rank; g < nCh; g += CS) {
                const int i0 = g * STEPC + tid;
                unsigned M = 0u;
                if (g < nFullG) {
#pragma unroll
                    for (int u = 0; u < U; u++) {
                        float4 v = (u < PFD) ? pf[u < PFD ? u : 0] : X4[i0 + u * BLK];
                        if (v.x >= TF) M |= (1u << (u * 4 + 0));
                        if (v.y >= TF) M |= (1u << (u * 4 + 1));
                        if (v.z >= TF) M |= (1u << (u * 4 + 2));
                        if (v.w >= TF) M |= (1u << (u * 4 + 3));
                    }
                } else {
#pragma unroll
                for (int u = 0; u < U; u++) {
                    const int i = i0 + u * BLK;
                    const bool ok = (i < n4);
                    float4 v = (u < PFD) ? pf[u < PFD ? u : 0] : X4[ok ? i : lim4];
                    /* prod-fix: same +inf-padding escape as gvr_main -- the
                       clamp slot lim4 lives in [n, npad); gate on ok. */
                    if (ok) {
                        if (v.x >= TF) M |= (1u << (u * 4 + 0));
                        if (v.y >= TF) M |= (1u << (u * 4 + 1));
                        if (v.z >= TF) M |= (1u << (u * 4 + 2));
                        if (v.w >= TF) M |= (1u << (u * 4 + 3));
                    }
                }
                }
                // ROLL THE PREFETCH FORWARD (see gvr_main): the head of the
                // NEXT owned chunk depends on nothing this iteration produces,
                // so issue it before the slot reservation and the survivor walk
                // -- the walk is a chain of scattered dependent reloads, and
                // the roll covers a full DRAM round trip for the next chunk.
                const int g2 = g + CS;
                if (g2 < nCh) {
                    const int j0 = g2 * STEPC + tid;
                    if (g2 < nFullG) {
#pragma unroll
                        for (int u = 0; u < PFD; u++) pf[u] = X4[j0 + u * BLK];
                    } else {
#pragma unroll
                        for (int u = 0; u < PFD; u++) { const int j = j0 + u * BLK; pf[u] = X4[(j < n4) ? j : lim4]; }
                    }
                }
                // Warp-aggregated slot reservation.  Every thread of the CTA
                // hammers the SAME shared word and shared atomics to one
                // address serialise, so the per-thread form was ~14 serialised
                // RMWs per warp per iteration -- at 32 warps roughly 900 fully
                // serialised RMWs per CTA on the row pass's critical path.  A
                // 5-step shuffle scan turns each warp's into ONE.  (gvr_main
                // has had this since round 6; the clustered kernel had not.)
                const int cnt = __popc(M);
                int inc = cnt;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) { int z = __shfl_up_sync(FULLM, inc, o); if (lane >= o) inc += z; }
                unsigned bpos = 0u;
                if (lane == 31 && inc) bpos = atomicAdd(&s_bufn, (unsigned)inc);
                unsigned pos = __shfl_sync(FULLM, bpos, 31) + (unsigned)(inc - cnt);
#define GVR_EMITK(xv_, idx_)                                                       \
                do {                                                               \
                    unsigned bn = min(__float2uint_rz(((xv_) - TF) * SC),           \
                                      (unsigned)(NBS - 1));                        \
                    atomicAdd(&hist[bn], 1u);                                      \
                    cbuf[min(pos, (unsigned)SCAP)] =                               \
                        make_int2(__float_as_int(xv_), (int)(idx_));               \
                    pos++;                                                         \
                } while (0)
                if (M) {
                    int bp = __ffs(M) - 1;
                    M &= (M - 1u);
                    int idx = ((i0 + (bp >> 2) * BLK) << 2) + (bp & 3);
                    float xv = X[idx];
                    while (M) {
                        int bp2 = __ffs(M) - 1;
                        M &= (M - 1u);
                        int idx2 = ((i0 + (bp2 >> 2) * BLK) << 2) + (bp2 & 3);
                        float xv2 = X[idx2];
                        GVR_EMITK(xv, idx);
                        idx = idx2; xv = xv2;
                    }
                    GVR_EMITK(xv, idx);
                }
#undef GVR_EMITK
            }
            for (int i = tid; i < tailn; i += BLK) {
                float x = X[tail0 + i];
                if (x >= TF) {
                    unsigned bn = (unsigned)min(__float2int_rz((x - TF) * SC), NBS - 1);
                    atomicAdd(&hist[bn], 1u);
                    unsigned pos = atomicAdd(&s_bufn, 1u);
                    if (pos < (unsigned)SCAP) cbuf[pos] = make_int2(__float_as_int(x), tail0 + i);
                }
            }
        }
        // -------- cluster merge: warp0-fused DSMEM merge + scan --------------
        // (cluster.sync subsumes the block barrier and orders the hist
        // atomics + s_bufn for both the DSMEM readers and the myn read below)
        clus.sync();
        // r5 (peer ab6a7302): the 256-thread mrg/hoff materialisation and its
        // publishing barrier are folded into the warp-0 scan (merge_scan0):
        // CS DSMEM uint4 reads per lane-span, cursors biased by the r<rank
        // prefix inline.  One barrier here instead of two.
        const int myn = (int)s_bufn;
        merge_scan0<NBS, CS>(hist, mrg, clus, rank, k, tid, lane,
                             &s_B, &s_m, &s_above, &s_tot);
        __syncthreads();
        const int tot = s_tot;
        if (tot >= k) {
            valid = true; complete = (myn <= SCAP); listN = myn;
            above = s_above; m = s_m; need = k - s_above; B = s_B;
            break;
        }
        if (att == 2) break;
        // r4 (a000) ladder: att0 miss -> TSH (sample rank-(2*TGT) floor) with
        // NO hint gather -- if TSH <= GMIN the accept is guaranteed
        // (count(>=GMIN) >= k by construction), and a TSH miss pays the
        // gather on the way to attempt 2.  Every rank armed the identical
        // s_TSH from the identical whole-row sample, so the ladder is
        // cluster-consistent.
        if (att == 0 && s_TSH > -INFINITY && s_TSH < TF) { T = s_TSH; continue; }
        // LAZY GATHER -- both ranks compute identical GMIN.
        if (GMIN == -3.0e38f) GVR_GATHER_HINT(GMIN, GMAX, 1);
        if (!(T > GMIN)) break;
        T = GMIN;
    }


    const bool whole = valid && (need >= m);
    const int  lim1  = whole ? (above + m) : above;
    const bool degen = (!valid) || (m > CMP);
    const int  mc    = degen ? 0 : m;

    // crossing-bin candidates from every CTA of the cluster land in rank 0's
    // buffers directly (DSMEM store), so the refine needs no global slab and
    // no scattered re-gather in a single trailing CTA.
    unsigned long long* rk64 = (unsigned long long*)clus.map_shared_rank(ck64c, 0);

    if (!degen) {
        if (complete) {
            for (int i = tid; i < listN; i += BLK) {
                int2 wv = cbuf[i];
                const float xv = __int_as_float(wv.x); const int id = wv.y;
                int bn = (int)min(__float2int_rz((xv - TF) * SC), NBS - 1);
                if (bn >= B) {
                    unsigned p = atomicAdd(&mrg[bn], 1u);
                    if (p < (unsigned)lim1) O[p] = id;
                    else if (!whole) { unsigned q2 = p - (unsigned)above;
                                       if (q2 < (unsigned)CMP) { rk64[q2] = ((unsigned long long)fkey(xv) << 32) | (unsigned long long)(unsigned)id; } }
                }
            }
        } else {
            // EXACTNESS: sweep exactly this CTA's OWNED CHUNKS plus (rank 0)
            // the true row tail at [tail0, tail0+tailn) -- the old contiguous
            // (c1<<2)+tailn bound double-swept the partner's first elements
            // and missed the tail.
            for (int g = rank; g < nCh; g += CS) {
                const int lo2 = (g * STEPC) << 2;
                int e4 = (g + 1) * STEPC; if (e4 > n4) e4 = n4;
                const int hi2 = e4 << 2;
                for (int i = lo2 + tid; i < hi2; i += BLK) {
                    float x = X[i];
                    if (x >= TF) {
                        int bn = (int)min(__float2int_rz((x - TF) * SC), NBS - 1);
                        if (bn >= B) {
                            unsigned p = atomicAdd(&mrg[bn], 1u);
                            if (p < (unsigned)lim1) O[p] = i;
                            else if (!whole) { unsigned q2 = p - (unsigned)above;
                                               if (q2 < (unsigned)CMP) { rk64[q2] = ((unsigned long long)fkey(x) << 32) | (unsigned long long)(unsigned)i; } }
                        }
                    }
                }
            }
            for (int t2 = tid; t2 < tailn; t2 += BLK) {
                const int ii = tail0 + t2;
                float x = X[ii];
                if (x >= TF) {
                    int bn = (int)min(__float2int_rz((x - TF) * SC), NBS - 1);
                    if (bn >= B) {
                        unsigned p = atomicAdd(&mrg[bn], 1u);
                        if (p < (unsigned)lim1) O[p] = ii;
                        else if (!whole) { unsigned q2 = p - (unsigned)above;
                                           if (q2 < (unsigned)CMP) { rk64[q2] = ((unsigned long long)fkey(x) << 32) | (unsigned long long)(unsigned)ii; } }
                    }
                }
            }
        }
    }

    clus.sync();                     // all DSMEM traffic retired; safe to exit (subsumes the block barrier)
    if (rank != 0) return;

    if (!degen) {
        if (whole) return;

        if (mc <= QUADC_CLUS) {
            // exact selection by counting strictly-greater packed keys: the
            // index tie-break rides in the low half so ONE unsigned compare
            // decides, and a ulonglong2 read brings two candidates per 16B.
            const int mc2 = mc & ~1;
            for (int i = tid; i < mc; i += BLK) {
                unsigned long long u = ck64c[i];
                int r = 0;
                for (int j = 0; j < mc2; j += 2) {
                    ulonglong2 v = *(const ulonglong2*)(ck64c + j);
                    r += (v.x > u) + (v.y > u);
                }
                if (mc2 < mc) r += (ck64c[mc2] > u);
                if (r < need) O[above + r] = (int)(unsigned)u;
            }
            return;
        }
        if (tid == 0) { s_kmin = 0xffffffffu; s_kmax = 0u; }
        // cleared ONCE: every narrowing level's scan leaves the bins zeroed
        for (int i = tid; i < NBS; i += BLK) hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < mc; i += BLK) { uint32_t kk = (uint32_t)(ck64c[i] >> 32);
                                              atomicMin(&s_kmin, kk); atomicMax(&s_kmax, kk); }
        __syncthreads();
        uint32_t rlo = s_kmin, rhi = s_kmax;
        long long ethr = (long long)rlo; int aboveC = 0, needC = need, mm = mc;
        for (int lev = 0; ; ++lev) {
            if (needC == mm) { ethr = (long long)rlo - 1LL; aboveC += mm; needC = 0; break; }
            if (rlo >= rhi)  { ethr = (long long)rlo; break; }
            if (lev >= 6)    { ethr = (long long)rlo; break; }
            uint32_t d2 = rhi - rlo;
            int b2 = 32 - __clz(d2 | 1u);
            int lb = 0; { int t2 = NBS; while (t2 > 1) { t2 >>= 1; lb++; } }
            int sh2 = (b2 > lb) ? (b2 - lb) : 0;
            for (int i = tid; i < mc; i += BLK) {
                uint32_t u = (uint32_t)(ck64c[i] >> 32);
                if (u >= rlo && u <= rhi)
                    atomicAdd(&hist[min((unsigned)((u - rlo) >> sh2), (unsigned)(NBS - 1))], 1u);
            }
            __syncthreads();
            scan_cross0<NBS, true>(hist, needC, tid, lane, &s_B, &s_m, &s_above, &s_tot);
            __syncthreads();
            aboveC += s_above; needC -= s_above; mm = s_m;
            uint32_t nlo = rlo + ((uint32_t)s_B << sh2);
            rhi = (s_B == NBS - 1) ? rhi : (nlo + ((1u << sh2) - 1u));
            rlo = nlo;
        }
        if (tid == 0) { s_o1 = 0u; s_o2 = 0u; }
        __syncthreads();
        int it2 = (mc + BLK - 1) / BLK;
        for (int it = 0; it < it2; ++it) {
            int i = it * BLK + tid;
            bool val = i < mc;
            unsigned long long w64 = val ? ck64c[i] : 0ull;
            uint32_t u = (uint32_t)(w64 >> 32);
            int id = (int)(unsigned)w64;
            bool q1 = val && ((long long)u > ethr);
            bool q2 = val && ((long long)u == ethr);
            unsigned n1 = __ballot_sync(FULLM, q1);
            unsigned n2 = __ballot_sync(FULLM, q2);
            unsigned b1 = 0, b2 = 0;
            if (lane == 0) {
                if (n1) b1 = atomicAdd(&s_o1, (unsigned)__popc(n1));
                if (n2) b2 = atomicAdd(&s_o2, (unsigned)__popc(n2));
            }
            b1 = __shfl_sync(FULLM, b1, 0);
            b2 = __shfl_sync(FULLM, b2, 0);
            if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u)); if (p < (unsigned)aboveC) O[above + p] = id; }
            if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u)); if (p < (unsigned)needC) O[above + aboveC + p] = id; }
        }
        return;
    }

    // ---- degenerate: exact key-space narrowing over the whole row (rank 0) --
    {
        uint32_t rlo = 0u, rhi = 0xffffffffu;
        int above2 = 0, need2 = k, m2 = n;
        long long ethr = 0; bool tieM = true;
        int lb = 0; { int t2 = NBS; while (t2 > 1) { t2 >>= 1; lb++; } }
        for (int lev = 0; ; ++lev) {
            if (need2 == m2) { ethr = (long long)rlo - 1LL; above2 += m2; need2 = 0; tieM = false; break; }
            if (rlo >= rhi)  { ethr = (long long)rlo; break; }
            if (lev >= 8)    { ethr = (long long)rlo; break; }
            uint32_t d2 = rhi - rlo;
            int b2 = 32 - __clz(d2 | 1u);
            int sh2 = (b2 > lb) ? (b2 - lb) : 0;
            for (int i = tid; i < NBS; i += BLK) hist[i] = 0u;
            __syncthreads();
            for (int i = tid; i < n; i += BLK) {
                uint32_t u = fkey(X[i]);
                if (u >= rlo && u <= rhi)
                    atomicAdd(&hist[min((unsigned)((u - rlo) >> sh2), (unsigned)(NBS - 1))], 1u);
            }
            __syncthreads();
            scan_cross<BLK, NBS>(hist, ws, need2, tid, lane, &s_B, &s_m, &s_above, &s_tot);
            __syncthreads();
            above2 += s_above; need2 -= s_above; m2 = s_m;
            uint32_t nlo = rlo + ((uint32_t)s_B << sh2);
            rhi = (s_B == NBS - 1) ? rhi : (nlo + ((1u << sh2) - 1u));
            rlo = nlo;
        }
        if (tid == 0) { s_o1 = 0u; s_o2 = 0u; }
        __syncthreads();
        int nA = tieM ? above2 : k, nT = tieM ? need2 : 0;
        int it2 = (n + BLK - 1) / BLK;
        for (int it = 0; it < it2; ++it) {
            int i = it * BLK + tid;
            bool val = i < n;
            uint32_t u = val ? fkey(X[i]) : 0u;
            bool q1 = val && ((long long)u > ethr);
            bool q2 = val && tieM && ((long long)u == ethr);
            unsigned n1 = __ballot_sync(FULLM, q1);
            unsigned n2 = __ballot_sync(FULLM, q2);
            unsigned b1 = 0, b2 = 0;
            if (lane == 0) {
                if (n1) b1 = atomicAdd(&s_o1, (unsigned)__popc(n1));
                if (n2) b2 = atomicAdd(&s_o2, (unsigned)__popc(n2));
            }
            b1 = __shfl_sync(FULLM, b1, 0);
            b2 = __shfl_sync(FULLM, b2, 0);
            if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u)); if (p < (unsigned)nA) O[p] = i; }
            if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u)); if (p < (unsigned)nT) O[nA + p] = i; }
        }
    }
}


/* ---------------------------------------------------------------------------
   CLUSTERED register-resident GVR variant.

   The streaming/collect path carries ~4us of fixed cost that the register path
   does not: a strided quantile sample with its own histogram+scan+2 barriers, a
   shared candidate staging buffer written in the row pass and re-read in the
   emit, and a two-attempt verify loop.  A single CTA can only hold 16K elements
   in registers -- but a CLUSTER of 16 can hold 256K, i.e. every b=1 row in this
   suite.  So run the *register* algorithm (T = GMIN directly, one float-space
   histogram, one register sweep) across a cluster: the per-CTA instruction
   stream is then identical to the n<=16387 path that costs 7.5us, and the only
   additions are two hardware cluster barriers and CS DSMEM reads per bin.
--------------------------------------------------------------------------- */
#define CMPC   4096              /* crossing-bin slots PER CTA (power of two) */
#define LCMPC  12
#define BLKC   1024              /* CTA size of the clustered register path */

template <int BLK, int VPT, int CS>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(BLK, 1)
gvr_reg_clus(const float* __restrict__ logits, const int* __restrict__ pre_idx,
             int* __restrict__ out, int n, int npad, int k) {
    GVR_GDC_WAIT();
    constexpr int S    = VPT * 4;
    constexpr int SPAN = BLK * VPT;          // float4 owned by one CTA
    extern __shared__ __align__(16) unsigned char smr[];
    uint32_t* hist = (uint32_t*)smr;          // NB   (this CTA's raw counts)
    uint32_t* mrg  = hist + NB;               // NB   (cluster totals -> cursors)
    uint32_t* hoff = mrg + NB;                // NB   (this CTA's bin offset)
    uint32_t* ck   = hoff + NB;               // CMPC
    int*      ci   = (int*)(ck + CMPC);       // CMPC

    __shared__ uint32_t ws[BLK / 32], wmn[BLK / 32], wmx[BLK / 32];
    __shared__ int s_B, s_m, s_above, s_tot;
    __shared__ unsigned s_o1, s_o2;
    __shared__ uint32_t s_kmin, s_kmax;

    cg::cluster_group clus = cg::this_cluster();
    const int rank = (int)blockIdx.x;
    const int row  = (int)blockIdx.y;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    const float*  X  = logits + (size_t)row * (size_t)npad;
    const float4* X4 = (const float4*)X;
    const int*    P  = pre_idx + (size_t)row * (size_t)k;
    int*          O  = out + (size_t)row * (size_t)k;

    const int n4    = n >> 2;
    const int ntail = n - (n4 << 2);
    const int base4 = rank * SPAN;

    // Every CTA gathers the WHOLE hint (k <= BLK, one coalesced load and one
    // scattered gather per thread): redundant across the cluster, but it makes
    // GMIN/GMAX identical everywhere with no cluster barrier at all.
    int pv0 = (tid < k) ? P[tid] : -1;

    float val[S];
#pragma unroll
    for (int u = 0; u < VPT; u++) {
        int i = base4 + tid + u * BLK;
        float4 v;
        if (i < n4) v = X4[i];
        else { v.x = -INFINITY; v.y = -INFINITY; v.z = -INFINITY; v.w = -INFINITY; }
        val[4 * u + 0] = v.x; val[4 * u + 1] = v.y;
        val[4 * u + 2] = v.z; val[4 * u + 3] = v.w;
    }
    const int   tidx = (n4 << 2) + tid;
    const float tval = (rank == 0 && tid < ntail) ? X[tidx] : -INFINITY;

    if (tid == 0) { s_o1 = 0u; s_o2 = 0u; }
    for (int i = tid; i < NB; i += BLK) hist[i] = 0u;

    uint32_t lmin = 0xffffffffu, lmax = 0u;
    if ((unsigned)pv0 < (unsigned)n) {
        uint32_t u = fkey(__ldg(X + pv0));
        lmin = u; lmax = u;
    }
    lmin = warp_min_u32(lmin); lmax = warp_max_u32(lmax);
    if (lane == 0) { wmn[tid >> 5] = lmin; wmx[tid >> 5] = lmax; }
    __syncthreads();                 // one barrier (see the note in gvr_main)
    {
        constexpr int NW = BLK / 32;
        uint32_t a = (lane < NW) ? wmn[lane] : 0xffffffffu;
        uint32_t c = (lane < NW) ? wmx[lane] : 0u;
        lmin = warp_min_u32(a); lmax = warp_max_u32(c);
    }
    float T = invkey(lmin), GMAX = invkey(lmax);
    // The one-FFMA fold below evaluates val*SC - T*SC, so it needs SC to be a
    // sane scale: if GMAX-T collapses, the reciprocal explodes to 1e30 and
    // val*SC can overflow to +-inf (and inf-inf to NaN, which would saturate
    // EVERY element into the trash bin and break the count(bins>=1) >= k
    // invariant).  A collapsed range is treated as degenerate exactly as an
    // empty one already was: every real value then lands in one middle bin, the
    // crossing bin is the whole row, and the exact key-space narrowing takes it.
    if (!(T < GMAX) || !((GMAX - T) > 1e-30f)) { T = -3.0e38f; GMAX = 3.0e38f; }
    // BIN 0 IS A TRASH BIN under the branchless classify: q = fma(val-T, SC, 1)
    // is >= 1 for exactly the elements >= T (val-T is exactly signed and SC > 0,
    // so the fma of a non-negative product with 1 can only round to >= 1), and
    // the float->UNSIGNED convert saturates every negative q -- and the
    // -INFINITY padding -- to 0.  That replaces the per-slot `if (q >= 0)`
    // guard, which NCU charges ~30 warp-instructions of pure BSSY/BRA/BSYNC to,
    // with two clamps folded into the address.  Bins stay MONOTONE in val, and
    // count(bins >= 1) >= count(>= T) >= k, so the trash bin can never be the
    // crossing bin and nothing downstream changes.
    const float WD = (GMAX - T) * (1.0f / (float)(NB - 2));
    const float SC = 1.0f / (WD > 0.f ? WD : 1e-30f);
    // one FFMA per slot -- see the note in gvr_topk_reg
    const float CQ0 = 1.0f - T * SC;
    const float CQ  = CQ0 + 1e-6f * (fabsf(CQ0) + 1.0f);

#pragma unroll
    for (int s = 0; s < S; s++)
        atomicAdd(&hist[min(__float2uint_rz(fmaf(val[s], SC, CQ)), (unsigned)(NB - 1))], 1u);
    atomicAdd(&hist[min(__float2uint_rz(fmaf(tval, SC, CQ)), (unsigned)(NB - 1))], 1u);

    clus.sync();                                   // local histograms complete
    for (int i = tid; i < NB; i += BLK) {
        uint32_t tot = 0u, pre = 0u;
#pragma unroll
        for (int r = 0; r < CS; r++) {
            uint32_t v = *(const uint32_t*)clus.map_shared_rank(hist + i, r);
            if (r < rank) pre += v;
            tot += v;
        }
        mrg[i] = tot; hoff[i] = pre;
    }
    __syncthreads();
    scan_cross_w<BLK, NB>(mrg, ws, k, tid, lane, &s_B, &s_m, &s_above, &s_tot);
    __syncthreads();
    const int above = s_above, m = s_m, need = k - s_above, B = s_B;
    const bool whole = (need >= m);
    const bool degen = (m > CS * CMPC);
    for (int i = tid; i < NB; i += BLK) mrg[i] += hoff[i];
    __syncthreads();

    if (!degen) {
        const float LOQ = (float)B;
        const int   lim1 = whole ? (above + m) : above;
#pragma unroll
        for (int s = 0; s < S; s++) {
            float q = fmaf(val[s], SC, CQ);
            if (q >= LOQ) {
                unsigned bn = min(__float2uint_rz(q), (unsigned)(NB - 1));
                unsigned p = atomicAdd(&mrg[bn], 1u);
                int idx = ((base4 + tid + (s >> 2) * BLK) << 2) + (s & 3);
                if (p < (unsigned)lim1) O[p] = idx;
                else if (!whole) {
                    unsigned q2 = p - (unsigned)above;
                    uint32_t* rk = (uint32_t*)clus.map_shared_rank(ck, (int)(q2 >> LCMPC));
                    int*      ri = (int*)     clus.map_shared_rank(ci, (int)(q2 >> LCMPC));
                    rk[q2 & (CMPC - 1)] = fkey(val[s]);
                    ri[q2 & (CMPC - 1)] = idx;
                }
            }
        }
        { float q = fmaf(tval, SC, CQ);
          if (q >= LOQ) {
              unsigned bn = min(__float2uint_rz(q), (unsigned)(NB - 1));
              unsigned p = atomicAdd(&mrg[bn], 1u);
              if (p < (unsigned)lim1) O[p] = tidx;
              else if (!whole) {
                  unsigned q2 = p - (unsigned)above;
                  uint32_t* rk = (uint32_t*)clus.map_shared_rank(ck, (int)(q2 >> LCMPC));
                  int*      ri = (int*)     clus.map_shared_rank(ci, (int)(q2 >> LCMPC));
                  rk[q2 & (CMPC - 1)] = fkey(tval);
                  ri[q2 & (CMPC - 1)] = tidx;
              }
          } }
    }

    __syncthreads();
    clus.sync();                                   // staging retired

    if (rank == 0 && !whole) {
        const int mc = degen ? 0 : m;
        if (!degen && mc <= QUADC) {
            for (int i = tid; i < mc; i += BLK) {
                uint32_t u = ck[i];
                int r = 0;
                for (int j = 0; j < mc; j++) {
                    uint32_t v = ck[j];
                    r += (v > u) || (v == u && j < i);
                }
                if (r < need) O[above + r] = ci[i];
            }
        } else if (!degen) {
            if (tid == 0) { s_kmin = 0xffffffffu; s_kmax = 0u; }
            __syncthreads();
            for (int i = tid; i < mc; i += BLK) {
                uint32_t u = *(const uint32_t*)clus.map_shared_rank(ck + (i & (CMPC - 1)), i >> LCMPC);
                atomicMin(&s_kmin, u); atomicMax(&s_kmax, u);
            }
            __syncthreads();
            uint32_t rlo = s_kmin, rhi = s_kmax;
            long long ethr = (long long)rlo; int aboveC = 0, needC = need, mm = mc;
            for (int lev = 0; ; ++lev) {
                if (needC == mm) { ethr = (long long)rlo - 1LL; aboveC += mm; needC = 0; break; }
                if (rlo >= rhi)  { ethr = (long long)rlo; break; }
                if (lev >= 6)    { ethr = (long long)rlo; break; }
                uint32_t d2 = rhi - rlo;
                int b2 = 32 - __clz(d2 | 1u);
                int sh2 = (b2 > LNB) ? (b2 - LNB) : 0;
                for (int i = tid; i < NB; i += BLK) hist[i] = 0u;
                __syncthreads();
                for (int i = tid; i < mc; i += BLK) {
                    uint32_t u = *(const uint32_t*)clus.map_shared_rank(ck + (i & (CMPC - 1)), i >> LCMPC);
                    if (u >= rlo && u <= rhi)
                        atomicAdd(&hist[min((unsigned)((u - rlo) >> sh2), (unsigned)(NB - 1))], 1u);
                }
                __syncthreads();
                find_cross(hist, needC, tid, lane, &s_B, &s_m, &s_above, &s_tot);
                __syncthreads();
                aboveC += s_above; needC -= s_above; mm = s_m;
                uint32_t nlo = rlo + ((uint32_t)s_B << sh2);
                rhi = (s_B == NB - 1) ? rhi : (nlo + ((1u << sh2) - 1u));
                rlo = nlo;
            }
            __syncthreads();
            int it2 = (mc + BLK - 1) / BLK;
            for (int it = 0; it < it2; ++it) {
                int i = it * BLK + tid;
                bool v = i < mc;
                uint32_t u = v ? *(const uint32_t*)clus.map_shared_rank(ck + (i & (CMPC - 1)), i >> LCMPC) : 0u;
                int id = v ? *(const int*)clus.map_shared_rank(ci + (i & (CMPC - 1)), i >> LCMPC) : 0;
                bool q1 = v && ((long long)u > ethr);
                bool q2 = v && ((long long)u == ethr);
                unsigned n1 = __ballot_sync(FULLM, q1);
                unsigned n2 = __ballot_sync(FULLM, q2);
                unsigned b1 = 0, b2 = 0;
                if (lane == 0) {
                    if (n1) b1 = atomicAdd(&s_o1, (unsigned)__popc(n1));
                    if (n2) b2 = atomicAdd(&s_o2, (unsigned)__popc(n2));
                }
                b1 = __shfl_sync(FULLM, b1, 0);
                b2 = __shfl_sync(FULLM, b2, 0);
                if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u)); if (p < (unsigned)aboveC) O[above + p] = id; }
                if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u)); if (p < (unsigned)needC) O[above + aboveC + p] = id; }
            }
        } else {
            // safety net: crossing bin larger than the whole cluster buffer.
            // Exact key-space narrowing over the row, run by rank 0 alone.
            uint32_t rlo = 0u, rhi = 0xffffffffu;
            int above2 = 0, need2 = k, m2 = n;
            long long ethr = 0; bool tieM = true;
            for (int lev = 0; ; ++lev) {
                if (need2 == m2) { ethr = (long long)rlo - 1LL; above2 += m2; need2 = 0; tieM = false; break; }
                if (rlo >= rhi)  { ethr = (long long)rlo; break; }
                if (lev >= 8)    { ethr = (long long)rlo; break; }
                uint32_t d2 = rhi - rlo;
                int b2 = 32 - __clz(d2 | 1u);
                int sh2 = (b2 > LNB) ? (b2 - LNB) : 0;
                for (int i = tid; i < NB; i += BLK) hist[i] = 0u;
                __syncthreads();
                for (int i = tid; i < n; i += BLK) {
                    uint32_t u = fkey(X[i]);
                    if (u >= rlo && u <= rhi)
                        atomicAdd(&hist[min((unsigned)((u - rlo) >> sh2), (unsigned)(NB - 1))], 1u);
                }
                __syncthreads();
                find_cross(hist, need2, tid, lane, &s_B, &s_m, &s_above, &s_tot);
                __syncthreads();
                above2 += s_above; need2 -= s_above; m2 = s_m;
                uint32_t nlo = rlo + ((uint32_t)s_B << sh2);
                rhi = (s_B == NB - 1) ? rhi : (nlo + ((1u << sh2) - 1u));
                rlo = nlo;
            }
            __syncthreads();
            int nA = tieM ? above2 : k, nT = tieM ? need2 : 0;
            int it2 = (n + BLK - 1) / BLK;
            for (int it = 0; it < it2; ++it) {
                int i = it * BLK + tid;
                bool v = i < n;
                uint32_t u = v ? fkey(X[i]) : 0u;
                bool q1 = v && ((long long)u > ethr);
                bool q2 = v && tieM && ((long long)u == ethr);
                unsigned n1 = __ballot_sync(FULLM, q1);
                unsigned n2 = __ballot_sync(FULLM, q2);
                unsigned b1 = 0, b2 = 0;
                if (lane == 0) {
                    if (n1) b1 = atomicAdd(&s_o1, (unsigned)__popc(n1));
                    if (n2) b2 = atomicAdd(&s_o2, (unsigned)__popc(n2));
                }
                b1 = __shfl_sync(FULLM, b1, 0);
                b2 = __shfl_sync(FULLM, b2, 0);
                if (q1) { unsigned p = b1 + (unsigned)__popc(n1 & ((1u << lane) - 1u)); if (p < (unsigned)nA) O[p] = i; }
                if (q2) { unsigned p = b2 + (unsigned)__popc(n2 & ((1u << lane) - 1u)); if (p < (unsigned)nT) O[nA + p] = i; }
            }
        }
    }
    clus.sync();                                   // rank 0 done with DSMEM
}

template <int BLK, int VPT, int CS>
static inline void launch_regclus(int b, size_t smem, cudaStream_t stream,
                                  const float* logits, const int* pre_idx, int* out,
                                  int n, int npad, int k) {
    static bool init[GVR_MAX_DEV] = {};   /* attr is PER-DEVICE (B1c) */
    int dev_ = 0; cudaGetDevice(&dev_);
    bool& in_ = init[dev_ & (GVR_MAX_DEV - 1)];
    if (!in_) {
        cudaFuncSetAttribute(gvr_reg_clus<BLK, VPT, CS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 120 * 1024);
        if (CS > 8)
            cudaFuncSetAttribute(gvr_reg_clus<BLK, VPT, CS>,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        in_ = true;
    }
    gvr_reg_clus<BLK, VPT, CS><<<dim3(CS, b), BLK, smem, stream>>>(
        logits, pre_idx, out, n, npad, k);
}

/* The image variant needs more dynamic smem than the 48KB a plain launch may
   use, so it goes through a wrapper that opts in once per instantiation. */
template <int BLK, int VPT, int MINB, int NBV = NB, int KPTV = 1>
static inline void launch_regimg(int b, size_t smem, cudaStream_t stream,
                                 const float* logits, const int* pre_idx, int* out,
                                 int n, int npad, int k, int CMP, int IMGOFF, int QC) {
    static bool init[GVR_MAX_DEV] = {};   /* attr is PER-DEVICE (B1c) */
    int dev_ = 0; cudaGetDevice(&dev_);
    bool& in_ = init[dev_ & (GVR_MAX_DEV - 1)];
    if (!in_) {
        cudaFuncSetAttribute(gvr_topk_reg<BLK, VPT, MINB, KPTV, true, false, true, NBV>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 120 * 1024);
        in_ = true;
    }
    gvr_topk_reg<BLK, VPT, MINB, KPTV, true, false, true, NBV><<<b, BLK, smem, stream>>>(
        logits, pre_idx, out, n, npad, k, CMP, IMGOFF, QC);
}

template <int BLK, int U, int MINB, int NBS_, int CS>
static inline void launch_clus(int b, size_t smem, cudaStream_t stream,
                               const float* logits, const int* pre_idx, int* out,
                               int n, int npad, int k, int SCAP, int CMP,
                               int SMP, int TGT, int Q, int SS2, int TGT2) {
    static bool init[GVR_MAX_DEV] = {};   /* attr is PER-DEVICE (B1c) */
    int dev_ = 0; cudaGetDevice(&dev_);
    bool& in_ = init[dev_ & (GVR_MAX_DEV - 1)];
    if (!in_) {
        cudaFuncSetAttribute(gvr_clus<BLK, U, MINB, NBS_, CS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
        if (CS > 8)
            cudaFuncSetAttribute(gvr_clus<BLK, U, MINB, NBS_, CS>,
                                 cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        in_ = true;
    }
    gvr_clus<BLK, U, MINB, NBS_, CS><<<dim3(CS, b), BLK, smem, stream>>>(
        logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2);
}

/* B1a: the LAUNCH_REG2/LAUNCH_DEG macro family used to launch gvr_topk_reg
   DIRECTLY with default 48KB dynamic-smem ceiling; the DEG form widens CMP to n,
   so (2048+2n)*4 bytes blows the ceiling for k=2048 rows with n in [5069, 8256]
   (measured window; kernel static smem ~410B counts against the same 48KB).
   Route every register-family launch through this wrapper, which opts in once
   per instantiation exactly like the other launchers. */
template <int BLK, int VPT, int MINB, int KPT, bool CUR, bool DEG, bool IMGF, int NBH>
static inline void launch_reg_any(int b, size_t smem, cudaStream_t stream,
                                  const float* logits, const int* pre_idx, int* out,
                                  int n, int npad, int k, int CMP, int IMGOFF, int QC) {
    static bool init[GVR_MAX_DEV] = {};   /* attr is PER-DEVICE (B1c) */
    int dev_ = 0; cudaGetDevice(&dev_);
    bool& in_ = init[dev_ & (GVR_MAX_DEV - 1)];
    if (!in_) {
        cudaFuncSetAttribute(gvr_topk_reg<BLK, VPT, MINB, KPT, CUR, DEG, IMGF, NBH>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 120 * 1024);
        in_ = true;
    }
    gvr_topk_reg<BLK, VPT, MINB, KPT, CUR, DEG, IMGF, NBH><<<b, BLK, smem, stream>>>(
        logits, pre_idx, out, n, npad, k, CMP, IMGOFF, QC);
}

/* One lazily-initialised attribute set per instantiation. */
template <int BLK, int U, int MINB, int NBS_, int KPT, bool SPLIT>
static inline void launch_main(int gx, int gy, size_t smem, cudaStream_t stream,
                               const float* logits, const int* pre_idx, int* out,
                               int n, int npad, int k, int SCAP, int CMP, int R,
                               int SMP, int TGT, int Q, int SS2, int TGT2,
                               void* slabws) {
    static bool init[GVR_MAX_DEV] = {};   /* attr is PER-DEVICE (B1c) */
    int dev_ = 0; cudaGetDevice(&dev_);
    bool& in_ = init[dev_ & (GVR_MAX_DEV - 1)];
    if (!in_) {
        // 168KB: the KBIG (k=2048) non-split R==1 variant carries
        // (16384+4)*8 + (4096+1)*8 = 163.9KB of dynamic shared -- the old
        // 160KB cap made every v32 cell at b in (32,148] die with
        // cudaErrorInvalidValue (BS 64/128 band, wbv32p2 first launch).
        cudaFuncSetAttribute(gvr_main<BLK, U, MINB, NBS_, KPT, SPLIT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (BLK >= 1024 ? 168 : 100) * 1024);
        in_ = true;
    }
    gvr_main<BLK, U, MINB, NBS_, KPT, SPLIT><<<dim3(gx, gy), BLK, smem, stream>>>(
        logits, pre_idx, out, n, npad, k, SCAP, CMP, R, SMP, TGT, Q, SS2, TGT2, slabws);
}

cudaError_t gvr_topk_launch(const float* logits, const int* pre_idx, int* out,
                            int b, int n, int npad, int k, void* workspace,
                            cudaStream_t stream) {
    const bool wide = (b <= 148);
    {
        int n4 = n >> 2;
        // 2560 keeps dynamic smem at 24.6KB so 5 CTAs/SM fit in the 135KB
        // carveout; the crossing bin measures ~250 entries, far below it.
        // 2560 keeps dynamic smem at 24.6KB so 5 CTAs/SM fit in the 135KB
        // carveout; the crossing bin measures ~250 entries, far below it.
        int CMP = n < 2560 ? n : 2560;
        // The image aliases the not-yet-live crossing-bin buffer when it fits;
        // otherwise it is appended (only the 1-CTA/SM variant ever needs that).
        int IMGOFF = NB;   // set from NBSEL below
        const int QC = (b > 148) ? 1024 : QUADC;
        size_t smem;
        // KPT = ceil(k/BLK) hint slots per thread, resolved at compile time so
        // the prefetch array stays in registers.
        // Near-degenerate rows (n < 2k) at wide batch are the one place the
        // cursor emit loses: almost every element issues a shared atomic and the
        // machine is already saturated, so those keep the two-mask ballot emit.
        const bool CURE = !(n < 2 * k && b > 148);
        // Once the whole row already fits the candidate budget the pre_idx guess
        // cannot shrink the candidate set below that budget, so it is pure cost
        // -- k words of pre_idx plus k scattered gathers, or k shared atomicOr
        // in the bitmap form.  A formula of (n,k) only.
        // r2: n/k in (3, 4.06] still wins hint-free.  The measured pro_16k
        // cell (n=4099, k=1024, b=1024) misses the 3k gate by 3 elements and
        // pays 1024 scattered hop-2 sectors per row -- 2x the row's own bytes
        // -- plus two dependent round trips per wave, to cull less than half
        // the binning work of a row that is already register-resident.  CMP
        // still widens to n (below), so the exactness guarantee (crossing bin
        // can never overflow its buffer) is unchanged; the cost is 3 CTAs/SM
        // instead of 4 on that shape, which the removed DRAM traffic dwarfs.
        const bool DEGE = (n <= 3 * k) || (n <= 4 * k + 64);
        // DEG bins the WHOLE row, so its crossing bin can in principle hold
        // every element; the compaction buffer has to be able to take them.
        if (DEGE && CMP < n) CMP = n;
        // The bin count follows n: at n<=1027 a 1024-bin histogram holds ONE
        // element per bin, so its clear (4 stores/thread), its per-thread scan
        // span and -- worst -- the warp-0 find_cross walk (32 shared reads per
        // lane, with the other seven warps parked at the barrier) are all paid
        // 4x over for resolution nothing uses.  256 bins keep ~4 elements per
        // bin, far inside the crossing-bin budget.  A formula of n alone.
        // MEASURED: 256 bins here is DATA-DEPENDENT poison.  At k=512 of
        // n=1027 the k-th value sits at the MEDIAN of the row range -- the
        // densest part of the histogram -- so quartering the resolution
        // quadruples the crossing bin, pushes it past the O(mc^2) rank's
        // crossover and into the key-space narrowing fallback: +24%/+13% on
        // flash_4k b=1/b=64 while the k=1024 shape (whose threshold sits in the
        // sparse lower tail) gained 3%.  Resolution must be sized by the
        // DENSITY at rank k, which is not a function of (n,k); keep 1024.
        // r8-N2K: NCU (flash_64k_L38_bs64, register path) charges 49% of ALL
        // shared wavefronts to EXCESSIVE ones, split between the histogram
        // atomicAdd (20% of the instruction stream) and the emit's per-bin
        // cursor atomicAdd.  Both conflicts are SAME-ADDRESS: real indexer
        // logits are strongly peaked, so a warp's 32 values repeatedly land in
        // one bin.  Doubling the resolution halves the expected multiplicity
        // -- and halves the crossing bin, which keeps more rows inside the
        // O(mc^2) exact rank instead of the six-level narrowing fallback.
        // Only where a thread's scan span stays short (NBH/BLK <= 4) and the
        // row actually has elements to spread: at n<=2048 a 1024-bin histogram
        // is already at one element per bin.
        // r3 MEASURED: extending 2*NB to the wide-batch DEG shape (pro_16k,
        // n4 in (512,1024], b>148) is NEUTRAL (15.23 -> 15.18us, noise) --
        // the emit cursor conflicts it halves are not that cell's bottleneck.
        const int NBSEL = (n4 > 512 && !(n4 <= 1024 && !wide)) ? (2 * NB) : NB;
        IMGOFF = NBSEL;
        smem = (size_t)(NBSEL + 2 * CMP) * sizeof(uint32_t);
#define LAUNCH_REG2(BLKV, VPTV, MINBV, CURV, NBV)                                                \
        do {                                                                                \
            if (k <= (BLKV))                                                                \
                launch_reg_any<BLKV, VPTV, MINBV, 1, CURV, false, false, NBV>(b, smem, stream, \
                    logits, pre_idx, out, n, npad, k, CMP, IMGOFF, QC);                         \
            else if (k <= 2 * (BLKV))                                                       \
                launch_reg_any<BLKV, VPTV, MINBV, 2, CURV, false, false, NBV>(b, smem, stream, \
                    logits, pre_idx, out, n, npad, k, CMP, IMGOFF, QC);                         \
            else                                                                            \
                launch_reg_any<BLKV, VPTV, MINBV, 4, CURV, false, false, NBV>(b, smem, stream, \
                    logits, pre_idx, out, n, npad, k, CMP, IMGOFF, QC);                         \
        } while (0)
        // DEG never reads pre_idx, so it needs no KPT dimension at all.
#define LAUNCH_DEG(BLKV, VPTV, MINBV, NBV)                                                       \
        do { if (CURE)                                                                      \
                launch_reg_any<BLKV, VPTV, MINBV, 1, true,  true, false, NBV>(b, smem, stream, \
                    logits, pre_idx, out, n, npad, k, CMP, IMGOFF, QC);                         \
             else                                                                           \
                launch_reg_any<BLKV, VPTV, MINBV, 1, false, true, false, NBV>(b, smem, stream, \
                    logits, pre_idx, out, n, npad, k, CMP, IMGOFF, QC);                         \
        } while (0)
#define LAUNCH_REG(BLKV, VPTV, MINBV, NBV)                                                  \
        do { if (DEGE)      LAUNCH_DEG(BLKV, VPTV, MINBV, NBV);                             \
             else if (CURE) LAUNCH_REG2(BLKV, VPTV, MINBV, true, NBV);                      \
             else           LAUNCH_REG2(BLKV, VPTV, MINBV, false, NBV); } while (0)
        // The shared row image only exists for the LATENCY-bound wide-batch
        // variants (at most one CTA per SM, nothing to cover the second hint hop
        // with) and only when the guess stage is still live.  KPT is 1 there
        // because k <= 1024 == BLK, so it costs exactly one instantiation each.
        const int IMGW = (n + 3) & ~3;
        const size_t smi = (size_t)(NBSEL + (2 * CMP > IMGW ? 2 * CMP : IMGW)) * 4;
        const bool IMGE = wide && !DEGE && k <= 1024;
        // wide-batch (b>148) image: same row, same 24.6KB of shared (it aliases
        // the not-yet-live crossing-bin buffer), CURE is always true here.
        // MEASURED r8: enabling the image on the wide-batch VPT=2 variant costs
        // +2.2 to +2.6% -- at 4 CTAs/SM the scattered gather's latency is
        // already covered, and the image's own barrier is not.
        (void)0;
#define LAUNCH_REGIMG(BLKV, VPTV, MINBV) \
        launch_regimg<BLKV, VPTV, MINBV, 2 * NB>(b, smi, stream, logits, pre_idx, out,   \
                                         n, npad, k, CMP, IMGOFF, QC)
        if (n4 <= 256)       { LAUNCH_REG(256, 1, 8, NB); return cudaGetLastError(); }
        else if (n4 <= 512)  { LAUNCH_REG(512, 1, 4, NB); return cudaGetLastError(); }
        else if (n4 <= 1024) {
            // Wide batches keep the 256-thread/VPT=4 variant even though it is
            // register-capped at 4 CTAs/SM: raising occupancy via 1024-thread
            // CTAs costs 46% here, because the per-row fixed work (block
            // barriers, single-warp find_cross) is paid per CTA and 32-warp
            // CTAs quadruple the warps parked in it.
            if (wide) { if (IMGE) LAUNCH_REGIMG(1024, 1, 2); else LAUNCH_REG(1024, 1, 2, 2 * NB); }
            // The grid is exactly b CTAs, so at b=1024 the 256-thread/VPT=4
            // variant gets 5 CTAs/SM = 740 slots and needs TWO scheduling
            // rounds; NCU measures 48.4% achieved occupancy against a 62.5%
            // theoretical, i.e. the second round runs nearly empty.  Two rounds
            // are unavoidable below 7 CTAs/SM, and asking for 7 makes the
            // compiler fit 16 resident values under 36 registers and spill
            // (+44% measured).  So halve the RESIDENT VALUES instead: 512
            // threads x VPT=2 holds 8 values per thread, fits 4 CTAs/SM = 2048
            // threads/SM, and both rounds then run at full occupancy instead of
            // 62.5% and ~24%.
            else      { LAUNCH_REG(512, 2, 4, NB); }
            return cudaGetLastError(); }
        // ---- CLUSTERED register-resident path -------------------------------
        // A single CTA holds BLKC*VPT float4 in registers; a CLUSTER of CS CTAs
        // holds CS times that, so the CHEAP register algorithm (T = GMIN
        // directly, one float-space histogram, one register sweep -- no quantile
        // sample, no candidate staging, no verify retry) reaches rows up to
        // 131075.  Pick the SMALLEST VPT whose cluster still fits the SM budget:
        // that maximises CS, i.e. the number of SMs the row is spread over, and
        // it is CS -- not resident values -- that sets the row-load latency.
        // (r5 re-measured: flipping the boundary row n4 == 4096 into the
        // clustered register path costs 8.11 vs 7.68us on flash_64k_bs64 --
        // the cluster barriers + per-rank eager hint exceed the halved row
        // load at this size.  Keep the strict gate.)
        if (n4 > 4096 && n4 <= 8 * BLKC * 4 && k <= BLKC) {
            int av = 148 / (b > 0 ? b : 1);
            int amax = 1; while ((amax << 1) <= av && amax < 8) amax <<= 1;
            int vsel = 0, cs = 0;
            if (amax >= 2) {
                // knife4-L2w: the cs=8 co-residency veto is a PREFERENCE,
                // not a prohibition.  cs=8 clusters need 8 SMs inside ONE
                // GPC; B200's ~18-SM GPCs co-host at most 2 of them, so at
                // most 15 cs=8 clusters are resident machine-wide, and
                // b=16..18 x cs=8 spills to a SECOND cluster wave
                // (measured: flash_512k b15 10.1us -> b16 16.3us, +62%,
                // same kernel).  Pass 0 skips one-wave-exceeding cs=8 fits
                // so 128k/256k land on a deeper-VPT/smaller-cs single wave.
                // But where NO smaller-cs geometry exists (512k: pass 0
                // finds nothing), falling through to streaming is data-
                // dependent: 49/51 512k cells win 1.4-1.7x, yet retry-heavy
                // rows (pro_512k L46/L52, k4f verdict) explode 5x
                // (16 -> 79us).  Pass 1 re-runs the scan without the veto,
                // keeping the spilled-but-bounded cs=8 cluster exactly as
                // the shipped v32t dispatch would.
                for (int pass = 0; pass < 2 && !vsel; pass++) {
                    for (int v = 1; v <= 4; v <<= 1) {
                        int c = 1; while ((long long)c * BLKC * v < n4) c <<= 1;
                        if (pass == 0 && c == 8 && b > 15) continue;
                        if (c <= amax) { vsel = v; cs = c; break; }
                    }
                }
            }
            if (vsel && cs >= 2) {
                size_t smc = (size_t)(3 * NB + 2 * CMPC) * 4;
#define LAUNCH_RC(VPTV)                                                                     \
                do { if (cs == 2)      launch_regclus<BLKC, VPTV, 2>(b, smc, stream,        \
                                            logits, pre_idx, out, n, npad, k);              \
                     else if (cs == 4) launch_regclus<BLKC, VPTV, 4>(b, smc, stream,        \
                                            logits, pre_idx, out, n, npad, k);              \
                     else              launch_regclus<BLKC, VPTV, 8>(b, smc, stream,        \
                                            logits, pre_idx, out, n, npad, k); } while (0)
                if      (vsel == 1) LAUNCH_RC(1);
                else if (vsel == 2) LAUNCH_RC(2);
                else                LAUNCH_RC(4);
#undef LAUNCH_RC
                return cudaGetLastError();
            }
        }
        // (knife3 iter1 reverted: `wide` here means b <= 148 -- SMALL batch,
        // not large; routing it to streaming cost 1.65-1.77x on the 4k/8k
        // small-BS band and bought nothing.  The real k=2048 dense-corner fix
        // is the small_dense sampling gate on the !big streaming path below.)
        if (n4 <= 4096 && wide) {
            LAUNCH_REG(1024, 4, 1, 2 * NB);
            return cudaGetLastError(); }
#undef LAUNCH_REGIMG
    }
    // ---- streaming / collect path ----
    // Row splitting is a MEMORY-PARALLELISM decision, not a work decision: one
    // CTA per row leaves 148-b SMs idle and the row load then stalls on the
    // long scoreboard.  Aim for ~2 CTAs per SM.
    // MEASURED: letting the formula split b=64 into R=2 (128 CTAs instead of 64,
    // filling the idle 84 SMs and halving the row pass) LOST 16-22%.  Each extra
    // CTA re-pays the whole k-element scattered hint gather AND adds the ~3us
    // slab/select epilogue to the row's critical path; below b=32 there is no
    // other source of parallelism so it still pays, above it never does.
    int R = 1;
    if (b <= 32) {
        int r1 = 148 / b; if (r1 < 1) r1 = 1;
        // The depth cap used to be n/2048 == n4/512, i.e. HALF a 1024-thread CTA
        // of float4 per slice -- the worst possible landing.  A CTA issues its
        // row-pass loads, its compares, its 5-step slot-reservation shuffle scan
        // and its barriers for all 1024 threads regardless, so a 517-float4 slice
        // runs 32 warps to do 16 warps of work: the row pass costs the same
        // per SM as a full slice would, while the k-element scattered hint gather
        // and the quantile sample -- which NCU charges 43% of all global sectors
        // to at R>1 -- are paid TWICE as many times.  Cap the depth at one full
        // CTA of float4 instead, which halves the CTA count for the same row-pass
        // wall time.
        int r2 = (n >> 2) + 1023; r2 /= 1024; if (r2 < 1) r2 = 1;
        R = r1 < r2 ? r1 : r2;
        if (R < 1) R = 1;
    }
    // r11: b in [33,74] leaves half the SMs idle while a DEEP row streams at
    // half the machine's bandwidth.  A 2-CTA cluster split keeps 2*b <= 148
    // CTAs co-resident (the 64-reg x 1024-thread budget is exactly 1 CTA/SM),
    // halving the row-pass wall time.  The old n4 >= 32768 gate existed
    // because the split doubled the EAGER hint gather (n4 == 16384 measured
    // +4% on pro_256k_bs64 then); with the gather now lazy on gvr_clus the
    // doubled fixed cost is just the sample (~85 sectors) and the DSMEM
    // merge, which the halved 64KB row pass dwarfs -- gate at n4 >= 16384.
    // A formula of (n, k, b) only.
    else if (b <= 74 && (n >> 2) >= 16384 && k <= 1024) R = 2;
    // A SHALLOW split (R <= 8) can live inside one thread-block cluster, where
    // the cross-CTA histogram merge is CS distributed-shared reads and two
    // hardware cluster barriers instead of a global slab write, a threadfence,
    // an arrival RMW and a slab read-back on the last CTA's critical path.
    // Beyond 8 CTAs the cluster cannot express the split at all, and at b=1 the
    // DEPTH is worth more than the cheap merge (R=127 through the slab beats
    // R=8 through a cluster by 20%), so the slab keeps the deep splits.
    bool useclus = false;
    if (R >= 2 && R <= 8 && k <= 1024) { int p2 = 1; while ((p2 << 1) <= R) p2 <<= 1; R = p2; useclus = true; }
    const bool big = (b * R <= 148);          // 1024-thread variant, 1 CTA/SM
    // A WIDE accept window is what makes the quantile sample cheap: with
    // [k, SCAP] spanning >8x, a rung fed by ~40 sample hits never leaves it,
    // so the sample can be ~8x smaller than a narrow window needs.
    // Fixed by `big` alone so the kernel can treat them as compile-time
    // constants (see the note in gvr_main).  Every dispatch that reaches the
    // streaming path has n > 4*SCAP/4 and k <= 1024, so the old n/k clamps
    // never bound; dropping them costs nothing and buys constant-folded
    // shared-memory offsets throughout the kernel.
    // R==1 big staging doubled to 16K words (see the SCPB note in gvr_main):
    // must stay consistent with the kernel-side constexpr.
    // k > 1024 (v32 K=2048): must mirror the kernel-side KBIG constexpr
    // exactly -- the KPT the dispatch below picks satisfies KPT*BLK >= 2048
    // precisely when k > 1024, and SCPB/CMPB double there.
    const int SCAP = big ? ((R == 1) ? 16384 : 8192)
                         : ((k > 1024) ? 8192 : 4096);
    const int CMP  = big ? (((k > 1024) ? 4096 : 2048)) : 1024;
    // Sample cost ~ hits*n/aim, collect cost ~ aim: the balance point is
    // aim ~ sqrt(c*n), floored at 1.5k so an undershoot needs a 3.5-sigma miss.
    // r8-a004: the survivor WALK and the P5 candidate pass are both O(aim), and
    // NCU on pro_64k_L56_bs512 charges them 430 of 1265 warp-instructions (34%).
    // aim only has to clear k by enough that a sample-estimated rung never lands
    // UNDER it, so buying a bigger sample -- a few extra 32B sectors per row
    // against a 64KB row -- lets that margin shrink from 2.0x to 1.5x: at ~63
    // sample hits the relative error is 12.6%, so 1.5x is a 4-sigma miss where
    // 2.0x on ~32 hits was 5.6-sigma, and the accept test is exact either way so
    // a miss costs a retry, never correctness.  MEASURED -3.7% on the b>148
    // (4 CTAs/SM, BLK=256) variant and +1.1% on the 1-CTA/SM variant, whose walk
    // is a much smaller share of a 4x longer row -- so it is gated to the shape
    // it was measured on.
    // r1-weakband: the 1-CTA/SM R==1 rung target moves from 2k to 4k.  The
    // measured overshoot on the pro RETRY band is ~2.5x (sample said 2048,
    // row had 828 of k=1024): a 4x margin absorbs it and the first attempt
    // lands; the extra ~2k candidates are on-chip cost only, ~2% of a 131k
    // row pass.  Split paths keep their tuned targets.
    // r4 (a004 r3, measured): the R==1 4x-aim raise and the SFAC 64 sample
    // doubling exist for the k=1024 RETRY band; k=512 flash rows never
    // retried and pay the fatter sample plus a 2x longer walk for nothing.
    // Gate both on k >= 1024 (a formula of k only).
    // r5 (a005): the !big k>=1024 aim drops 3k/2 -> 11k/8 now that the TSH
    // ladder reaches the midband/wide variants (a rung miss retries bounded
    // instead of flooding to GMIN).  NCU charged 34% of pro_64k_L56_bs512 to
    // the survivor walk + P5, both O(aim); TGT ~ SFAC = 64 hits keeps the
    // trimmed margin at ~2.2 sigma with the ladder as backstop.
    int aim = big ? ((R == 1) ? ((k >= 1024) ? 4 * k : 2 * k) : (2 * k))
                  : ((k >= 1024) ? (11 * k) / 8 : (3 * k) / 2);
    { long long q = 6LL * (long long)n; int r = (int)(0.5 + std::sqrt((double)q));
      if (r > aim) aim = r; }
    // A split row pays the SAME quantile sample R times over (every CTA has to
    // derive the identical rung or the merged histogram is meaningless), so at
    // R>1 the sample is the dominant memory consumer -- NCU charged 43% of all
    // global sectors to it at b=1,n=65538, three times the row itself.  Trade
    // sample size for a higher aim: the [k,SCAP] accept window spans >8x, so a
    // 3.5k aim still lands from a ~24-hit estimate and the extra candidates are
    // paid on chip instead of in DRAM sectors.
    // r9: the aim inflation exists because a DEEP slab split (R up to 148) pays
    // the quantile sample R times over in DRAM sectors.  A CLUSTER split is at
    // most 8 CTAs, so the sample is at worst 8x -- a few KB against a 256KB row
    // -- while the inflated aim makes the survivor walk, the candidate store and
    // the whole P5 pass 1.75x longer, and NCU charges 27% of the kernel to those
    // three.  Give the cluster path the same budget the unsplit path gets.
    // r1-weakband: the big-variant sample doubles (SFAC 32 -> 64, ~16KB reads
    // against a 512KB row): TGT goes 31 -> ~64 so the rung estimate's relative
    // noise halves exactly where the RETRY band lives.
    // R==2 only arises from the r11 shallow cluster split: the deep-split
    // SFAC=16 rationale (sample paid R times over in DRAM) barely binds at 2x,
    // while the rung wants the evidence -- 48 keeps ~1.8k samples.
    // r2-v7: at R==2 the sample is only paid twice, but with the hint gather
    // gone (lazy) it IS the whole pre-row-pass critical chain -- shrink it.
    // aim 3.5k -> 3k and SFAC 48 -> 32 keep TGT ~32 hits: a rung overshoot to
    // below k needs a ~3x (6 sigma) miss, and the staged-candidate budget
    // (SCAP 8192) keeps 2.5x slack.
    // k > 1024 deep splits (v32): SFAC=16 leaves the rung estimate ~16 sample
    // hits (25% relative error) -- on flat-tail layers (128k L50/L51) the
    // estimate deterministically overshoots and every row cascades to the
    // GMIN backstop through the R-way slab (0.21x).  48 keeps ~3x the
    // evidence for a few KB of extra sample reads per CTA against a 512KB row.
    const int SFAC = (R > 1) ? (R == 2 ? 32 : ((k > 1024) ? 48 : 16))
                             : ((k >= 1024) ? 64 : 32);
    // r5 (a005): the R==2 floor trim 3k -> 2.5k measured -0.05..-0.10us/cell
    // under the PAIRED sample, but COMPOSED with the QUAD 64B-line sample it
    // detonates the L60 retry cell (15.58 -> 22.9us): the trimmed rung plus
    // the QUAD estimator's lower effective-N push the TSH retry past its
    // margin on heavy-tail layers.  The QUAD trio is worth more; keep 3k.
    { const int amin = (R == 2) ? 3 * k : (7 * k) / 2;
      if (R > 1 && aim < amin) aim = amin; }
    if (aim > (SCAP >> 1)) aim = SCAP >> 1;
    if (aim < k) aim = k;

    const int n4s = n >> 2;
    int SMP = 0, SS2 = 1, TGT = 0, TGT2 = 0;
    // k > 1024 small rows (v32 4k band): with n <= SCAP no sample rung forms,
    // so EVERY row pays the eager k=2048-gather hint and T=GMIN stages nearly
    // the whole row (k/n ~ 0.5 makes the hint floor worthless) -- measured
    // 0.61-0.86x vs SGLang v2 at BS>=256.  Let the sample rung form whenever
    // aim < n has room (n > 2k); k <= 1024 dispatches keep the old gate.
    const bool small_dense = (k > 1024) && !big && n <= SCAP && n > 2 * k;
    if ((n > SCAP || small_dense) && n4s >= 4) {
        // the sample is read as 32B-aligned float4 PAIRS: the second float4
        // rides in the same sector as the first, so it is free traffic.
        long long sel = (long long)SFAC * (long long)n / (long long)aim;
        if (sel < 256) sel = 256;
        if (sel > n / 2) sel = n / 2;
        int pairs = (int)(sel >> 3); if (pairs < 1) pairs = 1;
        int half = n4s >> 1; if (half < 1) half = 1;
        if (pairs > half) pairs = half;
        SS2 = half / pairs; if (SS2 < 1) SS2 = 1;
        SMP = half / SS2; if (SMP < 1) SMP = 1;
        TGT = (int)(((long long)aim * (long long)(SMP * 8)) / (long long)n);
        if (TGT < 1) TGT = 1;
        // second rung: the sample count that corresponds to ~k row elements,
        // i.e. an estimate of where rank k sits inside the candidate set.
        TGT2 = (int)(((long long)k * (long long)(SMP * 8)) / (long long)n);
        if (TGT2 < 1) TGT2 = 1;
    }
    int Q = (n4s + R - 1) / R;
    if (useclus) {
        // r5 (peer ab6a7302): clus-only QUAD sample geometry (see gvr_clus):
        // each location is one 64B line (4 float4, 16 elements) served by two
        // threads -- half the random-page activations at unchanged TGT.
        if (n > SCAP && n4s >= 4) {
            long long sel = (long long)SFAC * (long long)n / (long long)aim;
            if (sel < 256) sel = 256;
            if (sel > n / 2) sel = n / 2;
            int quads = (int)(sel >> 4); if (quads < 1) quads = 1;
            int quarter = n4s >> 2; if (quarter < 1) quarter = 1;
            if (quads > quarter) quads = quarter;
            SS2 = quarter / quads; if (SS2 < 1) SS2 = 1;
            SMP = quarter / SS2; if (SMP < 1) SMP = 1;
            TGT = (int)(((long long)aim * (long long)(SMP * 16)) / (long long)n);
            if (TGT < 1) TGT = 1;
            TGT2 = (int)(((long long)k * (long long)(SMP * 16)) / (long long)n);
            if (TGT2 < 1) TGT2 = 1;
        }
        // hist(NBS) | cbuf(SCAP) | ck64(CMP) | mrg(NBS)   (r5: hoff folded into merge_scan0)
        size_t smc = (size_t)SNB * 8 + (size_t)(SCAP + 4) * 8 + (size_t)CMP * 8;   // r4: int2 staging
        const int per = Q >> 10;
#define LAUNCH_CLUS(CSV)                                                                 \
        do {                                                                             \
            if      (per >= 8) launch_clus<1024, 8, 1, SNB, CSV>(b, smc, stream,         \
                logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2);    \
            else if (per >= 4) launch_clus<1024, 4, 1, SNB, CSV>(b, smc, stream,         \
                logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2);    \
            else if (per >= 2) launch_clus<1024, 2, 1, SNB, CSV>(b, smc, stream,         \
                logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2);    \
            else               launch_clus<1024, 1, 1, SNB, CSV>(b, smc, stream,         \
                logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2);    \
        } while (0)
        if      (R == 2) LAUNCH_CLUS(2);
        else if (R == 4) LAUNCH_CLUS(4);
        else             LAUNCH_CLUS(8);
#undef LAUNCH_CLUS
        return cudaGetLastError();
    }
    size_t smem = (size_t)(SCAP + 4) * ((R > 1 || b <= 296) ? 8 : 4) + (size_t)(CMP + 1) * 8;   // r4: VSTG(BLK>=512) int2 staging
#define LAUNCH_MAIN(BLKV, MINBV, UV, SPV)                                               \
    do {                                                                                \
        if (k <= (BLKV))                                                                \
            launch_main<BLKV, UV, MINBV, SNB, 1, SPV>(R, b, smem, stream,               \
                logits, pre_idx, out, n, npad, k, SCAP, CMP, R, SMP, TGT, Q, SS2, TGT2, workspace); \
        else if (k <= 2 * (BLKV))                                                       \
            launch_main<BLKV, UV, MINBV, SNB, 2, SPV>(R, b, smem, stream,               \
                logits, pre_idx, out, n, npad, k, SCAP, CMP, R, SMP, TGT, Q, SS2, TGT2, workspace); \
        else if (k <= 4 * (BLKV))                                                       \
            launch_main<BLKV, UV, MINBV, SNB, 4, SPV>(R, b, smem, stream,               \
                logits, pre_idx, out, n, npad, k, SCAP, CMP, R, SMP, TGT, Q, SS2, TGT2, workspace); \
        else                                                                            \
            /* k=2048-only (v32): KPT covers all k hint slots; at BLK=256 the   */      \
            /* old KPT=4 read only 1024 of them, so GMIN was a SUBSET min --    */      \
            /* biased HIGH -- and every hint-floored row undershot into the     */      \
            /* retry ladder.  k <= 1024 never reaches this branch (V4 domain    */      \
            /* dispatch is bit-identical).                                      */      \
            launch_main<BLKV, UV, MINBV, SNB, 8, SPV>(R, b, smem, stream,               \
                logits, pre_idx, out, n, npad, k, SCAP, CMP, R, SMP, TGT, Q, SS2, TGT2, workspace); \
    } while (0)
    // The unconditional loads only pay when the slice is deep enough to fill
    // the unroll: a split row whose CTA owns half a float4 per thread would
    // issue 8 loads for 1 useful one.  Size U by the slice depth.
    if (big) {
        const int per = Q >> 10;                       // float4 per thread at BLK=1024
        if (R > 1) {
            if      (per >= 8) LAUNCH_MAIN(1024, 1, 8, true);
            else if (per >= 4) LAUNCH_MAIN(1024, 1, 4, true);
            else if (per >= 2) LAUNCH_MAIN(1024, 1, 2, true);
            else               LAUNCH_MAIN(1024, 1, 1, true);
        } else {
            if      (per >= 8) LAUNCH_MAIN(1024, 1, 8, false);
            else if (per >= 4) LAUNCH_MAIN(1024, 1, 4, false);
            else if (per >= 2) LAUNCH_MAIN(1024, 1, 2, false);
            else               LAUNCH_MAIN(1024, 1, 1, false);
        }
    }
    // b in (148, 296]: 256-thread CTAs land 1-2 per SM = 22% occupancy and the
    // row load stalls on the long scoreboard.  512-thread CTAs at 2 CTAs/SM
    // double the resident warps for the same register budget (2*512*64 = 64K
    // regs exactly); b > 296 keeps the 256-thread variant, whose 4 CTAs/SM x
    // 148 = 592 slots still cover b=512 in ONE wave where 512-thread CTAs
    // would take two.
    else if (b <= 296) LAUNCH_MAIN(512, 2, 8, false);
    else               LAUNCH_MAIN(256, 4, 8, false);
#undef LAUNCH_MAIN
    return cudaGetLastError();
}
