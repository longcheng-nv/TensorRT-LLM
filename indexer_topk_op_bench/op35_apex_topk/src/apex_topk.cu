// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// op35 APEX-FR top-K v3 (iter12) — smem-staged filter + coalesced tail.
//
//   phase A (thresholds): stratified-jittered float4-granular sample
//     (deterministic per (seed,row,j)) -> 4-round 256-bin histogram order
//     statistic -> EXACT 32-bit t_lo = i_lo-th largest sample (rung0.2 band
//     math); t_hi at 16-bit (stats only). match_any warp-aggregated smem
//     atomics + warp0-parallel bin scan.
//   phase B (filter): v10 loop; admits staged as int2{bits,idx} into per-warp
//     regions of 96KB dynamic smem; ONE end-of-CTA flush: block prefix over
//     warp counts + 1 global atomicAdd reserve on cnt[1] + coalesced copy to
//     the row's dense candidate buffer. Region/global overflow -> cnt[2] flag.
//   phase C (tail): coalesced gather of M pairs -> dynamic smem, exact radix
//     select rank K with constant-leading-byte round skipping (kmin/kmax),
//     tie-aware warp-aggregated emission. miss (M<K) / overflow -> in-CTA
//     full-row radix fallback (exact, rare; covers degenerate rows).
//
// Launch modes (pick_config policy, ONE algorithm):
//   fused<NT>  — BS<=16 (multi-CTA/row, <=148 CTAs): A+B+C one launch,
//                last CTA of a row (global ticket) runs the tail.
//   split      — BS>=32: k_thr / k_filter / k_tail lean kernels so the filter
//                keeps v10-level registers (2 CTAs/SM at NT=1024).
// CUDA-graph compatible: no host sync; counters/tickets self-clean.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#define PAIR_CAP 12288                 // smem staging pairs (96KB dyn smem)
#define GCAP 32768                     // global candidate buffer cap per row
#define SEED_XOR(sd) (sd)
#define DYN_BYTES (PAIR_CAP * 8)

__device__ inline uint32_t f2u(float f) {
  uint32_t u = __float_as_uint(f);
  return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
__device__ inline uint32_t b2u(int bits) {
  uint32_t u = (uint32_t)bits;
  return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
__device__ inline float u2f(uint32_t u) {
  u ^= ((u & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu);
  return __uint_as_float(u);
}
__device__ inline uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ b * 0x85EBCA77u ^ c * 0xC2B2AE3Du;
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return h;
}

// dtype traits: every dtype loads 16B groups; EPL = elements per load
template <typename T>
__device__ inline uint4 ldraw(const T* base, long g) {
  return reinterpret_cast<const uint4*>(base)[g];
}
template <typename T> __device__ inline uint4 padraw();
template <> __device__ inline uint4 padraw<float>() {
  return make_uint4(0x7FC00000u, 0x7FC00000u, 0x7FC00000u, 0x7FC00000u);
}
template <> __device__ inline uint4 padraw<__nv_bfloat16>() {
  return make_uint4(0x7FC07FC0u, 0x7FC07FC0u, 0x7FC07FC0u, 0x7FC07FC0u);
}
template <> __device__ inline uint4 padraw<__half>() {
  return make_uint4(0x7E007E00u, 0x7E007E00u, 0x7E007E00u, 0x7E007E00u);
}
template <typename T> __device__ inline void cvtgrp(const uint4& r, float* out);
template <> __device__ inline void cvtgrp<float>(const uint4& r, float* out) {
  out[0] = __uint_as_float(r.x); out[1] = __uint_as_float(r.y);
  out[2] = __uint_as_float(r.z); out[3] = __uint_as_float(r.w);
}
template <> __device__ inline void cvtgrp<__nv_bfloat16>(const uint4& r, float* out) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const float2 f = __bfloat1622float2(h[t]);
    out[t * 2] = f.x; out[t * 2 + 1] = f.y;
  }
}
template <> __device__ inline void cvtgrp<__half>(const uint4& r, float* out) {
  const __half2* h = reinterpret_cast<const __half2*>(&r);
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const float2 f = __half22float2(h[t]);
    out[t * 2] = f.x; out[t * 2 + 1] = f.y;
  }
}
__device__ inline float to_f(float v) { return v; }
__device__ inline float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline float to_f(__half v) { return __half2float(v); }
template <typename T> struct EPL_OF { static constexpr int v = 8; };
template <> struct EPL_OF<float> { static constexpr int v = 4; };

// warp-aggregated histogram add; ALL 32 lanes must call (uniform control flow)
__device__ inline void hist_add(int* hist, int bin, bool sel) {
  const unsigned peers = __match_any_sync(~0u, sel ? bin : -1);
  if (sel && (int)(threadIdx.x & 31) == __ffs(peers) - 1)
    atomicAdd(&hist[bin], __popc(peers));
}

// warp0-parallel descending-cumulative bin search over hist[256]
__device__ inline void find_bin_warp(const int* hist, int rank, int* res) {
  const int lane = threadIdx.x & 31;
  const int base = 255 - 8 * lane;
  int c[8], mysum = 0;
#pragma unroll
  for (int t = 0; t < 8; ++t) { c[t] = hist[base - t]; mysum += c[t]; }
  int pre = mysum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(~0u, pre, o);
    if (lane >= o) pre += y;
  }
  pre -= mysum;
  if (rank >= pre && rank < pre + mysum) {
    int cc = pre;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      if (rank < cc + c[t]) { res[0] = base - t; res[1] = cc; break; }
      cc += c[t];
    }
  }
}

// ---------------- phase A device body ----------------
// Samples live in REGISTERS (s/NT <= 16 per thread) — no smem sample buffer,
// histogram rounds re-read registers (MIO relief). Histogram rounds skip
// leading bytes constant across all samples (kmin/kmax prefix): ~2-3 passes.
// Writes t_lo to sthr[0].
// warp0 two-level descending bin search: coarse cst[32] (64-bin groups) then
// the selected group's 64 fine bins (2/lane) — ~15 serial reads total.
__device__ void find_bin2k_warp(const int* hist, const int* cst, int rank, int* res) {
  const int lane = threadIdx.x & 31;
  // level 1: coarse group, descending (group g covers bins [64g, 64g+63])
  const int g = 31 - lane;              // lane 0 -> highest group
  const int csum = cst[g];
  int pre = csum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(~0u, pre, o);
    if (lane >= o) pre += y;
  }
  pre -= csum;  // count in groups above g
  const bool mine = (rank >= pre && rank < pre + csum);
  const unsigned who = __ballot_sync(~0u, mine);
  const int src_lane = __ffs(who) - 1;
  const int gsel = __shfl_sync(~0u, g, src_lane);
  const int above_g = __shfl_sync(~0u, pre, src_lane);
  // level 2: 64 fine bins of gsel, descending; lane l covers 2 bins
  const int b0 = gsel * 64 + 63 - 2 * lane;
  const int c0 = hist[b0], c1 = hist[b0 - 1];
  int fsum = c0 + c1;
  int fpre = fsum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(~0u, fpre, o);
    if (lane >= o) fpre += y;
  }
  fpre -= fsum;
  const int r2 = rank - above_g;
  if (r2 >= fpre && r2 < fpre + fsum) {
    if (r2 < fpre + c0) { res[0] = b0; res[1] = above_g + fpre; }
    else { res[0] = b0 - 1; res[1] = above_g + fpre + c0; }
  }
}

// Generic windowed exact-select: finds the (rank0+1)-th largest key and the
// count strictly above it, over n keys read as keys[p*stride] (smem or global;
// xf=true applies b2u to raw float bits). <=3-4 passes (>=11 bits/pass).
// ALL threads call; hist[2048], cst[32], sscal shared scratch.
template <int NT>
__device__ void window_select(const uint32_t* keys, int stride, int n, int rank0,
                              uint32_t kmin, uint32_t kmax, bool xf,
                              int* hist, int* cst, int* sscal,
                              uint32_t* out_kth, int* out_gt) {
  const int wid = threadIdx.x >> 5;
  uint32_t lo = kmin, hi = kmax;
  int rank = rank0;
  int gt_above = 0;
  while (lo < hi) {
    const uint32_t span = hi - lo;
    int hb = 31;
    while (!((span >> hb) & 1u)) --hb;
    const int shift = (hb >= 11) ? (hb - 10) : 0;
    for (int b = threadIdx.x; b < 2048; b += NT) hist[b] = 0;
    for (int b = threadIdx.x; b < 32; b += NT) cst[b] = 0;
    __syncthreads();
    const int npad = (n + NT - 1) / NT * NT;
    for (int p = threadIdx.x; p < npad; p += NT) {
      uint32_t u = 0;
      if (p < n) {
        u = keys[(long)p * stride];
        if (xf) u = b2u((int)u);
      }
      const bool sel = p < n && u >= lo && u <= hi;
      const int bin = sel ? (int)((u - lo) >> shift) : 0;
      hist_add(hist, bin, sel);
      hist_add(cst, bin >> 6, sel);
    }
    __syncthreads();
    if (wid == 0) find_bin2k_warp(hist, cst, rank, &sscal[0]);
    __syncthreads();
    const int bin = sscal[0];
    gt_above += sscal[1];
    rank -= sscal[1];
    const uint32_t nlo = lo + ((uint32_t)bin << shift);
    hi = (shift == 0) ? nlo : min(hi, nlo + (((uint32_t)1 << shift) - 1));
    lo = nlo;
    __syncthreads();
    if (shift == 0) break;
  }
  *out_kth = lo;
  *out_gt = gt_above;
}

template <int NT, typename T>
__device__ void dev_thresholds(const T* xr, long N, int row, int s,
                               int i_lo, uint32_t seed, int* hist_a, int* cst,
                               int* sscal, float* sthr) {
  constexpr int CMAX = 16;  // max samples per thread (s <= 16*NT)
  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  uint32_t sv[CMAX] = {};   // unowned slots stay 0 (never selected)
  if (threadIdx.x == 0) { sscal[2] = (int)0xFFFFFFFFu; sscal[3] = 0; }
  __syncthreads();
  uint32_t lmin = 0xFFFFFFFFu, lmax = 0u;
  // SCALAR strata (iter15): one independent element per stratum — no quad
  // correlation, honest IID order-statistic margins (sig x1, z=6)
  const float stride_s = (float)N / (float)s;
#pragma unroll
  for (int t = 0; t < CMAX; ++t) {
    const int j = threadIdx.x + t * NT;  // stratum
    if (j < s) {
      const uint32_t h = hash3(seed, (uint32_t)row, (uint32_t)j);
      const float u01 = (float)(h >> 8) * (1.0f / 16777216.0f);
      long idx = (long)(((float)j + u01) * stride_s);
      if (idx >= N) idx = N - 1;
      const uint32_t u = f2u(to_f(xr[idx]));
      sv[t] = u;
      lmin = min(lmin, u);
      lmax = max(lmax, u);
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    lmin = min(lmin, __shfl_down_sync(~0u, lmin, o));
    lmax = max(lmax, __shfl_down_sync(~0u, lmax, o));
  }
  if (lane == 0) {
    atomicMin((unsigned*)&sscal[2], lmin);
    atomicMax((unsigned*)&sscal[3], lmax);
  }
  __syncthreads();
  const uint32_t kmin = (uint32_t)sscal[2], kmax = (uint32_t)sscal[3];
  const uint32_t span = kmax - kmin;
  if (span == 0) {  // all samples equal (plateau/constant row)
    if (threadIdx.x == 0) sthr[0] = u2f(kmin);
    __syncthreads();
    return;
  }
  // single-pass 2048-bin window histogram over [kmin, kmax]
  int hb = 31;
  while (!((span >> hb) & 1u)) --hb;            // highest varying bit
  const int shift = (hb >= 11) ? (hb - 10) : 0; // (span>>shift) < 2048
  for (int b = threadIdx.x; b < 2048; b += NT) hist_a[b] = 0;
  for (int b = threadIdx.x; b < 32; b += NT) cst[b] = 0;
  __syncthreads();
#pragma unroll
  for (int t = 0; t < CMAX; ++t) {
    // single call site: match_any inside hist_add must see all 32 lanes
    const bool sel = (threadIdx.x + t * NT) < s;
    const int bin = sel ? (int)((sv[t] - kmin) >> shift) : 0;
    hist_add(hist_a, bin, sel);
    hist_add(cst, bin >> 6, sel);
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) find_bin2k_warp(hist_a, cst, i_lo, &sscal[0]);
  __syncthreads();
  // lower bin edge: <= exact i_lo-th sample value (strictly safer band)
  if (threadIdx.x == 0) sthr[0] = u2f(kmin + ((uint32_t)sscal[0] << shift));
  __syncthreads();
}

// ---------------- phase B device body ----------------
// Stages admits into per-warp regions of spair, then one block flush to the
// row's dense global candidate buffer. Returns is_last via sscal[7].
template <int NT, typename T>
__device__ void dev_filter(const T* xr, long N, float t_lo,
                           int ctas_per_row, int sub, int row, int2* spair,
                           int2* cand_row, int* cnt, int* tickets, int* wcnt,
                           int* wpref, int* sscal) {
  constexpr int EPL = EPL_OF<T>::v;  // elements per 16B load
  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarp = NT / 32;
  const int rcap = PAIR_CAP / nwarp;
  int2* wreg = spair + wid * rcap;
  const long ng = (N + EPL - 1) / EPL;
  const long ngfull = N / EPL;
  int wtot = 0;
  const long chunk = (ng + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(ng, begin + chunk);
  const long endf = min(end, ngfull);
  const long iters = (endf > begin) ? (endf - begin + NT - 1) / NT : 0;
  const uint4 PAD = padraw<T>();
  const long i0 = begin + threadIdx.x;
  uint4 a = (iters > 0 && i0 < endf) ? ldraw(xr, i0) : PAD;
  uint4 b = (iters > 1 && i0 + NT < endf) ? ldraw(xr, i0 + NT) : PAD;
  for (long it = 0; it < iters; ++it) {
    const long inx = i0 + (it + 2) * NT;
    uint4 nxt = (it + 2 < iters && inx < endf) ? ldraw(xr, inx) : PAD;
    const long i = i0 + it * NT;
    float vv[EPL];
    cvtgrp<T>(a, vv);
    int nb = 0;
#pragma unroll
    for (int j = 0; j < EPL; ++j) nb += (vv[j] >= t_lo);
    if (__any_sync(~0u, nb)) {
      int pre = nb;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(~0u, pre, o);
        if (lane >= o) pre += y;
      }
      const int total = __shfl_sync(~0u, pre, 31);
      int slot = wtot + pre - nb;
      wtot += total;
      int nsp = 0;
      float spv[EPL];
      int spi[EPL];
      if (nb) {
#pragma unroll
        for (int j = 0; j < EPL; ++j) {
          const float f = vv[j];
          if (f >= t_lo) {
            if (slot < rcap) wreg[slot] = make_int2(__float_as_int(f), (int)(i * EPL + j));
            else { spv[nsp] = f; spi[nsp] = (int)(i * EPL + j); ++nsp; }
            ++slot;
          }
        }
      }
      // graceful spill of region-overflow admits -> global (rare)
      if (__any_sync(~0u, nsp)) {
        int p2 = nsp;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const int y = __shfl_up_sync(~0u, p2, o);
          if (lane >= o) p2 += y;
        }
        const int tsp = __shfl_sync(~0u, p2, 31);
        p2 -= nsp;
        int base = 0;
        if (lane == 31) base = atomicAdd(&cnt[1], tsp);
        base = __shfl_sync(~0u, base, 31);
        for (int u = 0; u < nsp; ++u) {
          const int gpos = base + p2 + u;
          if (gpos < GCAP) cand_row[gpos] = make_int2(__float_as_int(spv[u]), spi[u]);
        }
      }
    }
    a = b; b = nxt;
  }
  // partial last group (N % EPL != 0), once, by warp 0 of the owning CTA
  if ((N % EPL) && ngfull >= begin && ngfull < end && wid == 0) {
    const long e = ngfull * EPL + lane;
    const float f = (lane < (int)(N % EPL)) ? to_f(xr[e]) : CUDART_NAN_F;
    const bool adm = (f >= t_lo);
    const unsigned bal = __ballot_sync(~0u, adm);
    const int rk = __popc(bal & ((1u << lane) - 1));
    const int slot = wtot + rk;
    const bool sp = adm && slot >= rcap;
    if (adm && !sp) wreg[slot] = make_int2(__float_as_int(f), (int)e);
    const unsigned bsp = __ballot_sync(~0u, sp);
    if (bsp) {
      int base = 0;
      if (lane == 0) base = atomicAdd(&cnt[1], __popc(bsp));
      base = __shfl_sync(~0u, base, 0);
      if (sp) {
        const int gpos = base + __popc(bsp & ((1u << lane) - 1));
        if (gpos < GCAP) cand_row[gpos] = make_int2(__float_as_int(f), (int)e);
      }
    }
    wtot += __popc(bal);
  }
  // block flush: prefix over warp counts -> 1 global reserve -> coalesced copy
  if (lane == 0) wcnt[wid] = wtot;
  __syncthreads();
  if (threadIdx.x == 0) {
    int tot = 0;
    for (int w = 0; w < nwarp; ++w) {
      wpref[w] = tot;
      tot += min(wcnt[w], rcap);  // region overflow already spilled to global
    }
    sscal[6] = atomicAdd(&cnt[1], tot);
  }
  __syncthreads();
  const int gb = sscal[6];
  const int m = min(wcnt[wid], rcap);
  const int wp = wpref[wid];
  for (int e = lane; e < m; e += 32) {
    const int gpos = gb + wp + e;
    if (gpos < GCAP) cand_row[gpos] = wreg[e];
  }
  // ticket (release)
  if (threadIdx.x == 0) {
    __threadfence();
    const int t = atomicAdd(&tickets[row], 1);
    sscal[7] = (t == ctas_per_row - 1);
    if (t == ctas_per_row - 1) tickets[row] = 0;  // self-clean
  }
  __syncthreads();
}

// ---------------- phase C device body ----------------// ---------------- phase C device body ----------------
template <int NT, typename T>
__device__ void dev_tail(const T* xr, long N, int K, int cap,
                         const int2* cand_row, int* cnt, int* out_row,
                         int2* spair, int* hist_a, int* cst, int* sscal,
                         int row, int* dbg) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  long tg0 = 0, tg1 = 0, tg2 = 0, tg3 = 0;
  if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tg0));
  __threadfence();  // acquire other CTAs' candidate data
  if (threadIdx.x == 0) {
    sscal[4] = __ldcg(&cnt[1]);
    sscal[2] = (int)0xFFFFFFFFu;  // kmin (as uint via atomicMin)
    sscal[3] = 0;                 // kmax
  }
  __syncthreads();
  const int M = sscal[4];
  const bool bad = M < K || M > GCAP;      // full-row fallback (rare)
  const bool bigm = !bad && M > cap;       // radix/emit direct from global
  if (!bad && !bigm) {  // coalesced gather + local min/max
    uint32_t lmin = 0xFFFFFFFFu, lmax = 0u;
    for (int p = threadIdx.x; p < M; p += NT) {
      const int2 pr = __ldcg(&cand_row[p]);
      const uint32_t u = b2u(pr.x);
      spair[p] = make_int2((int)u, pr.y);
      lmin = min(lmin, u);
      lmax = max(lmax, u);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {  // uniform: all lanes ran the loop bounds
      lmin = min(lmin, __shfl_down_sync(~0u, lmin, o));
      lmax = max(lmax, __shfl_down_sync(~0u, lmax, o));
    }
    if (lane == 0) {
      atomicMin((unsigned*)&sscal[2], lmin);
      atomicMax((unsigned*)&sscal[3], lmax);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tg1));
  if (dbg && threadIdx.x == 0) {
    dbg[row * 8] = M;
    dbg[row * 8 + 1] = (bigm ? 4 : 0) | (M < K ? 2 : 0) | (M > GCAP ? 1 : 0);
  }

  uint32_t u_kth;
  int count_gt, need_eq;
  if (!bad && bigm) {  // direct global select (M <= GCAP)
    window_select<NT>(reinterpret_cast<const uint32_t*>(cand_row), 2, M, K - 1,
                      0u, 0xFFFFFFFFu, true, hist_a, cst, sscal, &u_kth, &count_gt);
    need_eq = K - count_gt;
  } else {
    uint32_t prefix = 0;
    int rank = K - 1;
    int gt_total = 0;
    int shift0 = 24;
    if (!bad) {  // byte-skip via kmin/kmax (staged path)
      const uint32_t kmin = (uint32_t)sscal[2], kmax = (uint32_t)sscal[3];
      while (shift0 > 0 && ((kmin >> shift0) & 255u) == ((kmax >> shift0) & 255u)) {
        prefix |= kmin & (255u << shift0);
        shift0 -= 8;
      }
    }
    for (int shift = shift0; shift >= 0; shift -= 8) {
      __syncthreads();
      for (int bb = threadIdx.x; bb < 256; bb += NT) hist_a[bb] = 0;
      __syncthreads();
      const uint32_t pmask = (shift >= 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
      if (!bad) {
        const int Mpad = (M + NT - 1) / NT * NT;
        for (int p = threadIdx.x; p < Mpad; p += NT) {
          const uint32_t u = (p < M) ? (uint32_t)spair[p].x : 0u;
          hist_add(hist_a, (u >> shift) & 255,
                   p < M && (u & pmask) == (prefix & pmask));
        }
      } else {
        const long Npad = (N + NT - 1) / NT * NT;
        for (long e = threadIdx.x; e < Npad; e += NT) {
          const float f = (e < N) ? to_f(xr[e]) : CUDART_NAN_F;
          const uint32_t u = f2u(f);
          hist_add(hist_a, (u >> shift) & 255,
                   f == f && (u & pmask) == (prefix & pmask));
        }
      }
      __syncthreads();
      if (wid == 0) find_bin_warp(hist_a, rank, &sscal[0]);
      __syncthreads();
      gt_total += sscal[1];
      rank -= sscal[1];
      prefix |= ((uint32_t)sscal[0] << shift);
    }
    u_kth = prefix;
    count_gt = gt_total;
    need_eq = K - count_gt;
  }
  if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tg2));

  // emission (tie-aware; warp-aggregated counters)
  if (threadIdx.x == 0) { sscal[4] = 0; sscal[5] = 0; }
  __syncthreads();
  if (!bad) {
    const int Mpad = (M + NT - 1) / NT * NT;
    for (int p = threadIdx.x; p < Mpad; p += NT) {
      int2 pr = make_int2(0, 0);
      if (p < M) {
        if (bigm) { pr = __ldcg(&cand_row[p]); pr.x = (int)b2u(pr.x); }
        else pr = spair[p];
      }
      const uint32_t u = (uint32_t)pr.x;
      const bool sgt = p < M && u > u_kth;
      const bool seq = p < M && u == u_kth;
      const unsigned bgt = __ballot_sync(~0u, sgt);
      const unsigned beq = __ballot_sync(~0u, seq);
      int base_gt = 0, base_eq = 0;
      if (lane == 0) {
        if (bgt) base_gt = atomicAdd(&sscal[4], __popc(bgt));
        if (beq) base_eq = atomicAdd(&sscal[5], __popc(beq));
      }
      base_gt = __shfl_sync(~0u, base_gt, 0);
      base_eq = __shfl_sync(~0u, base_eq, 0);
      const unsigned lmask = (1u << lane) - 1;
      if (sgt) out_row[base_gt + __popc(bgt & lmask)] = pr.y;
      if (seq) {
        const int slot = base_eq + __popc(beq & lmask);
        if (slot < need_eq) out_row[count_gt + slot] = pr.y;
      }
    }
  } else {
    const long Npad = (N + NT - 1) / NT * NT;
    for (long e = threadIdx.x; e < Npad; e += NT) {
      const float f = (e < N) ? to_f(xr[e]) : CUDART_NAN_F;
      const uint32_t u = f2u(f);
      const bool ok = (f == f);
      const bool sgt = ok && u > u_kth;
      const bool seq = ok && u == u_kth;
      const unsigned bgt = __ballot_sync(~0u, sgt);
      const unsigned beq = __ballot_sync(~0u, seq);
      int base_gt = 0, base_eq = 0;
      if (lane == 0) {
        if (bgt) base_gt = atomicAdd(&sscal[4], __popc(bgt));
        if (beq) base_eq = atomicAdd(&sscal[5], __popc(beq));
      }
      base_gt = __shfl_sync(~0u, base_gt, 0);
      base_eq = __shfl_sync(~0u, base_eq, 0);
      const unsigned lmask = (1u << lane) - 1;
      if (sgt) out_row[base_gt + __popc(bgt & lmask)] = (int)e;
      if (seq) {
        const int slot = base_eq + __popc(beq & lmask);
        if (slot < need_eq) out_row[count_gt + slot] = (int)e;
      }
    }
  }
  __syncthreads();
  if (dbg && threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tg3));
    dbg[row * 8 + 4] = (int)(tg1 - tg0);  // gather ns
    dbg[row * 8 + 5] = (int)(tg2 - tg1);  // radix ns
    dbg[row * 8 + 6] = (int)(tg3 - tg2);  // emit ns
  }
  // self-clean counts for graph replay
  if (threadIdx.x == 0) { cnt[0] = 0; cnt[1] = 0; cnt[2] = 0; }
}

// ---------------- kernels ----------------
template <int NT, typename T>
__global__ void k_apex_fused(const T* __restrict__ xbase, int* __restrict__ out,
                             int2* __restrict__ cand, int* __restrict__ counts,
                             int* __restrict__ tickets, long row_stride, long N,
                             int K, int ctas_per_row, int s, int i_lo,
                             uint32_t seed, int mode, int* __restrict__ dbg) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const T* xr = xbase + (long)row * row_stride;
  int2* cand_row = cand + (long)row * GCAP;
  int* cnt = counts + (long)row * 3;
  extern __shared__ int2 spair[];
  __shared__ int hist_a[2048];  // phase A window hist; tail uses first 256
  __shared__ int cst[32];
  __shared__ int wcnt[32], wpref[32];
  __shared__ int sscal[8];
  __shared__ float sthr[2];
  dev_thresholds<NT, T>(xr, N, row, s, i_lo, seed, hist_a, cst, sscal, sthr);
  const float t_lo = sthr[0];
  if (mode == 1) {
    if (threadIdx.x == 0 && sub == 0 && dbg) {
      dbg[row * 8 + 2] = __float_as_int(t_lo);
    }
    return;
  }
  dev_filter<NT, T>(xr, N, t_lo, ctas_per_row, sub, row, spair, cand_row,
                 cnt, tickets, wcnt, wpref, sscal);
  if (mode == 2) return;  // probe (probe script resets counts)
  if (!sscal[7]) return;
  dev_tail<NT, T>(xr, N, K, PAIR_CAP, cand_row, cnt, out + (long)row * K, spair,
               hist_a, cst, sscal, row, dbg);
}

template <typename T>
__global__ void k_apex_thr(const T* __restrict__ xbase,
                           float* __restrict__ thr, long row_stride, long N,
                           int s, int i_lo, uint32_t seed) {
  constexpr int NT = 512;
  const int row = blockIdx.x;
  const T* xr = xbase + (long)row * row_stride;
  __shared__ int hist_a[2048];
  __shared__ int cst[32];
  __shared__ int sscal[8];
  __shared__ float sthr[2];
  dev_thresholds<NT, T>(xr, N, row, s, i_lo, seed, hist_a, cst, sscal, sthr);
  if (threadIdx.x == 0) thr[row] = sthr[0];
}

template <int NT, typename T>
__global__ void k_apex_filter(const T* __restrict__ xbase,
                              const float* __restrict__ thr,
                              int2* __restrict__ cand, int* __restrict__ counts,
                              int* __restrict__ tickets, long row_stride, long N,
                              int ctas_per_row) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const T* xr = xbase + (long)row * row_stride;
  extern __shared__ int2 spair[];
  __shared__ int wcnt[32], wpref[32];
  __shared__ int sscal[8];
  dev_filter<NT, T>(xr, N, thr[row], ctas_per_row, sub, row,
                 spair, cand + (long)row * GCAP, counts + (long)row * 3,
                 tickets, wcnt, wpref, sscal);
}

template <typename T>
__global__ void k_apex_tail(const T* __restrict__ xbase, int* __restrict__ out,
                            const int2* __restrict__ cand, int* __restrict__ counts,
                            long row_stride, long N, int K, int tail_cap,
                            int* __restrict__ dbg) {
  constexpr int NT = 512;
  const int row = blockIdx.x;
  const T* xr = xbase + (long)row * row_stride;
  extern __shared__ int2 spair[];
  __shared__ int hist_a[2048];
  __shared__ int cst[32];
  __shared__ int sscal[8];
  dev_tail<NT, T>(xr, N, K, tail_cap, cand + (long)row * GCAP,
               counts + (long)row * 3, out + (long)row * K, spair, hist_a,
               cst, sscal, row, dbg);
}

// small-N mode: whole row staged in smem as ordered keys; exact window select
// + emission, no sampling / no candidate buffers / no miss path. grid = rows.
__global__ void k_apex_small(const float* __restrict__ xbase, int* __restrict__ out,
                             long row_stride, long N, int K) {
  constexpr int NT = 512;
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const float* xr = xbase + (long)row * row_stride;
  const float4* xr4 = reinterpret_cast<const float4*>(xr);
  extern __shared__ uint32_t skey[];
  __shared__ int hist[2048], cst[32], sscal[8];
  if (threadIdx.x == 0) { sscal[2] = (int)0xFFFFFFFFu; sscal[3] = 0; }
  __syncthreads();
  const long n4full = N / 4;
  uint32_t lmin = 0xFFFFFFFFu, lmax = 0u;
  for (long i = threadIdx.x; i < n4full; i += NT) {
    const float4 v = xr4[i];
    const uint32_t u0 = f2u(v.x), u1 = f2u(v.y), u2 = f2u(v.z), u3 = f2u(v.w);
    skey[i * 4] = u0; skey[i * 4 + 1] = u1;
    skey[i * 4 + 2] = u2; skey[i * 4 + 3] = u3;
    lmin = min(min(min(lmin, u0), min(u1, u2)), u3);
    lmax = max(max(max(lmax, u0), max(u1, u2)), u3);
  }
  for (long e = n4full * 4 + threadIdx.x; e < N; e += NT) {  // partial tail
    const uint32_t u = f2u(xr[e]);
    skey[e] = u;
    lmin = min(lmin, u);
    lmax = max(lmax, u);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    lmin = min(lmin, __shfl_down_sync(~0u, lmin, o));
    lmax = max(lmax, __shfl_down_sync(~0u, lmax, o));
  }
  if (lane == 0) {
    atomicMin((unsigned*)&sscal[2], lmin);
    atomicMax((unsigned*)&sscal[3], lmax);
  }
  __syncthreads();
  uint32_t u_kth;
  int count_gt;
  window_select<NT>(skey, 1, (int)N, K - 1, (uint32_t)sscal[2],
                    (uint32_t)sscal[3], false, hist, cst, sscal, &u_kth, &count_gt);
  const int need_eq = K - count_gt;
  if (threadIdx.x == 0) { sscal[4] = 0; sscal[5] = 0; }
  __syncthreads();
  int* out_row = out + (long)row * K;
  const int n = (int)N;
  const int npad = (n + NT - 1) / NT * NT;
  for (int p = threadIdx.x; p < npad; p += NT) {
    const uint32_t u = (p < n) ? skey[p] : 0u;
    const bool sgt = p < n && u > u_kth;
    const bool seq = p < n && u == u_kth;
    const unsigned bgt = __ballot_sync(~0u, sgt);
    const unsigned beq = __ballot_sync(~0u, seq);
    int base_gt = 0, base_eq = 0;
    if (lane == 0) {
      if (bgt) base_gt = atomicAdd(&sscal[4], __popc(bgt));
      if (beq) base_eq = atomicAdd(&sscal[5], __popc(beq));
    }
    base_gt = __shfl_sync(~0u, base_gt, 0);
    base_eq = __shfl_sync(~0u, base_eq, 0);
    const unsigned lmask = (1u << lane) - 1;
    if (sgt) out_row[base_gt + __popc(bgt & lmask)] = p;
    if (seq) {
      const int slot = base_eq + __popc(beq & lmask);
      if (slot < need_eq) out_row[count_gt + slot] = p;
    }
  }
}

static void set_dyn(const void* fn, int bytes) {
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

template <typename T>
static void launch_apex(const T* xp, torch::Tensor& out, torch::Tensor& cand,
                        torch::Tensor& counts, torch::Tensor& tickets,
                        torch::Tensor& thr, int rows, long row_stride, long N,
                        int K, int ctas_per_row, int nt, int s, int i_lo,
                        long seed, int tail_cap, int mode, bool split, int* dp) {
  static bool init_done = false;
  if (!init_done) {
    set_dyn((const void*)k_apex_fused<512, T>, DYN_BYTES);
    set_dyn((const void*)k_apex_filter<512, T>, DYN_BYTES);
    set_dyn((const void*)k_apex_filter<1024, T>, DYN_BYTES);
    set_dyn((const void*)k_apex_tail<T>, DYN_BYTES);
    init_done = true;
  }
  if (!split) {
    k_apex_fused<512, T><<<rows * ctas_per_row, 512, DYN_BYTES>>>(
        xp, out.data_ptr<int>(), reinterpret_cast<int2*>(cand.data_ptr<int>()),
        counts.data_ptr<int>(), tickets.data_ptr<int>(), row_stride, N, K,
        ctas_per_row, s, i_lo, (uint32_t)SEED_XOR(seed), mode, dp);
    return;
  }
  k_apex_thr<T><<<rows, 512>>>(xp, thr.data_ptr<float>(), row_stride, N, s,
                               i_lo, (uint32_t)SEED_XOR(seed));
  if (mode < 2) return;
  if (nt == 1024)
    k_apex_filter<1024, T><<<rows * ctas_per_row, 1024, DYN_BYTES>>>(
        xp, thr.data_ptr<float>(), reinterpret_cast<int2*>(cand.data_ptr<int>()),
        counts.data_ptr<int>(), tickets.data_ptr<int>(), row_stride, N,
        ctas_per_row);
  else
    k_apex_filter<512, T><<<rows * ctas_per_row, 512, DYN_BYTES>>>(
        xp, thr.data_ptr<float>(), reinterpret_cast<int2*>(cand.data_ptr<int>()),
        counts.data_ptr<int>(), tickets.data_ptr<int>(), row_stride, N,
        ctas_per_row);
  if (mode < 3) return;
  k_apex_tail<T><<<rows, 512, tail_cap * 8>>>(
      xp, out.data_ptr<int>(),
      reinterpret_cast<const int2*>(cand.data_ptr<int>()),
      counts.data_ptr<int>(), row_stride, N, K, tail_cap, dp);
}

void apex_topk(torch::Tensor x, torch::Tensor out, torch::Tensor cand,
               torch::Tensor counts, torch::Tensor tickets, torch::Tensor thr,
               long N, int K, int ctas_per_row, int nt, int s, int i_lo,
               long seed, int tail_cap, int mode, bool split, torch::Tensor dbg) {
  const int rows = x.size(0);
  const long row_stride = x.size(1);
  int* dp = dbg.numel() ? dbg.data_ptr<int>() : nullptr;
  if (x.scalar_type() == torch::kFloat32)
    launch_apex<float>(x.data_ptr<float>(), out, cand, counts, tickets, thr,
                       rows, row_stride, N, K, ctas_per_row, nt, s, i_lo, seed,
                       tail_cap, mode, split, dp);
  else if (x.scalar_type() == torch::kBFloat16)
    launch_apex<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        out, cand, counts, tickets, thr, rows, row_stride, N, K, ctas_per_row,
        nt, s, i_lo, seed, tail_cap, mode, split, dp);
  else if (x.scalar_type() == torch::kHalf)
    launch_apex<__half>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        out, cand, counts, tickets, thr, rows, row_stride, N, K, ctas_per_row,
        nt, s, i_lo, seed, tail_cap, mode, split, dp);
  else
    TORCH_CHECK(false, "apex_topk: unsupported dtype");
}

void apex_small(torch::Tensor x, torch::Tensor out, long N, int K) {
  static bool init = false;
  if (!init) {
    set_dyn((const void*)k_apex_small, 32768 * 4);
    init = true;
  }
  const int rows = x.size(0);
  const int dynb = (int)(((N + 3) / 4) * 16);
  k_apex_small<<<rows, 512, dynb>>>(x.data_ptr<float>(), out.data_ptr<int>(),
                                    x.size(1), N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apex_topk", &apex_topk);
  m.def("apex_small", &apex_small);
}
