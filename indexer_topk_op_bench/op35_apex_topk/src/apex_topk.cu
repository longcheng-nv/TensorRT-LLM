// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// op35 APEX-FR top-K v2 (iter11) — single fused kernel:
//   phase A: stratified-jittered float4-granular sample (redundant per CTA;
//            deterministic per (seed,row,j)) -> 4-round 256-bin histogram
//            order statistic -> EXACT 32-bit t_lo = i_lo-th largest sample
//            (matches rung0.2 band math); t_hi at 16-bit (stats only).
//            match_any warp-aggregated smem atomics + warp0-parallel bin scan.
//   phase B: v10 filter body (clean hot loop; partial float4 handled once,
//            outside the loop, by warp 0 of the owning CTA). Admits stored as
//            int2{float_bits, idx} single 8B store into per-warp segments.
//   phase C: last-CTA tail — segment counts to warr, ONE gather pass staging
//            int2{ordkey, idx} pairs into 64KB dynamic smem, exact 4-round
//            radix select rank K (tie-aware), warp-aggregated emission from
//            smem (no global re-scan). miss (M<K) / overflow -> in-CTA
//            full-row radix fallback (exact, rare).
// CUDA-graph compatible: no host sync; counters/tickets self-clean.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define SBUF_CAP 8192                    // staged candidate pairs / samples
#define DYN_BYTES (SBUF_CAP * 8)         // 64KB dynamic smem (int2 pairs)
#define WARR_CAP 2432                    // per-segment counts cache (nseg <= 2368)

__device__ inline uint32_t f2u(float f) {
  uint32_t u = __float_as_uint(f);
  return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
__device__ inline uint32_t b2u(int bits) {  // raw float bits -> ordered uint
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

// warp-aggregated histogram add. ALL 32 lanes must call (uniform control
// flow); non-participants pass sel=false.
__device__ inline void hist_add(int* hist, int bin, bool sel) {
  const unsigned peers = __match_any_sync(~0u, sel ? bin : -1);
  if (sel && (int)(threadIdx.x & 31) == __ffs(peers) - 1)
    atomicAdd(&hist[bin], __popc(peers));
}

// warp0-parallel descending-cumulative bin search over hist[256]:
// res[0] = bin containing 0-based top-rank `rank`; res[1] = count above it.
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

template <int NT>
__global__ void k_apex_topk(const float* __restrict__ xbase, int* __restrict__ out,
                            int2* __restrict__ cand, int* __restrict__ counts,
                            int* __restrict__ tickets, long row_stride, long N,
                            int K, int ctas_per_row, int s, int i_hi, int i_lo,
                            uint32_t seed, int segcap, int mode,
                            int* __restrict__ dbg) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarp = NT / 32;
  const int nseg = nwarp * ctas_per_row;
  const long cap = (long)nseg * segcap;
  const float* xr = xbase + (long)row * row_stride;
  const float4* xr4 = reinterpret_cast<const float4*>(xr);
  const int seg = sub * nwarp + wid;
  int2* cw = cand + (long)row * cap + (long)seg * segcap;
  int* cnt = counts + (long)row * (2 + nseg);
  int* out_row = out + (long)row * K;

  extern __shared__ int2 spair[];  // tail pairs; phase A reuses as sample words
  uint32_t* samp = reinterpret_cast<uint32_t*>(spair);
  __shared__ int warr[WARR_CAP];
  __shared__ int hist_a[256], hist_b[256];
  __shared__ int sscal[8];  // 0:bin 1:above 2:bin2 3:above2 4:cnt 5:flag 7:is_last
  __shared__ float sthr[2];  // t_hi, t_lo

  // ---------------- phase A: sample + exact band thresholds ----------------
  // float4-granular strata: 4 samples per 32B sector (quad correlation is
  // absorbed by the z=6 margin + exact fallback).
  const int s4 = s / 4;
  const long n4s = N / 4;
  const float stride4_f = (float)n4s / (float)s4;
  for (int j = threadIdx.x; j < s4; j += NT) {
    const uint32_t h = hash3(seed, (uint32_t)row, (uint32_t)j);
    const float u01 = (float)(h >> 8) * (1.0f / 16777216.0f);
    long idx = (long)(((float)j + u01) * stride4_f);
    if (idx >= n4s) idx = n4s - 1;
    const float4 v = xr4[idx];
    samp[j * 4] = f2u(v.x);
    samp[j * 4 + 1] = f2u(v.y);
    samp[j * 4 + 2] = f2u(v.z);
    samp[j * 4 + 3] = f2u(v.w);
  }
  for (int b = threadIdx.x; b < 256; b += NT) hist_a[b] = 0;
  __syncthreads();
  // round 1: bits 31:24 (shared by both ranks); s % NT == 0 (host asserts)
  for (int j = threadIdx.x; j < s; j += NT) hist_add(hist_a, samp[j] >> 24, true);
  __syncthreads();
  if (wid == 0) find_bin_warp(hist_a, i_hi, &sscal[0]);
  if (wid == (nwarp > 1 ? 1 : 0)) find_bin_warp(hist_a, i_lo, &sscal[2]);
  __syncthreads();
  const int bhi1 = sscal[0], above_hi = sscal[1];
  const int blo1 = sscal[2];
  int above_lo = sscal[3];
  uint32_t pre_lo = (uint32_t)blo1 << 24;
  for (int b = threadIdx.x; b < 256; b += NT) { hist_a[b] = 0; hist_b[b] = 0; }
  __syncthreads();
  // round 2: bits 23:16 for both ranks (t_hi finishes here, 16-bit edge)
  for (int j = threadIdx.x; j < s; j += NT) {
    const uint32_t u = samp[j];
    const int top = u >> 24;
    hist_add(hist_a, (u >> 16) & 255, top == bhi1);
    hist_add(hist_b, (u >> 16) & 255, top == blo1);
  }
  __syncthreads();
  if (wid == 0) find_bin_warp(hist_a, i_hi - above_hi, &sscal[0]);
  if (wid == (nwarp > 1 ? 1 : 0)) find_bin_warp(hist_b, i_lo - above_lo, &sscal[2]);
  __syncthreads();
  const uint32_t thi_u = ((uint32_t)bhi1 << 24) | ((uint32_t)sscal[0] << 16);
  above_lo += sscal[3];
  pre_lo |= (uint32_t)sscal[2] << 16;
  // rounds 3-4: refine t_lo to the exact 32-bit sample value
  for (int shift = 8; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < 256; b += NT) hist_a[b] = 0;
    __syncthreads();
    const uint32_t pmask = 0xFFFFFFFFu << (shift + 8);
    for (int j = threadIdx.x; j < s; j += NT) {
      const uint32_t u = samp[j];
      hist_add(hist_a, (u >> shift) & 255, (u & pmask) == pre_lo);
    }
    __syncthreads();
    if (wid == 0) find_bin_warp(hist_a, i_lo - above_lo, &sscal[2]);
    __syncthreads();
    above_lo += sscal[3];
    pre_lo |= (uint32_t)sscal[2] << shift;
    __syncthreads();
  }
  if (threadIdx.x == 0) { sthr[0] = u2f(thi_u); sthr[1] = u2f(pre_lo); }
  __syncthreads();
  const float t_hi = sthr[0];
  const float t_lo = sthr[1];
  if (mode == 1) {  // phase-A-only probe
    if (threadIdx.x == 0 && sub == 0 && dbg) {
      dbg[row * 8 + 2] = __float_as_int(t_lo);
      dbg[row * 8 + 3] = __float_as_int(t_hi);
    }
    return;
  }

  // ---------------- phase B: v10 filter (clean hot loop) ----------------
  const long n4 = (N + 3) / 4;
  const long n4full = N / 4;
  int c_hi = 0;
  int wtot = 0;
  const long chunk = (n4 + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4, begin + chunk);
  const long endf = min(end, n4full);  // main loop covers full slots only
  const long iters = (endf > begin) ? (endf - begin + NT - 1) / NT : 0;
  const float4 PAD = make_float4(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
  const long i0 = begin + threadIdx.x;
  float4 a = (iters > 0 && i0 < endf) ? xr4[i0] : PAD;
  float4 b = (iters > 1 && i0 + NT < endf) ? xr4[i0 + NT] : PAD;
  for (long it = 0; it < iters; ++it) {
    const long inx = i0 + (it + 2) * NT;
    float4 nxt = (it + 2 < iters && inx < endf) ? xr4[inx] : PAD;
    const float4 v = a;
    const long i = i0 + it * NT;
    const int nb4 = (v.x >= t_lo) + (v.y >= t_lo) + (v.z >= t_lo) + (v.w >= t_lo);
    if (__any_sync(~0u, nb4)) {
      int pre = nb4;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(~0u, pre, o);
        if (lane >= o) pre += y;
      }
      const int total = __shfl_sync(~0u, pre, 31);
      int slot = wtot + pre - nb4;
      wtot += total;
      if (nb4) {
        const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float f = vv[j];
          if (f >= t_lo) {
            c_hi += (f >= t_hi);
            if (slot < segcap) cw[slot] = make_int2(__float_as_int(f), (int)(i * 4 + j));
            ++slot;
          }
        }
      }
    }
    a = b; b = nxt;
  }
  // partial last float4 slot (N % 4 != 0), handled once by warp 0 of its CTA
  if ((N & 3) && n4full >= begin && n4full < end && wid == 0) {
    const long e = n4full * 4 + lane;
    const float f = (lane < (int)(N & 3)) ? xr[e] : CUDART_NAN_F;
    const bool adm = (f >= t_lo);
    const unsigned bal = __ballot_sync(~0u, adm);
    if (adm) {
      const int rk = __popc(bal & ((1u << lane) - 1));
      const int slot = wtot + rk;
      c_hi += (f >= t_hi);
      if (slot < segcap) cw[slot] = make_int2(__float_as_int(f), (int)e);
    }
    wtot += __popc(bal);
  }
  if (lane == 0) cnt[2 + seg] = wtot;
  {
    __shared__ int sh[NT / 32];
    for (int o = 16; o; o >>= 1) c_hi += __shfl_down_sync(~0u, c_hi, o);
    if (lane == 0) sh[wid] = c_hi;
    __syncthreads();
    if (threadIdx.x == 0) {
      int ch = 0;
      for (int w = 0; w < nwarp; ++w) ch += sh[w];
      atomicAdd(&cnt[0], ch);
      __threadfence();
      const int t = atomicAdd(&tickets[row], 1);
      sscal[7] = (t == ctas_per_row - 1);
      if (t == ctas_per_row - 1) tickets[row] = 0;  // self-clean
    }
    __syncthreads();
  }
  if (mode == 2) return;  // phase-A+B probe (probe script resets counts)
  if (!sscal[7]) return;

  // ---------------- phase C: tail (last CTA of the row) ----------------
  long tg0 = 0, tg1 = 0, tg2 = 0, tg3 = 0;
  if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tg0));
  __threadfence();  // acquire other CTAs' segment data
  if (threadIdx.x == 0) { sscal[4] = 0; sscal[5] = 0; }
  __syncthreads();
  for (int sg = threadIdx.x; sg < nseg; sg += NT) {
    const int w = __ldcg(&cnt[2 + sg]);
    if (w > segcap) atomicExch(&sscal[5], 1);
    warr[sg] = min(w, segcap);
  }
  __syncthreads();
  // single gather pass: stage {ordkey, idx} pairs into dynamic smem
  if (nseg >= NT) {
    // straggler-proof: block exclusive scan of warr -> flat gather with
    // per-entry binary search (loads pipeline; no serial per-segment chains)
    __shared__ int wpart[NT / 32];
    const int C = (nseg + NT - 1) / NT;      // segs per thread (chunked)
    const int s0 = threadIdx.x * C;
    int loc = 0;
#pragma unroll 4
    for (int t = 0; t < C; ++t) {
      const int sg = s0 + t;
      if (sg < nseg) loc += warr[sg];
    }
    int pre = loc;  // warp inclusive scan
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(~0u, pre, o);
      if (lane >= o) pre += y;
    }
    if (lane == 31) wpart[wid] = pre;
    __syncthreads();
    if (wid == 0) {  // scan warp partials (nwarp <= 32)
      int v = (lane < nwarp) ? wpart[lane] : 0;
      int p2 = v;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(~0u, p2, o);
        if (lane >= o) p2 += y;
      }
      if (lane < nwarp) wpart[lane] = p2 - v;  // exclusive
      if (lane == 31) sscal[4] = p2;           // total M
    }
    __syncthreads();
    int off = wpart[wid] + pre - loc;  // exclusive offset of this thread's chunk
    // overwrite warr with exclusive prefix (soff)
    int soff_local[8];  // C <= ceil(2368/512)=5; bound 8
#pragma unroll 4
    for (int t = 0; t < C; ++t) {
      const int sg = s0 + t;
      if (sg < nseg) {
        soff_local[t] = off;
        off += warr[sg];
      }
    }
    __syncthreads();
#pragma unroll 4
    for (int t = 0; t < C; ++t) {
      const int sg = s0 + t;
      if (sg < nseg) warr[sg] = soff_local[t];
    }
    __syncthreads();
    const int M0 = min(sscal[4], SBUF_CAP);
    const int2* rowc = cand + (long)row * cap;
    for (int p = threadIdx.x; p < M0; p += NT) {
      // find segment: greatest sg with soff[sg] <= p
      int lo = 0, hi = nseg - 1;
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (warr[mid] <= p) lo = mid;
        else hi = mid - 1;
      }
      const int2 pr = __ldcg(&rowc[(long)lo * segcap + (p - warr[lo])]);
      spair[p] = make_int2((int)b2u(pr.x), pr.y);
    }
  } else {  // warp-per-segment (lane-parallel, coalesced)
    for (int sg = wid; sg < nseg; sg += nwarp) {
      const int m = warr[sg];
      if (!m) continue;
      int base = 0;
      if (lane == 0) base = atomicAdd(&sscal[4], m);
      base = __shfl_sync(~0u, base, 0);
      const int2* sv = cand + (long)row * cap + (long)sg * segcap;
      for (int e = lane; e < m; e += 32) {
        if (base + e < SBUF_CAP) {
          const int2 pr = __ldcg(&sv[e]);
          spair[base + e] = make_int2((int)b2u(pr.x), pr.y);
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tg1));
  const int M = sscal[4];
  const bool bad = sscal[5] || M < K || M > SBUF_CAP;
  if (dbg && threadIdx.x == 0) {
    dbg[row * 8] = M;
    dbg[row * 8 + 1] = (sscal[5] ? 4 : 0) | (M < K ? 2 : 0) | (M > SBUF_CAP ? 1 : 0);
  }

  uint32_t u_kth;
  int count_gt, need_eq;
  {
    uint32_t prefix = 0;
    int rank = K - 1;  // 0-based from top
    int gt_total = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
      __syncthreads();
      for (int bb = threadIdx.x; bb < 256; bb += NT) hist_a[bb] = 0;
      __syncthreads();
      const uint32_t pmask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
      if (!bad) {  // radix over staged pairs in smem (loop padded to NT)
        const int Mpad = (M + NT - 1) / NT * NT;
        for (int p = threadIdx.x; p < Mpad; p += NT) {
          const uint32_t u = (p < M) ? (uint32_t)spair[p].x : 0u;
          hist_add(hist_a, (u >> shift) & 255,
                   p < M && (u & pmask) == (prefix & pmask));
        }
      } else {  // fallback: radix over the full row (rare)
        const long Npad = (N + NT - 1) / NT * NT;
        for (long e = threadIdx.x; e < Npad; e += NT) {
          const float f = (e < N) ? xr[e] : CUDART_NAN_F;
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

  // -------- emission (tie-aware; warp-aggregated counters, from smem) -------
  if (threadIdx.x == 0) { sscal[4] = 0; sscal[5] = 0; }
  __syncthreads();
  if (!bad) {
    const int Mpad = (M + NT - 1) / NT * NT;
    for (int p = threadIdx.x; p < Mpad; p += NT) {
      const int2 pr = (p < M) ? spair[p] : make_int2(0, 0);
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
      const float f = (e < N) ? xr[e] : CUDART_NAN_F;
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
  for (int p = threadIdx.x; p < 2 + nseg; p += NT) cnt[p] = 0;
}

void apex_topk(torch::Tensor x, torch::Tensor out, torch::Tensor cand,
               torch::Tensor counts, torch::Tensor tickets, long N, int K,
               int ctas_per_row, int nt, int s, int i_hi, int i_lo, long seed,
               int segcap, int mode, torch::Tensor dbg) {
  const int rows = x.size(0);
  const long row_stride = x.size(1);
  int* dp = dbg.numel() ? dbg.data_ptr<int>() : nullptr;
  static bool init512 = false, init1024 = false;
  if (nt == 1024) {
    if (!init1024) {
      cudaFuncSetAttribute(k_apex_topk<1024>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_BYTES);
      init1024 = true;
    }
    k_apex_topk<1024><<<rows * ctas_per_row, 1024, DYN_BYTES>>>(
        x.data_ptr<float>(), out.data_ptr<int>(),
        reinterpret_cast<int2*>(cand.data_ptr<int>()), counts.data_ptr<int>(),
        tickets.data_ptr<int>(), row_stride, N, K, ctas_per_row, s, i_hi, i_lo,
        (uint32_t)seed, segcap, mode, dp);
  } else {
    if (!init512) {
      cudaFuncSetAttribute(k_apex_topk<512>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_BYTES);
      init512 = true;
    }
    k_apex_topk<512><<<rows * ctas_per_row, 512, DYN_BYTES>>>(
        x.data_ptr<float>(), out.data_ptr<int>(),
        reinterpret_cast<int2*>(cand.data_ptr<int>()), counts.data_ptr<int>(),
        tickets.data_ptr<int>(), row_stride, N, K, ctas_per_row, s, i_hi, i_lo,
        (uint32_t)seed, segcap, mode, dp);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("apex_topk", &apex_topk); }
