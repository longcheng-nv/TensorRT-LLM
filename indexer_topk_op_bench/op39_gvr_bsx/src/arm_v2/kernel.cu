// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// op39 iter4.5 production arm v2: hint-thresholded fused 1-pass collect top-K
// with a SECOND-CHANCE rescue pass for deep-hint overflow rows.
//
// K0 thresh: t_lo[row] = min hint value (count >= K guaranteed at h < 1).
// K1 arm<false>: tile collect >= t_lo; last CTA per row:
//   - fast path (stored candidates complete): exact 4-level bucket top-K.
//   - overflow (clipped stage/CAP): t2 = exact K-th value of the STORED
//     subset (kth(stored) >= kth(row) and the K stored top values are >= t2,
//     so a rescan at t2 yields C in [K, K+ties] candidates containing the
//     true top-K); write thr[row] = t2, set rescue[row] = 1.
// K2 arm<true>: rows with rescue flag only (near-zero cost otherwise):
//   re-collect at t2 and reduce; if even that overflows (massive value ties),
//   final resort = single-CTA full-row exact reduce. Counters self-clean.
#include <cuda_runtime.h>
#include <cstdint>

#ifndef CAP
#define CAP 8192
#endif
#define STAGE 4096

__device__ __forceinline__ unsigned mono_key(float x) {
  unsigned k = __float_as_uint(x);
  return (k & 0x80000000u) ? ~k : (k | 0x80000000u);
}
__device__ __forceinline__ float inv_mono(unsigned k) {
  unsigned r = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
  return __uint_as_float(r);
}

struct RedSmemT {
  unsigned hist[256];
  unsigned wsum[16];
  unsigned thr_bucket;
  int ocur, tcur;
};

// r-th largest key of a smem sample array (whole CTA); 4-level 8-bit descent.
__device__ void sample_kth_key(const float* sv, int n, int need0,
                               RedSmemT& S, unsigned* key_out) {
  const int lane = threadIdx.x & 31;
  unsigned prefix = 0;
  int need = need0;
  for (int lvl = 0; lvl < 4; ++lvl) {
    const int shift = 24 - lvl * 8;
    for (int j = threadIdx.x; j < 256; j += blockDim.x) S.hist[j] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      unsigned k = mono_key(sv[j]);
      if (lvl == 0 || (k >> (shift + 8)) == prefix)
        atomicAdd(&S.hist[(k >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    if (warp < 8) {
      unsigned v = S.hist[warp * 32 + lane];
      for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
      if (!lane) S.wsum[warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      // warp-parallel suffix search: lane l holds wsum[7-l] (l<8), suffix-scan
      unsigned wv = (lane < 8) ? S.wsum[7 - lane] : 0u;
      unsigned ws = wv;
      for (int o = 1; o < 8; o <<= 1) {
        unsigned p = __shfl_up_sync(0xffffffffu, ws, o);
        if (lane >= o) ws += p;
      }
      unsigned gm = __ballot_sync(0xffffffffu, lane < 8 && ws >= (unsigned)need);
      int gl = __ffs(gm) - 1;
      int g = 7 - gl;
      unsigned above_g = __shfl_sync(0xffffffffu, ws - wv, gl);
      unsigned hv = S.hist[g * 32 + 31 - lane];
      unsigned hs = hv;
      for (int o = 1; o < 32; o <<= 1) {
        unsigned p = __shfl_up_sync(0xffffffffu, hs, o);
        if (lane >= o) hs += p;
      }
      unsigned bm = __ballot_sync(0xffffffffu, above_g + hs >= (unsigned)need);
      int bl = __ffs(bm) - 1;
      unsigned hs_b = __shfl_sync(0xffffffffu, hs - hv, bl);
      if (lane == 0) {
        S.thr_bucket = g * 32 + 31 - bl;
        S.ocur = (int)(above_g + hs_b);
      }
    }
    __syncthreads();
    need -= S.ocur;
    __syncthreads();
    prefix = (lvl == 0 ? S.thr_bucket : (prefix << 8) | S.thr_bucket);
    if (lvl == 1) {  // 16-bit precision suffices for a sampling threshold;
                     // pad with zeros = bucket lower bound (conservative)
      if (threadIdx.x == 0) *key_out = prefix << 16;
      return;
    }
  }
}

extern "C" __global__ void __launch_bounds__(512)
thresh_kernel(const float* __restrict__ logits, const int* __restrict__ pre_idx,
              float* __restrict__ thr, int npad, int K) {
  const int row = blockIdx.x;
  const float* lg = logits + (size_t)row * npad;
  const int* pre = pre_idx + (size_t)row * K;
  float m = __int_as_float(0x7f800000);
  const int hstep = (K + 511) / 512;   // subsample hints: <=512 gathers
  for (int j = threadIdx.x * hstep; j < K; j += blockDim.x * hstep) {
    int idx = pre[j];
    if (idx >= 0 && idx < npad) m = fminf(m, lg[idx]);
  }
  __shared__ float s[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o; o >>= 1) m = fminf(m, __shfl_down_sync(0xffffffffu, m, o));
  if (!lane) s[warp] = m;
  __syncthreads();
  __shared__ float s_min;
  if (!warp) {
    m = (lane < (int)(blockDim.x >> 5)) ? s[lane] : __int_as_float(0x7f800000);
    for (int o = 16; o; o >>= 1) m = fminf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (!lane) s_min = m;
  }
  __syncthreads();
  float t = s_min;
  if (npad > 2 * K) {
    // position-unbiased strided sample; select the r-th largest so that the
    // expected candidate count lands near 4608 (in [K, CAP] for all K<=2048)
    // clustered sampling: PROBES cache lines, 32 consecutive values each —
    // touches 1/32 of the row's DRAM lines (a flat stride equal to the line
    // size read the ENTIRE row's lines: 46us at BS256 1M-npad, falsified)
    const int PROBES = (npad <= 65600) ? 128 : 256;
    const int S_N = PROBES * 32;
    __shared__ float sv[256 * 32];
    __shared__ RedSmemT SS;
    __shared__ unsigned skey;
    const float4* lg4s = reinterpret_cast<const float4*>(lg);
    const int n4s = npad / 4;
    const unsigned pstep = ((unsigned)(n4s - 8) << 8) / PROBES;  // fp 24.8, float4 units
    for (int j = threadIdx.x; j < PROBES * 8; j += blockDim.x) {
      int probe = j >> 3, off = j & 7;
      float4 v = lg4s[(((unsigned)probe * pstep) >> 8) + off];
      sv[j * 4 + 0] = v.x; sv[j * 4 + 1] = v.y;
      sv[j * 4 + 2] = v.z; sv[j * 4 + 3] = v.w;
    }
    __syncthreads();
    // primary target: npad-adaptive — as tight as 1.5K when the sample rank
    // resolution allows, floored at rank>=48 for tail precision; always clear
    // of STAGE (T==STAGE trips per-chunk clips -> rescue storms)
    int T_pri = max(K + K / 2, (int)((long)48 * npad / S_N));
    T_pri = min(T_pri, min(STAGE - 256, 3 * CAP / 4));
    long r = (long)S_N * T_pri / npad;
    int rk = (int)max(1L, min((long)S_N, r));
    sample_kth_key(sv, S_N, rk, SS, &skey);
    __syncthreads();
    t = fmaxf(t, inv_mono(skey));
    // fallback quantile at ~6K target: used by the undershoot rescue instead
    // of a full-row final resort (6x margin over K absorbs cluster noise)
    const int T_fb = min(6 * K, 3 * CAP / 4);  // fallback must fit CAP
    long rf = (long)S_N * T_fb / npad;
    int rkf = (int)max((long)rk + 1, min((long)S_N, rf));
    __syncthreads();
    sample_kth_key(sv, S_N, rkf, SS, &skey);
    __syncthreads();
    if (threadIdx.x == 0) thr[gridDim.x + row] = fminf(inv_mono(skey), s_min);
  } else if (threadIdx.x == 0) {
    thr[gridDim.x + row] = t;  // small rows: fallback == primary (min-hint)
  }
  if (threadIdx.x == 0) thr[row] = t;
}

struct RedSmem {
  unsigned hist[256];
  unsigned wsum[16];
  unsigned thr_bucket;
  int ocur, tcur;
};

// Exact top-K (EMIT) or exact-kth-key search (!EMIT) via 4-level 8-bit bucket
// refinement. Level 0 scans the full source; boundary-bucket survivors are
// compacted into smem ping-pong buffers so levels 1-3 touch only survivors
// (typically <<n). If survivors exceed the buffer, falls back to full scans.
// Whole CTA participates. Tie-exact in the value-multiset sense.
#define SURV 2048
template <bool FROM_ROW, bool EMIT>
__device__ void exact_topk_from(const float* __restrict__ src_val,
                                const int* __restrict__ src_idx,
                                const float* __restrict__ lg, int n, int K,
                                int* __restrict__ orow, RedSmem& S,
                                unsigned* kth_key_out,
                                float* sbufv, int* sbufi) {
  const int lane = threadIdx.x & 31;
  unsigned prefix = 0;
  int need = K;
  int emitted = 0;
  // survivor buffers: two halves of (sbufv,sbufi); level l>=1 reads cur, writes nxt
  int cur = -1;              // -1 = reading from the original source
  int n_cur = n;
  bool spill = (sbufv == nullptr);  // no buffer -> always full scans
  for (int lvl = 0; lvl < 4; ++lvl) {
    const int shift = 24 - lvl * 8;
    for (int j = threadIdx.x; j < 256; j += blockDim.x) S.hist[j] = 0;
    if (threadIdx.x == 0) S.tcur = 0;
    __syncthreads();
    const bool from_buf = (cur >= 0) && !spill;
    const float* bv = sbufv + cur * SURV;
    const int* bi = sbufi + cur * SURV;
    int nn = from_buf ? n_cur : n;
    for (int j = threadIdx.x; j < nn; j += blockDim.x) {
      float x = from_buf ? bv[j] : (FROM_ROW ? lg[j] : src_val[j]);
      unsigned k = mono_key(x);
      if (from_buf || lvl == 0 || (k >> (shift + 8)) == prefix)
        atomicAdd(&S.hist[(k >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    if (warp < 8) {
      unsigned v = S.hist[warp * 32 + lane];
      for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
      if (!lane) S.wsum[warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      unsigned wv = (lane < 8) ? S.wsum[7 - lane] : 0u;
      unsigned ws = wv;
      for (int o = 1; o < 8; o <<= 1) {
        unsigned p = __shfl_up_sync(0xffffffffu, ws, o);
        if (lane >= o) ws += p;
      }
      unsigned gm = __ballot_sync(0xffffffffu, lane < 8 && ws >= (unsigned)need);
      int gl = __ffs(gm) - 1;
      int g = 7 - gl;
      unsigned above_g = __shfl_sync(0xffffffffu, ws - wv, gl);
      unsigned hv = S.hist[g * 32 + 31 - lane];
      unsigned hs = hv;
      for (int o = 1; o < 32; o <<= 1) {
        unsigned p = __shfl_up_sync(0xffffffffu, hs, o);
        if (lane >= o) hs += p;
      }
      unsigned bm = __ballot_sync(0xffffffffu, above_g + hs >= (unsigned)need);
      int bl = __ffs(bm) - 1;
      unsigned hs_b = __shfl_sync(0xffffffffu, hs - hv, bl);
      if (lane == 0) {
        S.thr_bucket = g * 32 + 31 - bl;
        S.ocur = (int)(above_g + hs_b);
      }
    }
    __syncthreads();
    unsigned tb = S.thr_bucket;
    int above = S.ocur;
    int nxt = (cur == 0) ? 1 : 0;
    float* nv = sbufv + nxt * SURV;
    int* ni = sbufi + nxt * SURV;
    __shared__ int scur;
    if (threadIdx.x == 0) scur = 0;
    __syncthreads();
    // combined emit(+survivor-compact) pass
    int npadded = (nn + blockDim.x - 1) / blockDim.x * blockDim.x;
    for (int j = threadIdx.x; j < npadded; j += blockDim.x) {
      bool inset = false;
      unsigned k = 0;
      float x = 0.f;
      int idx = 0;
      if (j < nn) {
        x = from_buf ? bv[j] : (FROM_ROW ? lg[j] : src_val[j]);
        k = mono_key(x);
        inset = from_buf || lvl == 0 || (k >> (shift + 8)) == prefix;
        idx = from_buf ? bi[j] : (FROM_ROW ? j : src_idx[j]);
      }
      bool hi = inset && ((k >> shift) & 0xffu) > tb;
      bool bd = inset && ((k >> shift) & 0xffu) == tb;
      if (EMIT) {
        unsigned m = __ballot_sync(0xffffffffu, hi);
        if (m) {
          int pos = 0;
          int leader = __ffs(m) - 1;
          if (lane == leader) pos = atomicAdd(&S.tcur, __popc(m));
          pos = __shfl_sync(0xffffffffu, pos, leader);
          if (hi) orow[emitted + pos + __popc(m & ((1u << lane) - 1))] = idx;
        }
      }
      if (!spill && lvl < 3) {
        unsigned mb = __ballot_sync(0xffffffffu, bd);
        if (mb) {
          int pos = 0;
          int leader = __ffs(mb) - 1;
          if (lane == leader) pos = atomicAdd(&scur, __popc(mb));
          pos = __shfl_sync(0xffffffffu, pos, leader);
          if (bd) {
            int p = pos + __popc(mb & ((1u << lane) - 1));
            if (p < SURV) { nv[p] = x; ni[p] = idx; }
          }
        }
      }
    }
    __syncthreads();
    emitted += above;
    need -= above;
    if (need <= 0) return;   // boundary fell between buckets (EMIT emitted all)
    prefix = (lvl == 0 ? tb : (prefix << 8) | tb);
    if (lvl == 3) break;
    // adopt survivors if they fit; else continue with full scans
    if (!spill) {
      if (scur <= SURV) {
        cur = nxt;
        n_cur = scur;
      } else {
        spill = true;
        cur = -1;
      }
    }
    __syncthreads();
  }
  // lvl == 3 exit: prefix = exact key of the kth element
  if (EMIT) {
    if (threadIdx.x == 0) S.tcur = 0;
    __syncthreads();
    const bool from_buf = (cur >= 0) && !spill;
    const float* bv = sbufv + (cur < 0 ? 0 : cur) * SURV;
    const int* bi = sbufi + (cur < 0 ? 0 : cur) * SURV;
    int nn = from_buf ? n_cur : n;
    int npadded2 = (nn + blockDim.x - 1) / blockDim.x * blockDim.x;
    for (int j = threadIdx.x; j < npadded2; j += blockDim.x) {
      bool hit = false;
      int idx = 0;
      if (j < nn) {
        float x = from_buf ? bv[j] : (FROM_ROW ? lg[j] : src_val[j]);
        hit = (mono_key(x) == prefix);
        idx = from_buf ? bi[j] : (FROM_ROW ? j : src_idx[j]);
      }
      unsigned m = __ballot_sync(0xffffffffu, hit);
      if (m) {
        int pos = 0;
        int leader = __ffs(m) - 1;
        if (lane == leader) pos = atomicAdd(&S.tcur, __popc(m));
        pos = __shfl_sync(0xffffffffu, pos, leader);
        int p = pos + __popc(m & ((1u << lane) - 1));
        if (hit && p < need) orow[emitted + p] = idx;
      }
    }
    __syncthreads();
  } else {
    if (threadIdx.x == 0 && kth_key_out) *kth_key_out = prefix;
  }
}


extern "C" __global__ void __launch_bounds__(512, 3)
arm_small_kernel(const float* __restrict__ logits, const int* __restrict__ pre_idx,
                 int* __restrict__ out, int npad, int K) {
  const int row = blockIdx.x;
  const float* lg = logits + (size_t)row * npad;
  const int* pre = pre_idx + (size_t)row * K;
  // inline min-hint threshold (subsampled <=512 gathers)
  float m = __int_as_float(0x7f800000);
  const int hstep = (K + 511) / 512;
  for (int j = threadIdx.x * hstep; j < K; j += blockDim.x * hstep) {
    int idx = pre[j];
    if (idx >= 0 && idx < npad) m = fminf(m, lg[idx]);
  }
  __shared__ float swr[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o; o >>= 1) m = fminf(m, __shfl_down_sync(0xffffffffu, m, o));
  if (!lane) swr[warp] = m;
  __syncthreads();
  __shared__ float s_thr;
  if (!warp) {
    m = (lane < (int)(blockDim.x >> 5)) ? swr[lane] : __int_as_float(0x7f800000);
    for (int o = 16; o; o >>= 1) m = fminf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (!lane) s_thr = m;
  }
  __syncthreads();
  const float t = s_thr;
  // inline collect to smem
  __shared__ float s_val[STAGE];
  __shared__ int s_idx[STAGE];
  __shared__ int s_cnt;
  if (threadIdx.x == 0) s_cnt = 0;
  __syncthreads();
  const int n4 = npad / 4;
  const float4* lg4 = reinterpret_cast<const float4*>(lg);
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    float4 a = lg4[i];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      float x = (c == 0) ? a.x : (c == 1) ? a.y : (c == 2) ? a.z : a.w;
      if (x >= t) {
        int p = atomicAdd(&s_cnt, 1);
        if (p < STAGE) { s_val[p] = x; s_idx[p] = i * 4 + c; }
      }
    }
  }
  __syncthreads();
  int n = s_cnt;
  __shared__ RedSmem S;
  int* orow = out + (size_t)row * K;
  if (n >= K && n <= STAGE) {
    // survivor buffers carved from the upper half of the stage (collect done)
    exact_topk_from<false, true>(s_val, s_idx, lg, n, K, orow, S, nullptr,
                                 nullptr, nullptr);
  } else {
    exact_topk_from<true, true>(nullptr, nullptr, lg, npad, K, orow, S,
                                nullptr, nullptr, nullptr);
  }
}

template <bool RESCUE>
__global__ void __launch_bounds__(512, 5)
arm_kernel(const float* __restrict__ logits, float* __restrict__ thr,
           float* __restrict__ cand_val, int* __restrict__ cand_idx,
           int* __restrict__ cnt, int* __restrict__ done, int* __restrict__ ovf,
           int* __restrict__ rescue, int* __restrict__ out,
           int npad, int K, int BS) {
  const int row = blockIdx.y;
  if (RESCUE && rescue[row] == 0) return;  // near-free when nothing to rescue
  const int nchunk = gridDim.x;
  const float t = thr[row];
  const float* lg = logits + (size_t)row * npad;
  const int n4 = npad / 4;
  const int per_chunk = (n4 + nchunk - 1) / nchunk;
  const int beg = blockIdx.x * per_chunk;
  const int end = min(beg + per_chunk, n4);
  const float4* lg4 = reinterpret_cast<const float4*>(lg);
  __shared__ float s_val[STAGE];
  __shared__ int s_idx[STAGE];
  __shared__ int s_cnt, s_base, s_arr;
  if (threadIdx.x == 0) s_cnt = 0;
  __syncthreads();
  int i = beg + threadIdx.x;
  const int bd = blockDim.x;
  for (; i + 3 * bd < end; i += 4 * bd) {
    float4 a0 = lg4[i];
    float4 a1 = lg4[i + bd];
    float4 a2 = lg4[i + 2 * bd];
    float4 a3 = lg4[i + 3 * bd];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      float4 v = (q == 0) ? a0 : (q == 1) ? a1 : (q == 2) ? a2 : a3;
      int base = (i + q * bd) * 4;
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        float x = (c == 0) ? v.x : (c == 1) ? v.y : (c == 2) ? v.z : v.w;
        if (x >= t) {
          int p = atomicAdd(&s_cnt, 1);
          if (p < STAGE) { s_val[p] = x; s_idx[p] = base + c; }
        }
      }
    }
  }
  for (; i + (int)blockDim.x < end; i += 2 * blockDim.x) {
    float4 a = lg4[i];
    float4 b = lg4[i + blockDim.x];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      float xa = (c == 0) ? a.x : (c == 1) ? a.y : (c == 2) ? a.z : a.w;
      if (xa >= t) {
        int p = atomicAdd(&s_cnt, 1);
        if (p < STAGE) { s_val[p] = xa; s_idx[p] = i * 4 + c; }
      }
      float xb = (c == 0) ? b.x : (c == 1) ? b.y : (c == 2) ? b.z : b.w;
      if (xb >= t) {
        int p = atomicAdd(&s_cnt, 1);
        if (p < STAGE) { s_val[p] = xb; s_idx[p] = (i + blockDim.x) * 4 + c; }
      }
    }
  }
  for (; i < end; i += blockDim.x) {
    float4 a = lg4[i];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      float xa = (c == 0) ? a.x : (c == 1) ? a.y : (c == 2) ? a.z : a.w;
      if (xa >= t) {
        int p = atomicAdd(&s_cnt, 1);
        if (p < STAGE) { s_val[p] = xa; s_idx[p] = i * 4 + c; }
      }
    }
  }
  __syncthreads();
  int local_n = s_cnt;
  int store_n = min(local_n, STAGE);
  if (threadIdx.x == 0) {
    s_base = atomicAdd(cnt + row, store_n);  // stored entries stay CONTIGUOUS
    atomicAdd(ovf + row, local_n);           // ovf doubles as the TRUE count
  }
  __syncthreads();
  int gb = s_base;
  float* cv = cand_val + (size_t)row * CAP;
  int* ci = cand_idx + (size_t)row * CAP;
  for (int j = threadIdx.x; j < store_n; j += blockDim.x) {
    int off = gb + j;
    if (off < CAP) { cv[off] = s_val[j]; ci[off] = s_idx[j]; }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_arr = atomicAdd(done + row, 1);
  __syncthreads();
  if (s_arr != nchunk - 1) return;
  __threadfence();
  // ---- last CTA of the row ----
  int true_n = ovf[row];
  int n_stored = min(cnt[row], CAP);
  bool overflow = (true_n > n_stored);
#ifdef DBG_TRACE
  if (RESCUE && threadIdx.x == 0)
    done[row + BS] = (overflow ? 1000000000 : 0) + true_n;  // caller sizes done as 2*BS
#endif
  __shared__ RedSmem S;
  __shared__ unsigned kth_key;
  int* orow = out + (size_t)row * K;
  if (!overflow && n_stored >= K) {
    exact_topk_from<false, true>(cv, ci, lg, n_stored, K, orow, S, nullptr, s_val, s_idx);
    if (RESCUE) {
      __syncthreads();
      if (threadIdx.x == 0) rescue[row] = 0;
    }
  } else if (!RESCUE && n_stored >= K) {
    // second chance: t2 = r-th value of the stored subset, r scaled so the
    // rescan count lands near (5/8)*CAP in expectation: row_rank(t2) ~=
    // r * true_n / n_stored. r >= K*stored/true_n ensures count >= K
    // (5*CAP/8 = 5120 > K for all K <= 2048); 8K-sample noise is ~70 << margin.
    long r = (long)(5 * CAP / 8) * n_stored / max(true_n, 1);
    int rk = (int)min((long)n_stored, max((long)1, r));
    exact_topk_from<false, false>(cv, ci, lg, n_stored, rk, orow, S, &kth_key,
                                  s_val, s_idx);
    __syncthreads();
    if (threadIdx.x == 0) {
      thr[row] = inv_mono(kth_key);
      rescue[row] = 1;
    }
  } else if (!RESCUE && true_n < K) {
    // undershoot: rescue at the deep fallback quantile (cheap re-collect),
    // not the full-row single-CTA path
    if (threadIdx.x == 0) {
      thr[row] = thr[BS + row];
      rescue[row] = 1;
    }
  } else {
    // final resort (rescue-level overflow/undershoot or degenerate hints)
    exact_topk_from<true, true>(nullptr, nullptr, lg, npad, K, orow, S, nullptr, s_val, s_idx);
    if (RESCUE) {
      __syncthreads();
      if (threadIdx.x == 0) rescue[row] = 0;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) { cnt[row] = 0; done[row] = 0; ovf[row] = 0; }
}

extern "C" void arm_v2_launch(const float* logits, const int* pre_idx, float* thr,
                              float* cand_val, int* cand_idx, int* cnt, int* done,
                              int* ovf, int* rescue, int* out, int npad, int K,
                              int BS, int chunks, cudaStream_t stream) {
  if (npad <= 8192) {  // small rows: single launch, inline hint-thresholded
    arm_small_kernel<<<BS, 512, 0, stream>>>(logits, pre_idx, out, npad, K);
    return;
  }
  thresh_kernel<<<BS, 512, 0, stream>>>(logits, pre_idx, thr, npad, K);
  arm_kernel<false><<<dim3(chunks, BS), 512, 0, stream>>>(
      logits, thr, cand_val, cand_idx, cnt, done, ovf, rescue, out, npad, K, BS);
  arm_kernel<true><<<dim3(chunks, BS), 512, 0, stream>>>(
      logits, thr, cand_val, cand_idx, cnt, done, ovf, rescue, out, npad, K, BS);
}
