// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// op39 iter4 production arm v1: hint-thresholded fused 1-pass collect top-K.
//
// K0 thresh_kernel: t_lo[row] = min over hint values lg[pre[row][0..K)].
//    With hit-rate < 1 at least one hint lies below the true kth value, so
//    count(x >= t_lo) >= K (no undershoot); h == 1 gives count == K exactly.
// K1 fused_kernel: tile-parallel scan (grid (chunks, BS)); candidates >= t_lo
//    staged in smem, flushed once per CTA; last CTA of the row reduces:
//    exact top-K via up-to-4-level 8-bit bucket refinement (tie-exact in the
//    value-multiset sense), warp-aggregated emit.
//    Fallback (rare): candidate overflow (cnt > CAP, deep-hint rows at low
//    hit rate) -> the reducer CTA re-scans the WHOLE row with the same bucket
//    machinery (slow for that row only, still exact).
#include <cuda_runtime.h>
#include <cstdint>

#ifndef CAP
#define CAP 8192
#endif
#define STAGE 6144

__device__ __forceinline__ unsigned mono_key(float x) {
  unsigned k = __float_as_uint(x);
  return (k & 0x80000000u) ? ~k : (k | 0x80000000u);
}

// ---------------- K0: per-row threshold from hints -------------------------
extern "C" __global__ void thresh_kernel(const float* __restrict__ logits,
                                         const int* __restrict__ pre_idx,
                                         float* __restrict__ thr, int npad, int K) {
  const int row = blockIdx.x;
  const float* lg = logits + (size_t)row * npad;
  const int* pre = pre_idx + (size_t)row * K;
  float m = __int_as_float(0x7f800000);  // +inf
  for (int j = threadIdx.x; j < K; j += blockDim.x) {
    int idx = pre[j];
    if (idx >= 0 && idx < npad) m = fminf(m, lg[idx]);
  }
  __shared__ float s[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o; o >>= 1) m = fminf(m, __shfl_down_sync(0xffffffffu, m, o));
  if (!lane) s[warp] = m;
  __syncthreads();
  if (!warp) {
    m = (lane < (int)(blockDim.x >> 5)) ? s[lane] : __int_as_float(0x7f800000);
    for (int o = 16; o; o >>= 1) m = fminf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (!lane) thr[row] = m;
  }
}

// ---------------- reduce helpers (run by one CTA) ---------------------------
// Exact top-K from a candidate list (global mem) via multi-level 8-bit bucket
// refinement on the monotonic key. Returns nothing; writes out[row*K..].
// src_val/src_idx may be the staged candidates OR (fallback) recomputed from
// the full row. All threads of the CTA participate.
struct RedSmem {
  unsigned hist[256];
  unsigned wsum[16];
  unsigned thr_bucket, prefix_lo;  // key prefix of the threshold path
  int ocur, tcur, level_n;
};

// count/emit pass over candidates with a key filter: elements whose key
// matches `prefix` on the top `lvl*8` bits participate at this level.
template <bool FROM_ROW>
__device__ void exact_topk_from(const float* __restrict__ src_val,
                                const int* __restrict__ src_idx,
                                const float* __restrict__ lg, int n, int K,
                                int* __restrict__ orow, RedSmem& S) {
  const int lane = threadIdx.x & 31;
  // level loop: at each level we know (prefix, remaining K') and restrict to
  // elements with key top-bits == prefix of the threshold bucket chain.
  unsigned prefix = 0;      // matched high bits so far (value)
  int need = K;             // how many still to take from the prefix subtree
  int emitted = 0;
  for (int lvl = 0; lvl < 4; ++lvl) {
    const int shift = 24 - lvl * 8;
    // histogram of this level among elements matching prefix
    for (int j = threadIdx.x; j < 256; j += blockDim.x) S.hist[j] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      float x = FROM_ROW ? lg[j] : src_val[j];
      unsigned k = mono_key(x);
      if (lvl == 0 || (k >> (shift + 8)) == prefix)
        atomicAdd(&S.hist[(k >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    // suffix search for the bucket where cumulative reaches `need`
    const int warp = threadIdx.x >> 5;
    if (warp < 8) {
      unsigned v = S.hist[warp * 32 + lane];
      for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
      if (!lane) S.wsum[warp] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned cum = 0;
      int g = 7;
      for (; g > 0; --g) {
        if (cum + S.wsum[g] >= (unsigned)need) break;
        cum += S.wsum[g];
      }
      unsigned tb = g * 32;
      for (int b = g * 32 + 31; b >= g * 32; --b) {
        unsigned c = S.hist[b];
        if (cum + c >= (unsigned)need) { tb = b; break; }
        cum += c;
      }
      S.thr_bucket = tb;
      S.prefix_lo = cum;  // taken from buckets above tb at this level
      S.ocur = 0;
    }
    __syncthreads();
    unsigned tb = S.thr_bucket;
    // emit everything strictly above the threshold bucket at this level
    int npadded = (n + blockDim.x - 1) / blockDim.x * blockDim.x;
    for (int j = threadIdx.x; j < npadded; j += blockDim.x) {
      bool hit = false;
      int idx = 0;
      if (j < n) {
        float x = FROM_ROW ? lg[j] : src_val[j];
        unsigned k = mono_key(x);
        if (lvl == 0 || (k >> (shift + 8)) == prefix) {
          unsigned b = (k >> shift) & 0xffu;
          hit = b > tb;
        }
        idx = FROM_ROW ? j : src_idx[j];
      }
      unsigned m = __ballot_sync(0xffffffffu, hit);
      if (m) {
        int pos = 0;
        int leader = __ffs(m) - 1;
        if (lane == leader) pos = atomicAdd(&S.ocur, __popc(m));
        pos = __shfl_sync(0xffffffffu, pos, leader);
        if (hit) orow[emitted + pos + __popc(m & ((1u << lane) - 1))] = idx;
      }
    }
    __syncthreads();
    emitted += S.ocur;
    need -= S.ocur;
    __syncthreads();
    if (need <= 0) return;  // exact boundary fell between buckets
    prefix = (lvl == 0 ? 0 : prefix << 8) | tb;
    if (lvl == 3) {
      // full 32-bit prefix: remaining `need` slots are exact value ties —
      // any `need` of them are multiset-correct.
      if (threadIdx.x == 0) S.tcur = 0;
      __syncthreads();
      int npadded2 = (n + blockDim.x - 1) / blockDim.x * blockDim.x;
      for (int j = threadIdx.x; j < npadded2; j += blockDim.x) {
        bool hit = false;
        int idx = 0;
        if (j < n) {
          float x = FROM_ROW ? lg[j] : src_val[j];
          hit = (mono_key(x) == prefix);
          idx = FROM_ROW ? j : src_idx[j];
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
      return;
    }
  }
}

// ---------------- K1: fused collect + last-CTA exact reduce ----------------
extern "C" __global__ void __launch_bounds__(512, 3)
arm_kernel(const float* __restrict__ logits, const float* __restrict__ thr,
           float* __restrict__ cand_val, int* __restrict__ cand_idx,
           int* __restrict__ cnt, int* __restrict__ done, int* __restrict__ out,
           int npad, int K, int BS) {
  const int nchunk = gridDim.x;
  const int row = blockIdx.y;
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
  int local_n = s_cnt;                    // true count in this chunk
  int store_n = min(local_n, STAGE);      // what we actually staged
  // a clipped chunk force-trips the reducer's overflow predicate (cnt > CAP)
  if (threadIdx.x == 0)
    s_base = atomicAdd(cnt + row, local_n > STAGE ? local_n + CAP + 1 : local_n);
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
  // ---- last CTA reduces ----
  int n = cnt[row];
  __shared__ RedSmem S;
  int* orow = out + (size_t)row * K;
  bool overflow = (n > CAP) || (n < K);
  // n > CAP covers cross-chunk overflow AND clipped chunks (they add CAP+1);
  // n < K is degenerate-hint insurance (e.g. all hints invalid -> t = +inf)
  if (!overflow) {
    exact_topk_from<false>(cv, ci, lg, n, K, orow, S);
  } else {
    // fallback: exact top-K over the whole row (reads npad scalars; pad
    // values are -inf-like lows and cannot enter the top-K)
    exact_topk_from<true>(nullptr, nullptr, lg, npad, K, orow, S);
  }
  __syncthreads();
  if (threadIdx.x == 0) { cnt[row] = 0; done[row] = 0; }
}

extern "C" void arm_v1_launch(const float* logits, const int* pre_idx, float* thr,
                              float* cand_val, int* cand_idx, int* cnt, int* done,
                              int* out, int npad, int K, int BS, int chunks,
                              cudaStream_t stream) {
  thresh_kernel<<<BS, 256, 0, stream>>>(logits, pre_idx, thr, npad, K);
  arm_kernel<<<dim3(chunks, BS), 512, 0, stream>>>(logits, thr, cand_val, cand_idx,
                                                   cnt, done, out, npad, K, BS);
}
