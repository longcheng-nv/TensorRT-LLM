// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// op39 iter2: FUSED tile-parallel 1-pass collect + last-CTA-per-row reduce.
// - grid (chunks, BS) x 512 thr; smem-staged collect (iter1-validated);
// - after writeout, each CTA arrives on done[row]; the LAST CTA of the row
//   runs the candidate top-K reduce in-place (release/acquire threadfences —
//   cf. cluster_arrive_relaxed DSMEM staleness lesson);
// - reducer self-cleans cnt[row]/done[row] so no memset launch is needed
//   (buffers must be zeroed once at allocation);
// - scan loop 2x float4 ILP; reduce emit warp-aggregated (uniform loop bound,
//   OOB lanes predicated false inside the full-mask ballot).
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

extern "C" __global__ void __launch_bounds__(512, 5)
fused_kernel(const float* __restrict__ logits, const float* __restrict__ thr,
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
  // 2x float4 ILP scan
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
  int nc = min(s_cnt, STAGE);
  if (threadIdx.x == 0) s_base = atomicAdd(cnt + row, nc);
  __syncthreads();
  int gb = s_base;
  float* cv = cand_val + (size_t)row * CAP;
  int* ci = cand_idx + (size_t)row * CAP;
  for (int j = threadIdx.x; j < nc; j += blockDim.x) {
    int off = gb + j;
    if (off < CAP) { cv[off] = s_val[j]; ci[off] = s_idx[j]; }
  }
  // arrival (release) — last CTA of the row reduces
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_arr = atomicAdd(done + row, 1);
  __syncthreads();
  if (s_arr != nchunk - 1) return;
  __threadfence();  // acquire side
  // ---- reduce: exact-ish top-K over candidates (bucket select on top 8 bits)
  int n = min(cnt[row], CAP);
  __shared__ unsigned hist[256];
  __shared__ unsigned wsum[16];
  __shared__ unsigned thr_bucket;
  __shared__ int ocur;
  for (int j = threadIdx.x; j < 256; j += blockDim.x) hist[j] = 0;
  if (threadIdx.x == 0) ocur = 0;
  __syncthreads();
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    atomicAdd(&hist[mono_key(cv[j]) >> 24], 1u);
  __syncthreads();
  // two-level suffix search: warp w sums buckets [w*32, w*32+32)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp < 8) {
    unsigned v = hist[warp * 32 + lane];
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (!lane) wsum[warp] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned cum = 0;
    int g = 7;
    for (; g >= 0; --g) {
      if (cum + wsum[g] >= (unsigned)K) break;
      cum += wsum[g];
    }
    if (g < 0) g = 0;
    // within group g, from top bucket down
    unsigned tb = g * 32;
    for (int b = g * 32 + 31; b >= g * 32; --b) {
      unsigned c = hist[b];
      if (cum + c >= (unsigned)K) { tb = b; break; }
      cum += c;
    }
    thr_bucket = tb;
  }
  __syncthreads();
  unsigned tb = thr_bucket;
  int* orow = out + (size_t)row * K;
  // emit above-bucket, warp-aggregated (uniform padded loop)
  int npadded = (n + blockDim.x - 1) / blockDim.x * blockDim.x;
  for (int j = threadIdx.x; j < npadded; j += blockDim.x) {
    bool hit = (j < n) && (mono_key(cv[j]) >> 24) > tb;
    unsigned m = __ballot_sync(0xffffffffu, hit);
    if (m) {
      int pos = 0;
      if (lane == __ffs(m) - 1) pos = atomicAdd(&ocur, __popc(m));
      pos = __shfl_sync(0xffffffffu, pos, __ffs(m) - 1);
      if (hit) orow[pos + __popc(m & ((1u << lane) - 1))] = ci[j];
    }
  }
  __syncthreads();
  int need = K - ocur;
  if (need > 0) {  // fill from threshold bucket (tie order arbitrary)
    __shared__ int tcur;
    if (threadIdx.x == 0) tcur = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < npadded; j += blockDim.x) {
      bool hit = (j < n) && (mono_key(cv[j]) >> 24) == tb;
      unsigned m = __ballot_sync(0xffffffffu, hit);
      if (m) {
        int pos = 0;
        if (lane == __ffs(m) - 1) pos = atomicAdd(&tcur, __popc(m));
        pos = __shfl_sync(0xffffffffu, pos, __ffs(m) - 1);
        int p = pos + __popc(m & ((1u << lane) - 1));
        if (hit && p < need) orow[ocur + p] = ci[j];
      }
    }
  }
  // self-clean for the next launch
  __syncthreads();
  if (threadIdx.x == 0) { cnt[row] = 0; done[row] = 0; }
}

extern "C" void mb_fused_launch(const float* logits, const float* thr, float* cand_val,
                                int* cand_idx, int* cnt, int* done, int* out, int npad,
                                int K, int BS, int chunks, cudaStream_t stream) {
  fused_kernel<<<dim3(chunks, BS), 512, 0, stream>>>(logits, thr, cand_val, cand_idx,
                                                     cnt, done, out, npad, K, BS);
}
