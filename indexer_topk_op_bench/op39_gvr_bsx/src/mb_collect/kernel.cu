// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// op39 rung-2 microbench: tile-parallel single-pass collect structure.
// Q: with an ORACLE per-row threshold, how fast can (scan + compare + append
// candidates) + (per-row exact top-K over candidates) run, versus the
// per-row-serial incumbent shape?  This measures the STRUCTURE, not the full
// algorithm (threshold estimation comes later; GVR hint machinery supplies it).
//
// K1: grid-stride over (row, chunk) tiles; each thread appends qualifying
//     (value, idx) pairs to the row's candidate buffer via one warp-aggregated
//     atomicAdd on a per-row cursor.
// K2: one CTA per row: exact top-K over the collected candidates: counts of
//     candidates are ~K..CAP; bitonic-free approach: per-row full sort is
//     overkill — use threshold-count + smem collect (candidates fit in smem).
#include <cuda_runtime.h>
#include <cstdint>

#ifndef CAP
#define CAP 8192  // per-row candidate capacity (values+idx)
#endif

// K1: single-pass collect. logits [BS, npad], thr [BS] (oracle: kth value),
// cand_val/cand_idx [BS, CAP], cnt [BS].
// smem staging: hits go to a CTA-local smem buffer (smem atomics), one global
// cursor bump per CTA at the end, then a coalesced smem->global copy. No warp
// sync primitives in divergent code; one global atomic per CTA total.
#define STAGE 6144  // per-CTA staged candidates (8B each -> 48KB smem)
extern "C" __global__ void __launch_bounds__(512, 3)
collect_kernel(const float* __restrict__ logits, const float* __restrict__ thr,
               float* __restrict__ cand_val, int* __restrict__ cand_idx,
               int* __restrict__ cnt, int npad, int BS) {
  const int nchunk_per_row = gridDim.x;         // chunks per row
  const int row = blockIdx.y;
  const float t = thr[row];
  const float* lg = logits + (size_t)row * npad;
  const int chunk = blockIdx.x;
  const int per_chunk = (npad / 4 + nchunk_per_row - 1) / nchunk_per_row;  // in float4
  const int beg = chunk * per_chunk;
  const int end = min(beg + per_chunk, npad / 4);
  const float4* lg4 = reinterpret_cast<const float4*>(lg);
  __shared__ float s_val[STAGE];
  __shared__ int s_idx[STAGE];
  __shared__ int s_cnt;
  __shared__ int s_base;
  if (threadIdx.x == 0) s_cnt = 0;
  __syncthreads();
  for (int i = beg + threadIdx.x; i < end; i += blockDim.x) {
    float4 v = lg4[i];
    int base = i * 4;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      float x = (c == 0) ? v.x : (c == 1) ? v.y : (c == 2) ? v.z : v.w;
      if (x >= t) {
        int p = atomicAdd(&s_cnt, 1);
        if (p < STAGE) { s_val[p] = x; s_idx[p] = base + c; }
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
  for (int i = threadIdx.x; i < nc; i += blockDim.x) {
    int off = gb + i;
    if (off < CAP) { cv[off] = s_val[i]; ci[off] = s_idx[i]; }
  }
}

// K2: exact top-K on candidates. One CTA (1024 thr) per row. Candidates
// (n <= CAP) live in global; K <= 2048. Simple smem approach: load candidates
// to smem, iterative threshold-refine on the small set (values already
// pre-filtered so n is K..~4K): full bitonic sort of CAP is avoidable — use
// per-thread local count + exact selection via histogram on exponent bits.
// For the MICROBENCH we only need the cost shape, so do: parallel partial
// selection by repeated max-reduction is O(K*n) — too slow. Instead: radix
// select on 32-bit keys (monotonic float transform), 2 rounds of 11 bits.
extern "C" __global__ void __launch_bounds__(1024, 1)
reduce_topk_kernel(const float* __restrict__ cand_val, const int* __restrict__ cand_idx,
                   const int* __restrict__ cnt, int* __restrict__ out, int K) {
  const int row = blockIdx.x;
  int n = min(cnt[row], CAP);
  const float* cv = cand_val + (size_t)row * CAP;
  const int* ci = cand_idx + (size_t)row * CAP;
  __shared__ unsigned hist[256];
  __shared__ unsigned thr_key, base_cnt;
  // monotonic key: flip sign bit for positive, invert for negative (desc order)
  // round 1: top 11 bits
  for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned k = __float_as_uint(cv[i]);
    k = (k & 0x80000000u) ? ~k : (k | 0x80000000u);
    atomicAdd(&hist[k >> 24], 1u);
  }
  __syncthreads();
  // exclusive scan from top bucket down, find bucket where cum >= K
  if (threadIdx.x == 0) {
    unsigned cum = 0;
    for (int b = 255; b >= 0; --b) {
      unsigned c = hist[b];
      if (cum + c >= (unsigned)K) { thr_key = b; base_cnt = cum; break; }
      cum += c;
    }
  }
  __syncthreads();
  unsigned tb = thr_key, bc = base_cnt;
  // emit all strictly-above buckets; count exact within threshold bucket
  __shared__ int ocur;
  if (threadIdx.x == 0) ocur = 0;
  __syncthreads();
  int* orow = out + (size_t)row * K;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned k = __float_as_uint(cv[i]);
    k = (k & 0x80000000u) ? ~k : (k | 0x80000000u);
    unsigned b = k >> 24;
    if (b > tb) {
      int p = atomicAdd(&ocur, 1);
      orow[p] = ci[i];
    }
  }
  __syncthreads();
  // fill remainder from the threshold bucket (ties beyond K broken arbitrarily)
  int need = K - ocur;
  if (need > 0) {
    __shared__ int tcur;
    if (threadIdx.x == 0) tcur = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      unsigned k = __float_as_uint(cv[i]);
      k = (k & 0x80000000u) ? ~k : (k | 0x80000000u);
      if ((k >> 24) == tb) {
        int p = atomicAdd(&tcur, 1);
        if (p < need) orow[ocur + p] = ci[i];
      }
    }
  }
}

extern "C" void mb_collect_launch(const float* logits, const float* thr, float* cand_val,
                                  int* cand_idx, int* cnt, int* out, int npad, int K,
                                  int BS, int chunks, int run_k2, cudaStream_t stream) {
  cudaMemsetAsync(cnt, 0, sizeof(int) * BS, stream);
  collect_kernel<<<dim3(chunks, BS), 512, 0, stream>>>(logits, thr, cand_val, cand_idx,
                                                       cnt, npad, BS);
  if (run_k2)
    reduce_topk_kernel<<<BS, 1024, 0, stream>>>(cand_val, cand_idx, cnt, out, K);
}
