// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// op35 rung-0/rung-2 probes:
//  k_empty          — nsys-visible launch floor
//  k_stream_reduce  — 1-pass streaming read (float4) + per-row atomic last-CTA
//                     finalize (H0's coordination primitive), self-cleaning counters
//  k_filter_append  — 1-pass 2-threshold filter with warp-aggregated append
//                     (cost must scale with admitted count, not N) + rung counts
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TPB 512

__global__ void k_empty() {}

// grid.x = rows * ctas_per_row; CTA r of row handles slice [r*chunk, ...)
__global__ void k_stream_reduce(const float4* __restrict__ x, float* __restrict__ out,
                                int* __restrict__ tickets, long n4_per_row,
                                int ctas_per_row) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const float4* xr = x + (long)row * n4_per_row;
  float m = -1e30f;
  const long begin = (long)sub * ((n4_per_row + ctas_per_row - 1) / ctas_per_row);
  const long end = min(n4_per_row, begin + ((n4_per_row + ctas_per_row - 1) / ctas_per_row));
  for (long i = begin + threadIdx.x; i < end; i += TPB) {
    float4 v = xr[i];
    m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  __shared__ float sm[TPB / 32];
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(~0u, m, o));
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < TPB / 32) {
    m = sm[threadIdx.x];
    for (int o = TPB / 64; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(~0u, m, o));
    if (threadIdx.x == 0) {
      // per-CTA partial -> atomicMax on out[row] (float via int trick, values > 0 assumed shifted)
      atomicMax((int*)&out[row], __float_as_int(m));
      __threadfence();
      int t = atomicAdd(&tickets[row], 1);
      if (t == ctas_per_row - 1) tickets[row] = 0;  // self-clean, graph-replayable
    }
  }
}

// 2-rung filter: admit x >= t_lo into cand buffer (warp-aggregated), count x >= t_hi.
// scratch layout per row: cand values [cap], cand idx [cap]; counts[row*2 +{0,1}]
__global__ void k_filter_append(const float4* __restrict__ x, float t_hi, float t_lo,
                                float* __restrict__ cand_v, int* __restrict__ cand_i,
                                int* __restrict__ counts, int* __restrict__ tickets,
                                long n4_per_row, int ctas_per_row, int cap) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const float4* xr = x + (long)row * n4_per_row;
  float* cv = cand_v + (long)row * cap;
  int* ci = cand_i + (long)row * cap;
  int c_hi = 0;
  const long chunk = (n4_per_row + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4_per_row, begin + chunk);
  // per-lane register buffer: amortize append coordination over LBUF admits
  const int LBUF = 8;
  float bv[LBUF];
  int bi[LBUF];
  int nb = 0;
  const long iters = (end - begin + TPB - 1) / TPB;  // uniform across warp
  for (long it = 0; it < iters; ++it) {
    const long i = begin + it * TPB + threadIdx.x;
    const bool valid = i < end;
    float4 v = valid ? xr[i] : make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
    float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float f = vv[j];
      c_hi += (f >= t_hi);
      if (f >= t_lo) { bv[nb] = f; bi[nb] = (int)(i * 4 + j); ++nb; }
    }
    // uniform-flow flush when any lane's buffer is nearly full
    if (__any_sync(~0u, nb >= LBUF - 4)) {
      int total;
      int pre = nb;
      // warp-exclusive prefix sum of nb
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(~0u, pre, o);
        if ((int)(threadIdx.x & 31) >= o) pre += y;
      }
      total = __shfl_sync(~0u, pre, 31);
      pre -= nb;
      int base_slot = 0;
      if ((threadIdx.x & 31) == 31) base_slot = atomicAdd(&counts[row * 2 + 1], total);
      base_slot = __shfl_sync(~0u, base_slot, 31);
      for (int u = 0; u < nb; ++u) {
        int slot = base_slot + pre + u;
        if (slot < cap) { cv[slot] = bv[u]; ci[slot] = bi[u]; }
      }
      nb = 0;
    }
  }
  {  // final flush (uniform)
    int pre = nb;
    for (int o = 1; o < 32; o <<= 1) {
      int y = __shfl_up_sync(~0u, pre, o);
      if ((int)(threadIdx.x & 31) >= o) pre += y;
    }
    int total = __shfl_sync(~0u, pre, 31);
    pre -= nb;
    int base_slot = 0;
    if (total > 0 && (threadIdx.x & 31) == 31) base_slot = atomicAdd(&counts[row * 2 + 1], total);
    base_slot = __shfl_sync(~0u, base_slot, 31);
    for (int u = 0; u < nb; ++u) {
      int slot = base_slot + pre + u;
      if (slot < cap) { cv[slot] = bv[u]; ci[slot] = bi[u]; }
    }
  }
  // reduce c_hi across CTA -> atomic
  __shared__ int sh[TPB / 32];
  for (int o = 16; o; o >>= 1) c_hi += __shfl_down_sync(~0u, c_hi, o);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = c_hi;
  __syncthreads();
  if (threadIdx.x < TPB / 32) {
    c_hi = sh[threadIdx.x];
    for (int o = TPB / 64; o; o >>= 1) c_hi += __shfl_down_sync(~0u, c_hi, o);
    if (threadIdx.x == 0) {
      atomicAdd(&counts[row * 2], c_hi);
      __threadfence();
      int t = atomicAdd(&tickets[row], 1);
      if (t == ctas_per_row - 1) tickets[row] = 0;
    }
  }
}


// ---- iter1: filter v2 — SMEM-atomic staging + coalesced bulk flush ---------
// Admits (>= t_lo) stage into SMEM via smem atomicAdd (cheap, no warp coord);
// CTA flushes SMEM->global once (or on overflow). c_hi counted per thread.
#define SCAP 2048  // staged pairs per CTA (16KB)
__global__ void k_filter_v2(const float4* __restrict__ x, float t_hi, float t_lo,
                            float* __restrict__ cand_v, int* __restrict__ cand_i,
                            int* __restrict__ counts, int* __restrict__ tickets,
                            long n4_per_row, int ctas_per_row, int cap) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const float4* xr = x + (long)row * n4_per_row;
  float* cv = cand_v + (long)row * cap;
  int* ci = cand_i + (long)row * cap;
  __shared__ float sv[SCAP];
  __shared__ int si[SCAP];
  __shared__ int scnt;
  __shared__ int gbase;
  if (threadIdx.x == 0) scnt = 0;
  __syncthreads();
  int c_hi = 0;
  const long chunk = (n4_per_row + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4_per_row, begin + chunk);
  for (long i = begin + threadIdx.x; i < end; i += TPB) {
    float4 v = xr[i];
    float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float f = vv[j];
      c_hi += (f >= t_hi);
      if (f >= t_lo) {
        int s = atomicAdd(&scnt, 1);
        if (s < SCAP) { sv[s] = f; si[s] = (int)(i * 4 + j); }
        else {  // overflow: direct global append (rare; band math keeps SCAP ample)
          int g = atomicAdd(&counts[row * 2 + 1], 1);
          if (g < cap) { cv[g] = f; ci[g] = (int)(i * 4 + j); }
        }
      }
    }
  }
  __syncthreads();
  const int nstage = min(scnt, SCAP);
  if (threadIdx.x == 0) gbase = atomicAdd(&counts[row * 2 + 1], nstage);
  __syncthreads();
  for (int s = threadIdx.x; s < nstage; s += TPB) {
    int g = gbase + s;
    if (g < cap) { cv[g] = sv[s]; ci[g] = si[s]; }
  }
  // c_hi reduce + ticket
  __shared__ int sh[TPB / 32];
  for (int o = 16; o; o >>= 1) c_hi += __shfl_down_sync(~0u, c_hi, o);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = c_hi;
  __syncthreads();
  if (threadIdx.x < TPB / 32) {
    c_hi = sh[threadIdx.x];
    for (int o = TPB / 64; o; o >>= 1) c_hi += __shfl_down_sync(~0u, c_hi, o);
    if (threadIdx.x == 0) {
      atomicAdd(&counts[row * 2], c_hi);
      __threadfence();
      int t = atomicAdd(&tickets[row], 1);
      if (t == ctas_per_row - 1) tickets[row] = 0;
    }
  }
}

// iter1b: pure-read with 2x float4 ILP per iteration (mid-BS wave inefficiency probe)
__global__ void k_stream_reduce2(const float4* __restrict__ x, float* __restrict__ out,
                                 int* __restrict__ tickets, long n4_per_row,
                                 int ctas_per_row) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const float4* xr = x + (long)row * n4_per_row;
  float m = -1e30f;
  const long chunk = (n4_per_row + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4_per_row, begin + chunk);
  long i = begin + threadIdx.x;
  for (; i + TPB < end; i += 2 * TPB) {
    float4 a = xr[i];
    float4 b = xr[i + TPB];
    m = fmaxf(m, fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));
    m = fmaxf(m, fmaxf(fmaxf(b.x, b.y), fmaxf(b.z, b.w)));
  }
  if (i < end) {
    float4 a = xr[i];
    m = fmaxf(m, fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));
  }
  __shared__ float sm[TPB / 32];
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(~0u, m, o));
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < TPB / 32) {
    m = sm[threadIdx.x];
    for (int o = TPB / 64; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(~0u, m, o));
    if (threadIdx.x == 0) {
      atomicMax((int*)&out[row], __float_as_int(m));
      __threadfence();
      int t = atomicAdd(&tickets[row], 1);
      if (t == ctas_per_row - 1) tickets[row] = 0;
    }
  }
}

void empty_launch(int grid) { k_empty<<<grid, TPB>>>(); }

void stream_reduce(torch::Tensor x, torch::Tensor out, torch::Tensor tickets,
                   int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  k_stream_reduce<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), out.data_ptr<float>(),
      tickets.data_ptr<int>(), n4, ctas_per_row);
}

void filter_append(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
                   torch::Tensor cand_i, torch::Tensor counts, torch::Tensor tickets,
                   int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  k_filter_append<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
      cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
      tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

void filter_v2(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
               torch::Tensor cand_i, torch::Tensor counts, torch::Tensor tickets,
               int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  k_filter_v2<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
      cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
      tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

void stream_reduce2(torch::Tensor x, torch::Tensor out, torch::Tensor tickets,
                    int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  k_stream_reduce2<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), out.data_ptr<float>(),
      tickets.data_ptr<int>(), n4, ctas_per_row);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("empty_launch", &empty_launch);
  m.def("stream_reduce", &stream_reduce);
  m.def("filter_append", &filter_append);
  m.def("filter_v2", &filter_v2);
  m.def("stream_reduce2", &stream_reduce2);
}
