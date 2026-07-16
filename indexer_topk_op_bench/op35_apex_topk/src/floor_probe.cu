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


// ---- iter2: filter v3 — SMEM staging + PERIODIC BULK FLUSH (no overflow path)
__global__ void k_filter_v3(const float4* __restrict__ x, float t_hi, float t_lo,
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
  __shared__ int scnt, gbase, sflush;
  if (threadIdx.x == 0) { scnt = 0; sflush = 0; }
  __syncthreads();
  int c_hi = 0;
  const long chunk = (n4_per_row + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4_per_row, begin + chunk);
  const long iters = (end > begin) ? (end - begin + TPB - 1) / TPB : 0;
  // flush cadence: SCAP-TPB*4 guarantees no overflow between checks
  const long FLUSH_EVERY = max(1L, (long)((SCAP - TPB * 4) / (TPB * 4)));
  long since = 0;
  for (long it = 0; it < iters; ++it) {
    const long i = begin + it * TPB + threadIdx.x;
    if (i < end) {
      float4 v = xr[i];
      float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float f = vv[j];
        c_hi += (f >= t_hi);
        if (f >= t_lo) {
          int s = atomicAdd(&scnt, 1);
          sv[s] = f; si[s] = (int)(i * 4 + j);
        }
      }
    }
    if (++since >= FLUSH_EVERY) {
      since = 0;
      __syncthreads();
      const int n = scnt;
      if (n) {
        if (threadIdx.x == 0) { gbase = atomicAdd(&counts[row * 2 + 1], n); scnt = 0; }
        __syncthreads();
        for (int s2 = threadIdx.x; s2 < n; s2 += TPB) {
          int gp = gbase + s2;
          if (gp < cap) { cv[gp] = sv[s2]; ci[gp] = si[s2]; }
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();
  {
    const int n = scnt;
    if (n) {
      if (threadIdx.x == 0) gbase = atomicAdd(&counts[row * 2 + 1], n);
      __syncthreads();
      for (int s2 = threadIdx.x; s2 < n; s2 += TPB) {
        int gp = gbase + s2;
        if (gp < cap) { cv[gp] = sv[s2]; ci[gp] = si[s2]; }
      }
    }
  }
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


// ---- iter3: filter v4 — predicated skip + rare-path warp-aggregated append
// Common case (no admit in float4): 4 compares + 1 branch. Rare path: ballot
// over admitting lanes only, 1 global atomic per admitting warp-group,
// scattered global writes (~1% traffic).
__global__ void k_filter_v4(const float4* __restrict__ x, float t_hi, float t_lo,
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
  const long iters = (end > begin) ? (end - begin + TPB - 1) / TPB : 0;
  for (long it = 0; it < iters; ++it) {
    const long i = begin + it * TPB + threadIdx.x;
    float4 v = (i < end) ? xr[i] : make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
    const int nb4 = (v.x >= t_lo) + (v.y >= t_lo) + (v.z >= t_lo) + (v.w >= t_lo);
    // uniform warp vote; skip all coordination when nobody admits (common)
    if (__any_sync(~0u, nb4)) {
      // rare path: per-lane count -> warp exclusive prefix -> 1 atomic -> scatter
      int pre = nb4;
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(~0u, pre, o);
        if ((int)(threadIdx.x & 31) >= o) pre += y;
      }
      const int total = __shfl_sync(~0u, pre, 31);
      pre -= nb4;
      int base_slot = 0;
      if ((threadIdx.x & 31) == 31) base_slot = atomicAdd(&counts[row * 2 + 1], total);
      base_slot = __shfl_sync(~0u, base_slot, 31);
      if (nb4) {
        int slot = base_slot + pre;
        const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float f = vv[j];
          if (f >= t_lo) {
            c_hi += (f >= t_hi);
            if (slot < cap) { cv[slot] = f; ci[slot] = (int)(i * 4 + j); }
            ++slot;
          }
        }
      }
    }
  }
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


// ---- iter4: filter v5 — v4 + register double-buffer prefetch (decouple LDG
// issue from the data-dependent rare path so loads pipeline like pure-read)
__global__ void k_filter_v5(const float4* __restrict__ x, float t_hi, float t_lo,
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
  const long iters = (end > begin) ? (end - begin + TPB - 1) / TPB : 0;
  const float4 NEG = make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
  long i0 = begin + threadIdx.x;
  float4 a = (iters > 0 && i0 < end) ? xr[i0] : NEG;
  float4 b = (iters > 1 && i0 + TPB < end) ? xr[i0 + TPB] : NEG;
  for (long it = 0; it < iters; ++it) {
    const long inx = i0 + (it + 2) * TPB;
    float4 nxt = (it + 2 < iters && inx < end) ? xr[inx] : NEG;
    const float4 v = a;
    const long i = i0 + it * TPB;
    const int nb4 = (v.x >= t_lo) + (v.y >= t_lo) + (v.z >= t_lo) + (v.w >= t_lo);
    if (__any_sync(~0u, nb4)) {
      int pre = nb4;
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(~0u, pre, o);
        if ((int)(threadIdx.x & 31) >= o) pre += y;
      }
      const int total = __shfl_sync(~0u, pre, 31);
      pre -= nb4;
      int base_slot = 0;
      if ((threadIdx.x & 31) == 31) base_slot = atomicAdd(&counts[row * 2 + 1], total);
      base_slot = __shfl_sync(~0u, base_slot, 31);
      if (nb4) {
        int slot = base_slot + pre;
        const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float f = vv[j];
          if (f >= t_lo) {
            c_hi += (f >= t_hi);
            if (slot < cap) { cv[slot] = f; ci[slot] = (int)(i * 4 + j); }
            ++slot;
          }
        }
      }
    }
    a = b; b = nxt;
  }
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


// ---- iter5: filter v6 — smem-slot allocation (no global-atomic latency in
// loop), smem staging, high-water peek flush, end bulk flush + prefetch.
#define SCAP6 4096
__global__ void k_filter_v6(const float4* __restrict__ x, float t_hi, float t_lo,
                            float* __restrict__ cand_v, int* __restrict__ cand_i,
                            int* __restrict__ counts, int* __restrict__ tickets,
                            long n4_per_row, int ctas_per_row, int cap) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const float4* xr = x + (long)row * n4_per_row;
  float* cv = cand_v + (long)row * cap;
  int* ci = cand_i + (long)row * cap;
  __shared__ float sv[SCAP6];
  __shared__ int si[SCAP6];
  __shared__ int scnt, gbase;
  if (threadIdx.x == 0) scnt = 0;
  __syncthreads();
  int c_hi = 0;
  const long chunk = (n4_per_row + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4_per_row, begin + chunk);
  const long iters = (end > begin) ? (end - begin + TPB - 1) / TPB : 0;
  const float4 NEG = make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
  const long i0 = begin + threadIdx.x;
  float4 a = (iters > 0 && i0 < end) ? xr[i0] : NEG;
  float4 b = (iters > 1 && i0 + TPB < end) ? xr[i0 + TPB] : NEG;
  for (long it = 0; it < iters; ++it) {
    const long inx = i0 + (it + 2) * TPB;
    float4 nxt = (it + 2 < iters && inx < end) ? xr[inx] : NEG;
    const float4 v = a;
    const long i = i0 + it * TPB;
    const int nb4 = (v.x >= t_lo) + (v.y >= t_lo) + (v.z >= t_lo) + (v.w >= t_lo);
    if (nb4) {
      int s = atomicAdd(&scnt, nb4);  // smem: ~20cyc, off the load critical path
      const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float f = vv[j];
        if (f >= t_lo) {
          c_hi += (f >= t_hi);
          if (s < SCAP6) { sv[s] = f; si[s] = (int)(i * 4 + j); }
          ++s;
        }
      }
    }
    // high-water peek (non-atomic read; SCAP6 - TPB*4 margin absorbs races)
    if (__syncthreads_or(scnt >= SCAP6 - TPB * 4)) {
      const int n = min(scnt, SCAP6);
      if (threadIdx.x == 0) { gbase = atomicAdd(&counts[row * 2 + 1], n); scnt = 0; }
      __syncthreads();
      for (int s2 = threadIdx.x; s2 < n; s2 += TPB) {
        int gp = gbase + s2;
        if (gp < cap) { cv[gp] = sv[s2]; ci[gp] = si[s2]; }
      }
      __syncthreads();
    }
    a = b; b = nxt;
  }
  __syncthreads();
  {
    const int n = min(scnt, SCAP6);
    if (threadIdx.x == 0) gbase = atomicAdd(&counts[row * 2 + 1], n);
    __syncthreads();
    for (int s2 = threadIdx.x; s2 < n; s2 += TPB) {
      int gp = gbase + s2;
      if (gp < cap) { cv[gp] = sv[s2]; ci[gp] = si[s2]; }
    }
  }
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

// ---- iter6: filter v7 (no mid-loop barrier; overflow -> flag+retry contract) — smem-slot allocation (no global-atomic latency in
// loop), smem staging, high-water peek flush, end bulk flush + prefetch.
#define SCAP7 4096
__global__ void k_filter_v7(const float4* __restrict__ x, float t_hi, float t_lo,
                            float* __restrict__ cand_v, int* __restrict__ cand_i,
                            int* __restrict__ counts, int* __restrict__ tickets,
                            long n4_per_row, int ctas_per_row, int cap) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const float4* xr = x + (long)row * n4_per_row;
  float* cv = cand_v + (long)row * cap;
  int* ci = cand_i + (long)row * cap;
  __shared__ float sv[SCAP7];
  __shared__ int si[SCAP7];
  __shared__ int scnt, gbase;
  if (threadIdx.x == 0) scnt = 0;
  __syncthreads();
  int c_hi = 0;
  const long chunk = (n4_per_row + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4_per_row, begin + chunk);
  const long iters = (end > begin) ? (end - begin + TPB - 1) / TPB : 0;
  const float4 NEG = make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
  const long i0 = begin + threadIdx.x;
  float4 a = (iters > 0 && i0 < end) ? xr[i0] : NEG;
  float4 b = (iters > 1 && i0 + TPB < end) ? xr[i0 + TPB] : NEG;
  for (long it = 0; it < iters; ++it) {
    const long inx = i0 + (it + 2) * TPB;
    float4 nxt = (it + 2 < iters && inx < end) ? xr[inx] : NEG;
    const float4 v = a;
    const long i = i0 + it * TPB;
    const int nb4 = (v.x >= t_lo) + (v.y >= t_lo) + (v.z >= t_lo) + (v.w >= t_lo);
    if (nb4) {
      int s = atomicAdd(&scnt, nb4);  // smem: ~20cyc, off the load critical path
      const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float f = vv[j];
        if (f >= t_lo) {
          c_hi += (f >= t_hi);
          if (s < SCAP7) { sv[s] = f; si[s] = (int)(i * 4 + j); }
          ++s;
        }
      }
    }
    a = b; b = nxt;
  }
  __syncthreads();
  {
    const int n = min(scnt, SCAP7);
    if (scnt > SCAP7 && threadIdx.x == 0) atomicExch(&counts[row * 2], -1);  // overflow flag (caller retries)
    if (threadIdx.x == 0) gbase = atomicAdd(&counts[row * 2 + 1], n);
    __syncthreads();
    for (int s2 = threadIdx.x; s2 < n; s2 += TPB) {
      int gp = gbase + s2;
      if (gp < cap) { cv[gp] = sv[s2]; ci[gp] = si[s2]; }
    }
  }
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


// ---- iter7: filter v8 — per-warp smem counters/regions (contention /16),
// end flush compacts the 16 regions via a small prefix pass.
#define WREG 256   // slots per warp region (16 x 256 = 4096 pairs = 32KB smem)
__global__ void __launch_bounds__(TPB, 2)
k_filter_v8(const float4* __restrict__ x, float t_hi, float t_lo,
            float* __restrict__ cand_v, int* __restrict__ cand_i,
            int* __restrict__ counts, int* __restrict__ tickets,
            long n4_per_row, int ctas_per_row, int cap) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const int wid = threadIdx.x >> 5;
  const float4* xr = x + (long)row * n4_per_row;
  float* cv = cand_v + (long)row * cap;
  int* ci = cand_i + (long)row * cap;
  __shared__ float sv[16 * WREG];
  __shared__ int si[16 * WREG];
  __shared__ int wcnt[16];
  __shared__ int wbase[16];
  __shared__ int gbase;
  if (threadIdx.x < 16) wcnt[threadIdx.x] = 0;
  __syncthreads();
  int c_hi = 0;
  const long chunk = (n4_per_row + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4_per_row, begin + chunk);
  const long iters = (end > begin) ? (end - begin + TPB - 1) / TPB : 0;
  const float4 NEG = make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
  const long i0 = begin + threadIdx.x;
  float4 a = (iters > 0 && i0 < end) ? xr[i0] : NEG;
  float4 b = (iters > 1 && i0 + TPB < end) ? xr[i0 + TPB] : NEG;
  float* wv = sv + wid * WREG;
  int* wi = si + wid * WREG;
  for (long it = 0; it < iters; ++it) {
    const long inx = i0 + (it + 2) * TPB;
    float4 nxt = (it + 2 < iters && inx < end) ? xr[inx] : NEG;
    const float4 v = a;
    const long i = i0 + it * TPB;
    const int nb4 = (v.x >= t_lo) + (v.y >= t_lo) + (v.z >= t_lo) + (v.w >= t_lo);
    if (nb4) {
      int s = atomicAdd(&wcnt[wid], nb4);
      const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float f = vv[j];
        if (f >= t_lo) {
          c_hi += (f >= t_hi);
          if (s < WREG) { wv[s] = f; wi[s] = (int)(i * 4 + j); }
          ++s;
        }
      }
    }
    a = b; b = nxt;
  }
  __syncthreads();
  // compact: thread 0 prefixes the 16 warp counts (tiny), then all copy
  if (threadIdx.x == 0) {
    int tot = 0, ovf = 0;
    for (int w = 0; w < 16; ++w) {
      wbase[w] = tot;
      int c = min(wcnt[w], WREG);
      ovf |= (wcnt[w] > WREG);
      tot += c;
    }
    gbase = atomicAdd(&counts[row * 2 + 1], tot);
    if (ovf) atomicExch(&counts[row * 2], -1);  // overflow flag -> caller retry
  }
  __syncthreads();
  for (int w = 0; w < 16; ++w) {
    const int n = min(wcnt[w], WREG);
    for (int s2 = threadIdx.x; s2 < n; s2 += TPB) {
      int gp = gbase + wbase[w] + s2;
      if (gp < cap) { cv[gp] = sv[w * WREG + s2]; ci[gp] = si[w * WREG + s2]; }
    }
  }
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


// ---- iter8: filter v9 — zero-atomic zero-smem append: ballot-rank + per-warp
// register running total + per-warp GLOBAL segments (cap/16 each). Per-warp
// counts written at end to counts[row*18 + 2 + w]; caller/tail compacts.
__global__ void k_filter_v9(const float4* __restrict__ x, float t_hi, float t_lo,
                            float* __restrict__ cand_v, int* __restrict__ cand_i,
                            int* __restrict__ counts18, int* __restrict__ tickets,
                            long n4_per_row, int ctas_per_row, int cap) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float4* xr = x + (long)row * n4_per_row;
  // per (CTA, warp) global segment: cap is per-row; segment = cap / (16*ctas_per_row)
  const int nseg = 16 * ctas_per_row;
  const int segcap = cap / nseg;
  const int seg = sub * 16 + wid;
  float* cv = cand_v + (long)row * cap + (long)seg * segcap;
  int* ci = cand_i + (long)row * cap + (long)seg * segcap;
  int* cnt = counts18 + (long)row * (2 + nseg);
  int c_hi = 0;
  int wtot = 0;  // warp-uniform running total (every lane maintains it)
  const long chunk = (n4_per_row + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4_per_row, begin + chunk);
  const long iters = (end > begin) ? (end - begin + TPB - 1) / TPB : 0;
  const float4 NEG = make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
  const long i0 = begin + threadIdx.x;
  float4 a = (iters > 0 && i0 < end) ? xr[i0] : NEG;
  float4 b = (iters > 1 && i0 + TPB < end) ? xr[i0 + TPB] : NEG;
  for (long it = 0; it < iters; ++it) {
    const long inx = i0 + (it + 2) * TPB;
    float4 nxt = (it + 2 < iters && inx < end) ? xr[inx] : NEG;
    const float4 v = a;
    const long i = i0 + it * TPB;
    const int nb4 = (v.x >= t_lo) + (v.y >= t_lo) + (v.z >= t_lo) + (v.w >= t_lo);
    if (__any_sync(~0u, nb4)) {
      const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float f = vv[j];
        const bool adm = (f >= t_lo);
        const unsigned bal = __ballot_sync(~0u, adm);
        if (adm) {
          const int rank = __popc(bal & ((1u << lane) - 1));
          const int slot = wtot + rank;
          c_hi += (f >= t_hi);
          if (slot < segcap) { cv[slot] = f; ci[slot] = (int)(i * 4 + j); }
        }
        wtot += __popc(bal);
      }
    }
    a = b; b = nxt;
  }
  if (lane == 0) cnt[2 + seg] = wtot;  // (overflow detectable: wtot > segcap)
  // c_hi reduce across CTA + ticket
  __shared__ int sh[TPB / 32];
  for (int o = 16; o; o >>= 1) c_hi += __shfl_down_sync(~0u, c_hi, o);
  if (lane == 0) sh[wid] = c_hi;
  __syncthreads();
  if (threadIdx.x < TPB / 32) {
    c_hi = sh[threadIdx.x];
    for (int o = TPB / 64; o; o >>= 1) c_hi += __shfl_down_sync(~0u, c_hi, o);
    if (threadIdx.x == 0) {
      atomicAdd(&cnt[0], c_hi);
      __threadfence();
      int t = atomicAdd(&tickets[row], 1);
      if (t == ctas_per_row - 1) tickets[row] = 0;
    }
  }
}


// ---- iter9: filter v10 — group-level prefix append (unordered within group,
// tail re-ranks anyway) + TPB-parametrized clone at 1024 threads.
template <int NT>
__global__ void k_filter_v10(const float4* __restrict__ x, float t_hi, float t_lo,
                             float* __restrict__ cand_v, int* __restrict__ cand_i,
                             int* __restrict__ counts, int* __restrict__ tickets,
                             long n4_per_row, int ctas_per_row, int cap) {
  const int row = blockIdx.x / ctas_per_row;
  const int sub = blockIdx.x % ctas_per_row;
  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarp = NT / 32;
  const float4* xr = x + (long)row * n4_per_row;
  const int nseg = nwarp * ctas_per_row;
  const int segcap = cap / nseg;
  const int seg = sub * nwarp + wid;
  float* cv = cand_v + (long)row * cap + (long)seg * segcap;
  int* ci = cand_i + (long)row * cap + (long)seg * segcap;
  int* cnt = counts + (long)row * (2 + nseg);
  int c_hi = 0;
  int wtot = 0;
  const long chunk = (n4_per_row + ctas_per_row - 1) / ctas_per_row;
  const long begin = (long)sub * chunk;
  const long end = min(n4_per_row, begin + chunk);
  const long iters = (end > begin) ? (end - begin + NT - 1) / NT : 0;
  const float4 NEG = make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
  const long i0 = begin + threadIdx.x;
  float4 a = (iters > 0 && i0 < end) ? xr[i0] : NEG;
  float4 b = (iters > 1 && i0 + NT < end) ? xr[i0 + NT] : NEG;
  for (long it = 0; it < iters; ++it) {
    const long inx = i0 + (it + 2) * NT;
    float4 nxt = (it + 2 < iters && inx < end) ? xr[inx] : NEG;
    const float4 v = a;
    const long i = i0 + it * NT;
    const int nb4 = (v.x >= t_lo) + (v.y >= t_lo) + (v.z >= t_lo) + (v.w >= t_lo);
    if (__any_sync(~0u, nb4)) {
      // one warp-prefix over per-lane admit counts; unordered pack per lane
      int pre = nb4;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(~0u, pre, o);
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
            if (slot < segcap) { cv[slot] = f; ci[slot] = (int)(i * 4 + j); }
            ++slot;
          }
        }
      }
    }
    a = b; b = nxt;
  }
  if (lane == 0) cnt[2 + seg] = wtot;
  __shared__ int sh[NT / 32];
  for (int o = 16; o; o >>= 1) c_hi += __shfl_down_sync(~0u, c_hi, o);
  if (lane == 0) sh[wid] = c_hi;
  __syncthreads();
  if (threadIdx.x < NT / 32) {
    c_hi = sh[threadIdx.x];
#pragma unroll
    for (int o = NT / 64; o; o >>= 1) c_hi += __shfl_down_sync(~0u, c_hi, o);
    if (threadIdx.x == 0) {
      atomicAdd(&cnt[0], c_hi);
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

void filter_v3(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
               torch::Tensor cand_i, torch::Tensor counts, torch::Tensor tickets,
               int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  k_filter_v3<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
      cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
      tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

void filter_v4(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
               torch::Tensor cand_i, torch::Tensor counts, torch::Tensor tickets,
               int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  k_filter_v4<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
      cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
      tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

void filter_v5(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
               torch::Tensor cand_i, torch::Tensor counts, torch::Tensor tickets,
               int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  k_filter_v5<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
      cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
      tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

void filter_v6(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
               torch::Tensor cand_i, torch::Tensor counts, torch::Tensor tickets,
               int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  k_filter_v6<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
      cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
      tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

void filter_v7(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
               torch::Tensor cand_i, torch::Tensor counts, torch::Tensor tickets,
               int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  k_filter_v7<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
      cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
      tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

void filter_v8(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
               torch::Tensor cand_i, torch::Tensor counts, torch::Tensor tickets,
               int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  k_filter_v8<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
      cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
      tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

void filter_v9(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
               torch::Tensor cand_i, torch::Tensor counts18, torch::Tensor tickets,
               int ctas_per_row) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  k_filter_v9<<<rows * ctas_per_row, TPB>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
      cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts18.data_ptr<int>(),
      tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

void filter_v10(torch::Tensor x, double t_hi, double t_lo, torch::Tensor cand_v,
                torch::Tensor cand_i, torch::Tensor counts, torch::Tensor tickets,
                int ctas_per_row, int nt) {
  const long n4 = x.size(1) / 4;
  const int rows = x.size(0);
  const int cap = cand_v.size(1);
  if (nt == 1024)
    k_filter_v10<1024><<<rows * ctas_per_row, 1024>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
        cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
        tickets.data_ptr<int>(), n4, ctas_per_row, cap);
  else
    k_filter_v10<512><<<rows * ctas_per_row, 512>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()), (float)t_hi, (float)t_lo,
        cand_v.data_ptr<float>(), cand_i.data_ptr<int>(), counts.data_ptr<int>(),
        tickets.data_ptr<int>(), n4, ctas_per_row, cap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("empty_launch", &empty_launch);
  m.def("stream_reduce", &stream_reduce);
  m.def("filter_append", &filter_append);
  m.def("filter_v2", &filter_v2);
  m.def("filter_v3", &filter_v3);
  m.def("filter_v4", &filter_v4);
  m.def("filter_v5", &filter_v5);
  m.def("filter_v6", &filter_v6);
  m.def("filter_v7", &filter_v7);
  m.def("filter_v8", &filter_v8);
  m.def("filter_v9", &filter_v9);
  m.def("filter_v10", &filter_v10);
  m.def("stream_reduce2", &stream_reduce2);
}
