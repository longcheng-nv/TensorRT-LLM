#include "kernel.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cfloat>

namespace cg = cooperative_groups;

namespace {

constexpr int CAP = 6144;
constexpr int RADIX = 256;
constexpr int MAX_WARPS = 32;

struct alignas(16) SharedState {
  int local_count;
  int total_count;
  int output_count;
  int done;             // 0: refine, 1: candidates, 2: boundary fallback
  int lo_count;
  int hi_count;
  int iteration;
  int pad0;
  float threshold;
  float lo;
  float hi;
  float local_min;
  float local_max;
  int probe_cnt[8];   // per-CTA multi-probe counts (DSMEM-merged by rank 0)
};

__device__ __forceinline__ unsigned ordered_key(float x) {
  unsigned u = __float_as_uint(x);
  return u ^ ((static_cast<int>(u) < 0) ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ float ordered_float(unsigned k) {
  unsigned u = k ^ ((k & 0x80000000u) ? 0x80000000u : 0xffffffffu);
  return __uint_as_float(u);
}

__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
  for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
  return v;
}

__device__ __forceinline__ float warp_min(float v) {
#pragma unroll
  for (int d = 16; d; d >>= 1) v = fminf(v, __shfl_down_sync(0xffffffffu, v, d));
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, d));
  return v;
}

__device__ __forceinline__ void cluster_or_block_sync(
    cg::cluster_group& cluster, int cluster_size) {
  if (cluster_size == 1)
    __syncthreads();
  else
    cluster.sync();
}

__device__ __forceinline__ int reserve_warp(int predicate, int* counter) {
  unsigned mask = __ballot_sync(0xffffffffu, predicate);
  int lane = threadIdx.x & 31;
  int base = 0;
  if (lane == 0 && mask) base = atomicAdd(counter, __popc(mask));
  base = __shfl_sync(0xffffffffu, base, 0);
  return base + __popc(mask & ((1u << lane) - 1u));
}

__device__ __forceinline__ int block_sum_dense(int value, int* warp_values) {
  value = warp_sum(value);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    int x = lane < (blockDim.x >> 5) ? warp_values[lane] : 0;
    x = warp_sum(x);
    if (lane == 0) warp_values[0] = x;
  }
  __syncthreads();
  return warp_values[0];
}

__device__ __forceinline__ unsigned block_min_dense(unsigned value,
                                                     unsigned* warp_values) {
#pragma unroll
  for (int d = 16; d; d >>= 1)
    value = min(value, __shfl_down_sync(0xffffffffu, value, d));
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    unsigned x = lane < (blockDim.x >> 5) ? warp_values[lane] : 0xffffffffu;
#pragma unroll
    for (int d = 16; d; d >>= 1)
      x = min(x, __shfl_down_sync(0xffffffffu, x, d));
    if (lane == 0) warp_values[0] = x;
  }
  __syncthreads();
  return warp_values[0];
}

// Dense n=k+small path. The temporal hint supplies the initial threshold;
// a full-row count verifies it. Only on a miss do we exactly refine by
// identifying the few excluded minima, then emit the surviving index set.
template <int Threads>
__global__ __launch_bounds__(Threads, Threads <= 256 ? 2 : 1) void dense_gvr_topk_kernel(
    const float* __restrict__ logits, const int* __restrict__ pre_idx,
    int* __restrict__ output, int stride, int n, int k) {
  int row_id = blockIdx.x;
  const float* row = logits + static_cast<long long>(row_id) * stride;
  const int* prior = pre_idx + static_cast<long long>(row_id) * k;
  int* out = output + static_cast<long long>(row_id) * k;
  __shared__ unsigned warp_keys[MAX_WARPS];
  __shared__ int warp_indices[MAX_WARPS];
  __shared__ int excluded[32];
  __shared__ int output_count;

  unsigned local_prior_min = 0xffffffffu;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int idx = prior[i];
    if (static_cast<unsigned>(idx) < static_cast<unsigned>(n))
      local_prior_min = min(local_prior_min, ordered_key(row[idx]));
  }
  unsigned prior_threshold = block_min_dense(local_prior_min, warp_keys);
  int local_admitted = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    local_admitted += ordered_key(row[i]) >= prior_threshold;
  int admitted = block_sum_dense(local_admitted, warp_indices);
  if (admitted == k) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) out[i] = prior[i];
    return;
  }

  int remove_count = n - k;
  for (int removed = 0; removed < remove_count; ++removed) {
    unsigned best_key = 0xffffffffu;
    int best_index = -1;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      bool skip = false;
      for (int j = 0; j < removed; ++j) skip |= i == excluded[j];
      unsigned key = skip ? 0xffffffffu : ordered_key(row[i]);
      if (key < best_key) { best_key = key; best_index = i; }
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) {
      unsigned other_key = __shfl_down_sync(0xffffffffu, best_key, d);
      int other_index = __shfl_down_sync(0xffffffffu, best_index, d);
      if (other_key < best_key) { best_key = other_key; best_index = other_index; }
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_keys[warp] = best_key; warp_indices[warp] = best_index; }
    __syncthreads();
    if (warp == 0) {
      best_key = lane < (Threads >> 5) ? warp_keys[lane] : 0xffffffffu;
      best_index = lane < (Threads >> 5) ? warp_indices[lane] : -1;
#pragma unroll
      for (int d = 16; d; d >>= 1) {
        unsigned other_key = __shfl_down_sync(0xffffffffu, best_key, d);
        int other_index = __shfl_down_sync(0xffffffffu, best_index, d);
        if (other_key < best_key) { best_key = other_key; best_index = other_index; }
      }
      if (lane == 0) excluded[removed] = best_index;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) output_count = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    bool keep = true;
    for (int j = 0; j < remove_count; ++j) keep &= i != excluded[j];
    int pos = reserve_warp(keep, &output_count);
    if (keep) out[pos] = i;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void hint_verify_kernel(
    const float* __restrict__ logits, const int* __restrict__ pre_idx,
    int* __restrict__ output, int stride, int n, int k) {
  int row_id = blockIdx.x;
  const float* row = logits + static_cast<long long>(row_id) * stride;
  const int* prior = pre_idx + static_cast<long long>(row_id) * k;
  int* out = output + static_cast<long long>(row_id) * k;
  __shared__ unsigned warp_keys[MAX_WARPS];
  __shared__ int warp_counts[MAX_WARPS];

  unsigned local_min = 0xffffffffu;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int idx = prior[i];
    if (static_cast<unsigned>(idx) < static_cast<unsigned>(n))
      local_min = min(local_min, ordered_key(row[idx]));
  }
  unsigned threshold = block_min_dense(local_min, warp_keys);
  int local_count = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    local_count += ordered_key(row[i]) >= threshold;
  int admitted = block_sum_dense(local_count, warp_counts);
  if (admitted == k) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) out[i] = prior[i];
  } else if (threadIdx.x == 0) {
    out[0] = -1;
  }
}

// TC: compile-time thread count (0 => runtime blockDim.x). Passing the launch
// width as a constant lets nvcc fold loop strides / the warp-count reduction,
// speeding the latency-bound secant scan loop (runs up to 64x/cell).
template <int TC = 0>
__device__ void block_scan_count(const float* row, int begin, int end, float threshold,
                                 int* scratch, int& count) {
  const int bd = TC ? TC : blockDim.x;
  int c = 0;
  // ILP: UF independent float4 loads in flight per iteration to hide global
  // memory latency (few active SMs at low batch => latency-bound scans).
  const int UF = 4;
  const int step1 = bd * 4;
  const int wide = step1 * UF;
  int base = begin + threadIdx.x * 4;
  for (; base + wide - step1 + 3 < end; base += wide) {
    float4 v[UF];
#pragma unroll
    for (int u = 0; u < UF; ++u)
      v[u] = *reinterpret_cast<const float4*>(row + base + u * step1);
#pragma unroll
    for (int u = 0; u < UF; ++u)
      c += (v[u].x >= threshold) + (v[u].y >= threshold) +
           (v[u].z >= threshold) + (v[u].w >= threshold);
  }
  for (; base + 3 < end; base += step1) {
    float4 values = *reinterpret_cast<const float4*>(row + base);
    c += (values.x >= threshold) + (values.y >= threshold) +
         (values.z >= threshold) + (values.w >= threshold);
  }
  int tail = begin + ((end - begin) & ~3) + threadIdx.x;
  if (tail < end) c += row[tail] >= threshold;
  c = warp_sum(c);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) {
    scratch[warp] = c;
  }
  __syncthreads();
  if (warp == 0) {
    int nw = bd >> 5;
    int bc = lane < nw ? scratch[lane] : 0;
    bc = warp_sum(bc);
    if (lane == 0) count = bc;
  }
}

__device__ float block_hint_min(const float* row, const int* hints, int n, int k,
                                int* scratch) {
  float mn = FLT_MAX;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int idx = hints[i];
    if (static_cast<unsigned>(idx) < static_cast<unsigned>(n)) mn = fminf(mn, row[idx]);
  }
  mn = warp_min(mn);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) reinterpret_cast<float*>(scratch)[warp] = mn;
  __syncthreads();
  if (warp == 0) {
    int nw = blockDim.x >> 5;
    float x = lane < nw ? reinterpret_cast<float*>(scratch)[lane] : FLT_MAX;
    x = warp_min(x);
    if (lane == 0) reinterpret_cast<float*>(scratch)[0] = x;
  }
  __syncthreads();
  return reinterpret_cast<float*>(scratch)[0];
}

// Cache gathered prior values and place two low-to-high threshold rungs at
// the 85% and 35% upper-tail quantiles, matching the production R0 ladder.
__device__ void block_hint_rungs(const float* row, const int* hints, int n, int k,
                                 unsigned* cache, int* hist, int* scratch,
                                 float& pmin, float& pmax, float& rung0, float& rung1) {
  float mn = FLT_MAX, mx = -FLT_MAX;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int idx = hints[i];
    float v = static_cast<unsigned>(idx) < static_cast<unsigned>(n) ? row[idx] : -FLT_MAX;
    cache[i] = __float_as_uint(v);
    if (v != -FLT_MAX) {
      mn = fminf(mn, v);
      mx = fmaxf(mx, v);
    }
  }
  mn = warp_min(mn);
  mx = warp_max(mx);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) {
    reinterpret_cast<float*>(scratch)[warp] = mn;
    reinterpret_cast<float*>(scratch + MAX_WARPS)[warp] = mx;
  }
  __syncthreads();
  if (warp == 0) {
    int nw = blockDim.x >> 5;
    float a = lane < nw ? reinterpret_cast<float*>(scratch)[lane] : FLT_MAX;
    float b = lane < nw ? reinterpret_cast<float*>(scratch + MAX_WARPS)[lane] : -FLT_MAX;
    a = warp_min(a);
    b = warp_max(b);
    if (lane == 0) {
      reinterpret_cast<float*>(scratch)[0] = a;
      reinterpret_cast<float*>(scratch + MAX_WARPS)[0] = b;
    }
  }
  __syncthreads();
  pmin = reinterpret_cast<float*>(scratch)[0];
  pmax = reinterpret_cast<float*>(scratch + MAX_WARPS)[0];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
  __syncthreads();
  float width = fmaxf(pmax - pmin, 1.0e-20f);
  float scale = 255.999f / width;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    float v = __uint_as_float(cache[i]);
    int bin = max(0, min(255, static_cast<int>((v - pmin) * scale)));
    atomicAdd(hist + bin, 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    int lane = threadIdx.x;
    int group_sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) group_sum += hist[lane * 8 + j];
    int inclusive = group_sum;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v = __shfl_down_sync(0xffffffffu, inclusive, d);
      if (lane + d < 32) inclusive += v;
    }
    int needs[2] = {(85 * k + 99) / 100, (35 * k + 99) / 100};
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      unsigned ge = __ballot_sync(0xffffffffu, inclusive >= needs[q]);
      int owner = 31 - __clz(ge);
      int above = __shfl_sync(0xffffffffu, inclusive - group_sum, owner);
      if (lane == 0) {
        int residual = needs[q] - above;
        int found = owner * 8;
#pragma unroll
        for (int b = 7; b >= 0; --b) {
          int bin = owner * 8 + b;
          int count = hist[bin];
          if (residual > count) residual -= count;
          else { found = bin; break; }
        }
        reinterpret_cast<float*>(scratch + 2 * MAX_WARPS)[q] =
            pmin + (static_cast<float>(found) / scale);
      }
    }
  }
  if (threadIdx.x == 0) {
    rung0 = reinterpret_cast<float*>(scratch + 2 * MAX_WARPS)[0];
    rung1 = reinterpret_cast<float*>(scratch + 2 * MAX_WARPS)[1];
  }
}

__device__ void block_scan_two(const float* row, int begin, int end, float t0, float t1,
                               int* scratch, int& c0, int& c1) {
  int a = 0, b = 0;
  const int UF = 4;
  const int step1 = blockDim.x * 4;
  const int wide = step1 * UF;
  int base = begin + threadIdx.x * 4;
  for (; base + wide - step1 + 3 < end; base += wide) {
    float4 v[UF];
#pragma unroll
    for (int u = 0; u < UF; ++u)
      v[u] = *reinterpret_cast<const float4*>(row + base + u * step1);
#pragma unroll
    for (int u = 0; u < UF; ++u) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float x = (&v[u].x)[q]; a += x >= t0; b += x >= t1;
      }
    }
  }
  for (; base + 3 < end; base += blockDim.x * 4) {
    float4 values = *reinterpret_cast<const float4*>(row + base);
    a += values.x >= t0; b += values.x >= t1;
    a += values.y >= t0; b += values.y >= t1;
    a += values.z >= t0; b += values.z >= t1;
    a += values.w >= t0; b += values.w >= t1;
  }
  int tail = begin + ((end - begin) & ~3) + threadIdx.x;
  if (tail < end) {
    float v = row[tail];
    a += v >= t0;
    b += v >= t1;
  }
  a = warp_sum(a);
  b = warp_sum(b);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) {
    scratch[warp] = a;
    scratch[MAX_WARPS + warp] = b;
  }
  __syncthreads();
  if (warp == 0) {
    int nw = blockDim.x >> 5;
    int x = lane < nw ? scratch[lane] : 0;
    int y = lane < nw ? scratch[MAX_WARPS + lane] : 0;
    x = warp_sum(x);
    y = warp_sum(y);
    if (lane == 0) {
      c0 = x;
      c1 = y;
    }
  }
}

// Count row[begin:end] >= each of NP thresholds t[0..NP-1] (ascending) in ONE
// full-row scan. Latency-bound scans make extra comparisons nearly free, so
// probing many thresholds per pass narrows the k-th bracket much faster than a
// 2-point secant (fewer barrier-heavy iterations). Results in cnt[0..NP-1].
template <int NP>
__device__ void block_scan_multi(const float* row, int begin, int end,
                                 const float* t, int* scratch, int* cnt) {
  int c[NP];
#pragma unroll
  for (int j = 0; j < NP; ++j) c[j] = 0;
  const int UF = 2;
  const int wide = blockDim.x * 4 * UF;
  int base = begin + threadIdx.x * 4;
  for (; base + blockDim.x * 4 + 3 < end; base += wide) {
    float4 v0 = *reinterpret_cast<const float4*>(row + base);
    float4 v1 = *reinterpret_cast<const float4*>(row + base + blockDim.x * 4);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float x = (&v0.x)[q];
#pragma unroll
      for (int j = 0; j < NP; ++j) c[j] += (x >= t[j]);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float x = (&v1.x)[q];
#pragma unroll
      for (int j = 0; j < NP; ++j) c[j] += (x >= t[j]);
    }
  }
  for (; base + 3 < end; base += blockDim.x * 4) {
    float4 v0 = *reinterpret_cast<const float4*>(row + base);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float x = (&v0.x)[q];
#pragma unroll
      for (int j = 0; j < NP; ++j) c[j] += (x >= t[j]);
    }
  }
  int tail = begin + ((end - begin) & ~3) + threadIdx.x;
  if (tail < end) {
    float x = row[tail];
#pragma unroll
    for (int j = 0; j < NP; ++j) c[j] += (x >= t[j]);
  }
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < NP; ++j) {
#pragma unroll
    for (int d = 16; d; d >>= 1) c[j] += __shfl_down_sync(0xffffffffu, c[j], d);
    if (lane == 0) scratch[j * MAX_WARPS + warp] = c[j];
  }
  __syncthreads();
  if (warp == 0) {
    int nw = blockDim.x >> 5;
#pragma unroll
    for (int j = 0; j < NP; ++j) {
      int v = lane < nw ? scratch[j * MAX_WARPS + lane] : 0;
#pragma unroll
      for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
      if (lane == 0) scratch[j * MAX_WARPS] = v;
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < NP; ++j) cnt[j] = scratch[j * MAX_WARPS];
}

__device__ void collect_local(const float* row, int begin, int end, float threshold,
                              unsigned* keys, int* vals, SharedState* st, int cap = CAP) {
  if (threadIdx.x == 0) st->local_count = 0;
  __syncthreads();
  int lane = threadIdx.x & 31;
  for (int base = begin + (threadIdx.x & ~31) * 4;
       base < end; base += blockDim.x * 4) {
    int vector_index = base + lane * 4;
    float4 vector;
    if (vector_index + 3 < end) {
      vector = *reinterpret_cast<const float4*>(row + vector_index);
    } else {
      vector.x = vector_index < end ? row[vector_index] : -FLT_MAX;
      vector.y = vector_index + 1 < end ? row[vector_index + 1] : -FLT_MAX;
      vector.z = vector_index + 2 < end ? row[vector_index + 2] : -FLT_MAX;
      vector.w = vector_index + 3 < end ? row[vector_index + 3] : -FLT_MAX;
    }
    float values[4] = {vector.x, vector.y, vector.z, vector.w};
    unsigned masks[4];
    int warp_total = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      int i = vector_index + q;
      masks[q] = __ballot_sync(0xffffffffu, i < end && values[q] >= threshold);
      warp_total += __popc(masks[q]);
    }
    int leader_base = 0;
    if (lane == 0 && warp_total) leader_base = atomicAdd(&st->local_count, warp_total);
    leader_base = __shfl_sync(0xffffffffu, leader_base, 0);
    int q_offset = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      int i = vector_index + q;
      bool pred = masks[q] & (1u << lane);
      if (pred) {
        int pos = leader_base + q_offset +
                  __popc(masks[q] & ((1u << lane) - 1u));
        if (pos < cap) {
          keys[pos] = ordered_key(values[q]);
          vals[pos] = i;
        }
      }
      q_offset += __popc(masks[q]);
    }
  }
}

template <int MODE>
__device__ void emit_global(const float* row, int begin, int end, float boundary,
                            int* output, int k, SharedState* root) {
  int lane = threadIdx.x & 31;
  for (int base = begin + (threadIdx.x & ~31); base < end; base += blockDim.x) {
    int i = base + lane;
    float v = i < end ? row[i] : -FLT_MAX;
    bool pred = i < end && (MODE == 0 ? v > boundary : v == boundary);
    unsigned mask = __ballot_sync(0xffffffffu, pred);
    int base_out = 0;
    if (lane == 0 && mask) base_out = atomicAdd(&root->output_count, __popc(mask));
    base_out = __shfl_sync(0xffffffffu, base_out, 0);
    if (pred) {
      int pos = base_out + __popc(mask & ((1u << lane) - 1u));
      if (pos < k) output[pos] = i;
    }
  }
}

template <int MODE>
__device__ void emit_candidates(const unsigned* keys, const int* vals, int count,
                                unsigned boundary, int* output, int k,
                                SharedState* root) {
  int lane = threadIdx.x & 31;
  for (int base = (threadIdx.x & ~31); base < count; base += blockDim.x) {
    int i = base + lane;
    bool pred = i < count && (MODE == 0 ? keys[i] > boundary : keys[i] == boundary);
    unsigned mask = __ballot_sync(0xffffffffu, pred);
    int base_out = 0;
    if (lane == 0 && mask) base_out = atomicAdd(&root->output_count, __popc(mask));
    base_out = __shfl_sync(0xffffffffu, base_out, 0);
    if (pred) {
      int pos = base_out + __popc(mask & ((1u << lane) - 1u));
      if (pos < k) output[pos] = vals[i];
    }
  }
}

__device__ void emit_candidates_both(const unsigned* keys, const int* vals,
                                     int count, unsigned boundary,
                                     int greater_count, int* output, int k,
                                     SharedState* root) {
  int lane = threadIdx.x & 31;
  int equal_needed = k - greater_count;
  for (int base = (threadIdx.x & ~31); base < count; base += blockDim.x) {
    int i = base + lane;
    bool gt = i < count && keys[i] > boundary;
    bool eq = i < count && keys[i] == boundary;
    unsigned gt_mask = __ballot_sync(0xffffffffu, gt);
    unsigned eq_mask = __ballot_sync(0xffffffffu, eq);
    int gt_base = 0, eq_base = 0;
    if (lane == 0) {
      if (gt_mask) gt_base = atomicAdd(&root->output_count, __popc(gt_mask));
      if (eq_mask) eq_base = atomicAdd(&root->local_count, __popc(eq_mask));
    }
    gt_base = __shfl_sync(0xffffffffu, gt_base, 0);
    eq_base = __shfl_sync(0xffffffffu, eq_base, 0);
    if (gt) {
      int pos = gt_base + __popc(gt_mask & ((1u << lane) - 1u));
      if (pos < greater_count) output[pos] = vals[i];
    }
    if (eq) {
      int pos = eq_base + __popc(eq_mask & ((1u << lane) - 1u));
      if (pos < equal_needed) output[greater_count + pos] = vals[i];
    }
  }
}

// Warp-cooperative top-down 256-bin radix scan. Reads scratch[0]=prefix and
// scratch[1]=residual; locates the bin holding the residual-th largest key and
// writes the updated prefix/residual UNCONDITIONALLY (the residual-th key is
// always in some group, so no stale value can survive -> no OOB downstream).
// Must be entered by the whole block; only the first warp does work. Callers
// bracket with __syncthreads() before (hist ready) and after (scratch stable).
__device__ __forceinline__ void warp_scan_256bin(const int* hist, int* scratch) {
  if (threadIdx.x < 32) {
    int lane = threadIdx.x;
    unsigned prefix = reinterpret_cast<unsigned*>(scratch)[0];
    int gs = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) gs += hist[lane * 8 + j];
    int incl = gs;  // inclusive suffix sum over the 32 group totals
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v = __shfl_down_sync(0xffffffffu, incl, d);
      if (lane + d < 32) incl += v;
    }
    int residual = scratch[1];
    unsigned ge = __ballot_sync(0xffffffffu, incl >= residual);
    int lstar = ge ? (31 - __clz(ge)) : 0;
    int above = __shfl_sync(0xffffffffu, incl - gs, lstar);  // bins above group
    if (lane == 0) {
      int res_in = residual - above;
      int found = lstar * 8;
      for (int b = lstar * 8 + 7; b >= lstar * 8; --b) {
        int cc = hist[b];
        if (res_in > cc) res_in -= cc;
        else { found = b; break; }
      }
      reinterpret_cast<unsigned*>(scratch)[0] = (prefix << 8) | static_cast<unsigned>(found);
      scratch[1] = res_in;
    }
  }
}

// Emit indices from an SMEM row-key cache (cache position == global index).
template <int MODE>
__device__ void emit_cached(const unsigned* keys, int n, unsigned boundary,
                            int* output, int k, SharedState* root) {
  int lane = threadIdx.x & 31;
  for (int base = (threadIdx.x & ~31); base < n; base += blockDim.x) {
    int i = base + lane;
    bool pred = i < n && (MODE == 0 ? keys[i] > boundary : keys[i] == boundary);
    unsigned mask = __ballot_sync(0xffffffffu, pred);
    int base_out = 0;
    if (lane == 0 && mask) base_out = atomicAdd(&root->output_count, __popc(mask));
    base_out = __shfl_sync(0xffffffffu, base_out, 0);
    if (pred) {
      int pos = base_out + __popc(mask & ((1u << lane) - 1u));
      if (pos < k) output[pos] = i;
    }
  }
}

__device__ void emit_cached_both(const unsigned* keys, int n,
                                 unsigned boundary, int greater_count,
                                 int* output, int k, SharedState* root) {
  int lane = threadIdx.x & 31;
  int equal_needed = k - greater_count;
  for (int base = (threadIdx.x & ~31); base < n; base += blockDim.x) {
    int i = base + lane;
    bool gt = i < n && keys[i] > boundary;
    bool eq = i < n && keys[i] == boundary;
    unsigned gt_mask = __ballot_sync(0xffffffffu, gt);
    unsigned eq_mask = __ballot_sync(0xffffffffu, eq);
    int gt_base = 0, eq_base = 0;
    if (lane == 0) {
      if (gt_mask) gt_base = atomicAdd(&root->output_count, __popc(gt_mask));
      if (eq_mask) eq_base = atomicAdd(&root->local_count, __popc(eq_mask));
    }
    gt_base = __shfl_sync(0xffffffffu, gt_base, 0);
    eq_base = __shfl_sync(0xffffffffu, eq_base, 0);
    if (gt) {
      int pos = gt_base + __popc(gt_mask & ((1u << lane) - 1u));
      if (pos < greater_count) output[pos] = i;
    }
    if (eq) {
      int pos = eq_base + __popc(eq_mask & ((1u << lane) - 1u));
      if (pos < equal_needed) output[greater_count + pos] = i;
    }
  }
}

__device__ void emit_cached_ge(const unsigned* keys, int n, unsigned thr,
                               int* output, int k, SharedState* root) {
  int lane = threadIdx.x & 31;
  for (int base = (threadIdx.x & ~31); base < n; base += blockDim.x) {
    int i = base + lane;
    bool pred = i < n && keys[i] >= thr;
    unsigned mask = __ballot_sync(0xffffffffu, pred);
    int base_out = 0;
    if (lane == 0 && mask) base_out = atomicAdd(&root->output_count, __popc(mask));
    base_out = __shfl_sync(0xffffffffu, base_out, 0);
    if (pred) {
      int pos = base_out + __popc(mask & ((1u << lane) - 1u));
      if (pos < k) output[pos] = i;
    }
  }
}

// Small-n specialization: cache the entire row in shared memory as ordered
// keys ONCE, then run the GVR skeleton (prior-min guess -> verify count ->
// exact radix refine) entirely against the SMEM cache. This eliminates the
// repeated full-row global re-reads of the secant loop and drops the separate
// candidate buffer (cache position == global index). Fully compliant GVR.
template <int THREADS, int MINB>
__global__ void __launch_bounds__(THREADS, MINB)
sb_cached_radix_kernel(const float* __restrict__ logits, const int* __restrict__ pre_idx,
                       int* __restrict__ output, int stride, int n, int k) {
  int row_id = static_cast<int>(blockIdx.x);
  const float* row = logits + static_cast<long long>(row_id) * stride;
  const int* hints = pre_idx + static_cast<long long>(row_id) * k;
  int* out = output + static_cast<long long>(row_id) * k;

  extern __shared__ __align__(16) unsigned char smem[];
  unsigned* keys = reinterpret_cast<unsigned*>(smem);
  int n_pad = (n + 3) & ~3;
  int* hist = reinterpret_cast<int*>(keys + n_pad);
  int* scratch = hist + RADIX;
  SharedState* st = reinterpret_cast<SharedState*>(scratch + 3 * MAX_WARPS);

  if (threadIdx.x == 0) st->output_count = 0;

  // GVR guess first, so verification can be fused into the one-time row load.
  float pmin = block_hint_min(row, hints, n, k, scratch);
  unsigned thr = ordered_key(pmin);
  int c = 0;

  // Load full row into SMEM as monotone ordered keys (coalesced float4), while
  // simultaneously counting keys that clear the prior threshold.
  for (int base = threadIdx.x * 4; base < n_pad; base += blockDim.x * 4) {
    float4 v;
    if (base + 3 < n) {
      v = *reinterpret_cast<const float4*>(row + base);
    } else {
      v.x = base     < n ? row[base]     : -FLT_MAX;
      v.y = base + 1 < n ? row[base + 1] : -FLT_MAX;
      v.z = base + 2 < n ? row[base + 2] : -FLT_MAX;
      v.w = base + 3 < n ? row[base + 3] : -FLT_MAX;
    }
    unsigned k0 = ordered_key(v.x), k1 = ordered_key(v.y);
    unsigned k2 = ordered_key(v.z), k3 = ordered_key(v.w);
    keys[base] = k0; keys[base + 1] = k1;
    keys[base + 2] = k2; keys[base + 3] = k3;
    c += (base < n && k0 >= thr) + (base + 1 < n && k1 >= thr) +
         (base + 2 < n && k2 >= thr) + (base + 3 < n && k3 >= thr);
  }
  __syncthreads();

  // Verify the fused per-thread admission counts.
  c = warp_sum(c);
  {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) scratch[warp] = c;
    __syncthreads();
    if (warp == 0) {
      int nw = blockDim.x >> 5;
      int bc = lane < nw ? scratch[lane] : 0;
      bc = warp_sum(bc);
      if (lane == 0) scratch[0] = bc;
    }
    __syncthreads();
  }
  int cnt = scratch[0];

  // Prior-admitted fast path: exactly k keys clear the prior threshold, so
  // they ARE the exact top-k set (k largest) regardless of ties.
  if (cnt == k) {
    emit_cached_ge(keys, n, thr, out, k, st);
    return;
  }

  // Refine: exact 4x8-bit radix select for the k-th largest key over the cache.
  if (threadIdx.x == 0) { reinterpret_cast<unsigned*>(scratch)[0] = 0; scratch[1] = k; }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < RADIX; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    unsigned prefix = reinterpret_cast<unsigned*>(scratch)[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      unsigned key = keys[i];
      bool active = shift == 24 || (key >> (shift + 8)) == prefix;
      if (active) atomicAdd(hist + ((key >> shift) & 255u), 1);
    }
    __syncthreads();
    warp_scan_256bin(hist, scratch);
    __syncthreads();
  }
  unsigned boundary = reinterpret_cast<unsigned*>(scratch)[0];
  int greater_count = k - scratch[1];
  if (threadIdx.x == 0) { st->output_count = 0; st->local_count = 0; }
  __syncthreads();
  emit_cached_both(keys, n, boundary, greater_count, out, k, st);
}

// Single-block-per-row GVR (cs==1) specialization. Templated on THREADS and
// MINB (launch_bounds occupancy) with a RUNTIME candidate-buffer cap so
// small-n / high-batch cells allocate only what they need and pack more CTAs
// per SM. Correctness identical to the cs==1 path of gvr_topk_kernel.
template <int THREADS, int MINB, int DO_FUSE = 1>
__global__ void __launch_bounds__(THREADS, MINB)
sb_gvr_topk_kernel(const float* __restrict__ logits, const int* __restrict__ pre_idx,
                   int* __restrict__ output, int stride, int n, int k, int cap_c) {
  int row_id = static_cast<int>(blockIdx.x);
  const float* row = logits + static_cast<long long>(row_id) * stride;
  const int* hints = pre_idx + static_cast<long long>(row_id) * k;
  int* out = output + static_cast<long long>(row_id) * k;
  int begin = 0;
  int end = n;

  extern __shared__ __align__(16) unsigned char smem[];
  unsigned* cand_keys = reinterpret_cast<unsigned*>(smem);
  int* cand_vals = reinterpret_cast<int*>(cand_keys + cap_c);
  int* hist = cand_vals + cap_c;
  int* scratch = hist + RADIX;
  SharedState* st = reinterpret_cast<SharedState*>(scratch + 3 * MAX_WARPS);

  if (threadIdx.x == 0) {
    st->local_count = 0;
    st->output_count = 0;
    st->done = 0;
    st->iteration = 0;
  }
  __syncthreads();

  // R0: prior histogram -> two quantile rungs -> both GE counts in one scan.
  {
    float pmin, pmax, rung0, rung1;
    block_hint_rungs(row, hints, n, k, cand_keys, hist, scratch,
                     pmin, pmax, rung0, rung1);
    if (threadIdx.x == 0) {
      st->lo = pmin;
      st->hi = pmax;
      st->local_min = rung0;
      st->local_max = rung1;
    }
  }
  __syncthreads();
  int r0c0, r0c1;
  block_scan_two(row, begin, end, st->local_min, st->local_max, scratch, r0c0, r0c1);
  if (threadIdx.x == 0) {
    int c0 = r0c0, c1 = r0c1;
    float t0 = st->local_min, t1 = st->local_max;
    if (c1 >= k && c1 <= cap_c) {
      st->threshold = t1;
      st->total_count = c1;
      st->done = 1;
    } else if (c0 >= k && c0 <= cap_c) {
      st->threshold = t0;
      st->total_count = c0;
      st->done = 1;
    } else if (c1 > cap_c) {
      st->lo = t1;
      st->lo_count = c1;
      st->hi_count = -1;
      st->threshold = st->hi;
    } else if (c0 < k) {
      st->hi = t0;
      st->hi_count = c0;
      st->lo_count = -1;
      st->threshold = st->lo;
    } else {
      st->lo = t0;
      st->lo_count = c0;
      st->hi = t1;
      st->hi_count = c1;
      float lc = logf(static_cast<float>(c0));
      float hc = logf(static_cast<float>(c1 > 0 ? c1 : 1));
      float target = static_cast<float>(k) * powf(static_cast<float>(cap_c) / k, 0.20f);
      float a = (lc - logf(target)) / fmaxf(lc - hc, 1.0e-7f);
      a = fminf(0.95f, fmaxf(0.05f, a));
      st->threshold = fmaf(t1 - t0, a, t0);
    }
  }
  __syncthreads();

  // Fused refine: collect_local both COUNTS (>= threshold) and gathers the
  // candidates in one scan. When a pass accepts (total in [k,cap_c]) the buffer
  // already holds the top-k candidates at the accepted threshold, so the
  // separate post-loop collect scan is skipped (one fewer full-row read).
  bool fused_collected = false;
  for (int pass = 0; pass < 64 && st->done == 0; ++pass) {
    float threshold = st->threshold;
    // DO_FUSE: gather+count every pass so acceptance skips the post-loop scan
    // (wins on low-batch where 1 block/SM makes the extra gather ~free). For
    // high-batch (occupancy-bound) the per-pass gather is wasted on rejected
    // passes -> use count-only block_scan_count and collect once post-loop.
    if (DO_FUSE) {
      collect_local(row, begin, end, threshold, cand_keys, cand_vals, st, cap_c);
    } else {
      int lc; block_scan_count<THREADS>(row, begin, end, threshold, scratch, lc);
      if (threadIdx.x == 0) st->local_count = lc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int total = st->local_count;
      st->total_count = total;
      float t = threshold;
      if (total >= k && total <= cap_c) {
        st->done = 1;
      } else {
        if (total < k) {
          st->hi = t;
          st->hi_count = total > 0 ? total : 1;
        } else {
          st->lo = t;
          st->lo_count = total;
        }
        unsigned lo_key = ordered_key(st->lo);
        unsigned hi_key = ordered_key(st->hi);
        if (hi_key <= lo_key + 1u) {
          st->threshold = st->lo;
          st->done = 2;
        } else {
          if (st->lo_count < 0) {
            st->threshold = st->lo;
          } else if (st->hi_count < 0) {
            st->threshold = st->hi;
          } else {
            float lc = logf(static_cast<float>(st->lo_count > 0 ? st->lo_count : 1));
            float hc = logf(static_cast<float>(st->hi_count > 0 ? st->hi_count : 1));
            float target = static_cast<float>(k) * powf(static_cast<float>(cap_c) / k, 0.20f);
            float a = (lc - logf(target)) / fmaxf(lc - hc, 1.0e-7f);
            a = fminf(0.90f, fmaxf(0.10f, a));
            float next = fmaf(st->hi - st->lo, a, st->lo);
            unsigned nk = ordered_key(next);
            if (pass >= 31 || nk <= lo_key || nk >= hi_key)
              nk = lo_key + ((hi_key - lo_key) >> 1);
            st->threshold = ordered_float(nk);
          }
        }
      }
      st->iteration = pass + 1;
    }
    __syncthreads();
    if (st->done == 1) { fused_collected = DO_FUSE; break; }
    if (st->done != 0) break;
  }

  if (st->done == 2) {
    if (threadIdx.x == 0) st->output_count = 0;
    __syncthreads();
    emit_global<0>(row, begin, end, st->threshold, out, k, st);
    __syncthreads();
    if (threadIdx.x == 0 && st->output_count > k) st->output_count = k;
    __syncthreads();
    emit_global<1>(row, begin, end, st->threshold, out, k, st);
    __syncthreads();
    return;
  }

  if (!fused_collected) {
    collect_local(row, begin, end, st->threshold, cand_keys, cand_vals, st, cap_c);
    __syncthreads();
  }
  int merged_count = st->local_count;
  unsigned* merged_keys = cand_keys;
  int* merged_vals = cand_vals;

  // Fast path: secant landed exactly on k candidates -> every candidate is in
  // the top-k, no k-th boundary search needed.
  if (merged_count == k) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) out[i] = merged_vals[i];
    return;
  }

  if (threadIdx.x == 0) {
    reinterpret_cast<unsigned*>(scratch)[0] = 0;
    scratch[1] = k;
  }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < RADIX; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    unsigned prefix = reinterpret_cast<unsigned*>(scratch)[0];
    for (int i = threadIdx.x; i < merged_count; i += blockDim.x) {
      unsigned key = merged_keys[i];
      bool active = shift == 24 || (key >> (shift + 8)) == prefix;
      if (active) atomicAdd(hist + ((key >> shift) & 255u), 1);
    }
    __syncthreads();
    warp_scan_256bin(hist, scratch);
    __syncthreads();
  }
  unsigned boundary_key = reinterpret_cast<unsigned*>(scratch)[0];
  int greater_count = k - scratch[1];
  if (threadIdx.x == 0) { st->output_count = 0; st->local_count = 0; }
  __syncthreads();
  emit_candidates_both(merged_keys, merged_vals, merged_count, boundary_key,
                       greater_count, out, k, st);
  __syncthreads();
}

template <int TC>
__global__ void __launch_bounds__(1024, 1)
gvr_topk_kernel(const float* __restrict__ logits, const int* __restrict__ pre_idx,
                int* __restrict__ output, int stride, int n, int k, int cluster_size) {
  cg::cluster_group cluster = cg::this_cluster();
  int rank = static_cast<int>(cluster.block_rank());
  int row_id = static_cast<int>(blockIdx.x) / cluster_size;
  const float* row = logits + static_cast<long long>(row_id) * stride;
  const int* hints = pre_idx + static_cast<long long>(row_id) * k;
  int* out = output + static_cast<long long>(row_id) * k;
  int vec_chunks = (n + 3) >> 2;
  int begin = 4 * static_cast<int>((static_cast<long long>(vec_chunks) * rank) / cluster_size);
  int end = min(n, 4 * static_cast<int>(
      (static_cast<long long>(vec_chunks) * (rank + 1)) / cluster_size));

  extern __shared__ __align__(16) unsigned char smem[];
  unsigned* cand_keys = reinterpret_cast<unsigned*>(smem);
  int* cand_vals = reinterpret_cast<int*>(cand_keys + CAP);
  int* hist = cand_vals + CAP;
  int* scratch = hist + RADIX;
  SharedState* st = reinterpret_cast<SharedState*>(scratch + 3 * MAX_WARPS);
  SharedState* root = cluster.map_shared_rank(st, 0);
  unsigned* merged_keys = reinterpret_cast<unsigned*>(st + 1);
  int* merged_vals = reinterpret_cast<int*>(merged_keys + CAP);

  if (threadIdx.x == 0) {
    st->local_count = 0;
    st->output_count = 0;
    st->done = 0;
    st->iteration = 0;
  }
  __syncthreads();

  // R0: prior histogram -> two quantile rungs -> both GE counts in one scan.
  if (rank == 0) {
    float pmin, pmax, rung0, rung1;
    block_hint_rungs(row, hints, n, k, cand_keys, hist, scratch,
                     pmin, pmax, rung0, rung1);
    if (threadIdx.x == 0) {
      root->lo = pmin;
      root->hi = pmax;
      root->local_min = rung0;
      root->local_max = rung1;
    }
  }
  cluster_or_block_sync(cluster, cluster_size);
  int r0c0, r0c1;
  block_scan_two(row, begin, end, root->local_min, root->local_max,
                 scratch, r0c0, r0c1);
  if (threadIdx.x == 0) {
    st->local_count = r0c0;
    st->total_count = r0c1;
  }
  cluster_or_block_sync(cluster, cluster_size);
  if (rank == 0 && threadIdx.x == 0) {
    int c0 = 0, c1 = 0;
    for (int r = 0; r < cluster_size; ++r) {
      SharedState* peer = cluster.map_shared_rank(st, r);
      c0 += peer->local_count;
      c1 += peer->total_count;
    }
    float t0 = root->local_min, t1 = root->local_max;
    if (c1 >= k && c1 <= CAP) {
      root->threshold = t1;
      root->total_count = c1;
      root->done = 1;
    } else if (c0 >= k && c0 <= CAP) {
      root->threshold = t0;
      root->total_count = c0;
      root->done = 1;
    } else if (c1 > CAP) {
      // Both rungs overshot. Measure the prior maximum as the high end.
      root->lo = t1;
      root->lo_count = c1;
      root->hi_count = -1;
      root->threshold = root->hi;
    } else if (c0 < k) {
      // Both rungs undershot. Measure the prior minimum as the low end.
      root->hi = t0;
      root->hi_count = c0;
      root->lo_count = -1;
      root->threshold = root->lo;
    } else {
      // The rungs bracket the acceptance window; take one bounded log-falsi shot.
      root->lo = t0;
      root->lo_count = c0;
      root->hi = t1;
      root->hi_count = c1;
      float lc = logf(static_cast<float>(c0));
      float hc = logf(static_cast<float>(c1 > 0 ? c1 : 1));
      float target = static_cast<float>(k) * powf(static_cast<float>(CAP) / k, 0.20f);
      float a = (lc - logf(target)) / fmaxf(lc - hc, 1.0e-7f);
      a = fminf(0.95f, fmaxf(0.05f, a));
      root->threshold = fmaf(t1 - t0, a, t0);
    }
  }
  cluster_or_block_sync(cluster, cluster_size);

  for (int pass = 0; pass < 64 && root->done == 0; ++pass) {
    float threshold = root->threshold;
    int local_count;
    block_scan_count<TC>(row, begin, end, threshold, scratch, local_count);
    if (threadIdx.x == 0) {
      st->local_count = local_count;
    }
    cluster_or_block_sync(cluster, cluster_size);

    if (rank == 0 && threadIdx.x == 0) {
      int total = 0;
      for (int r = 0; r < cluster_size; ++r) {
        SharedState* peer = cluster.map_shared_rank(st, r);
        total += peer->local_count;
      }
      root->total_count = total;
      float t = root->threshold;
      if (total >= k && total <= CAP) {
        root->done = 1;
      } else {
        if (total < k) {
          root->hi = t;
          root->hi_count = total > 0 ? total : 1;
        } else {
          root->lo = t;
          root->lo_count = total;
        }
        unsigned lo_key = ordered_key(root->lo);
        unsigned hi_key = ordered_key(root->hi);
        if (hi_key <= lo_key + 1u) {
          root->threshold = root->lo;
          root->done = 2;
        } else {
          if (root->lo_count < 0) {
            root->threshold = root->lo;
          } else if (root->hi_count < 0) {
            root->threshold = root->hi;
          } else {
            float lc = logf(static_cast<float>(root->lo_count > 0 ? root->lo_count : 1));
            float hc = logf(static_cast<float>(root->hi_count > 0 ? root->hi_count : 1));
            float target = static_cast<float>(k) * powf(static_cast<float>(CAP) / k, 0.20f);
            float a = (lc - logf(target)) / fmaxf(lc - hc, 1.0e-7f);
            a = fminf(0.90f, fmaxf(0.10f, a));
            float next = fmaf(root->hi - root->lo, a, root->lo);
            unsigned nk = ordered_key(next);
            if (pass >= 31 || nk <= lo_key || nk >= hi_key)
              nk = lo_key + ((hi_key - lo_key) >> 1);
            root->threshold = ordered_float(nk);
          }
        }
      }
      root->iteration = pass + 1;
    }
    cluster_or_block_sync(cluster, cluster_size);
    if (root->done != 0) break;
  }

  if (root->done == 2) {
    if (rank == 0 && threadIdx.x == 0) root->output_count = 0;
    cluster_or_block_sync(cluster, cluster_size);
    emit_global<0>(row, begin, end, root->threshold, out, k, root);
    cluster_or_block_sync(cluster, cluster_size);
    if (rank == 0 && threadIdx.x == 0 && root->output_count > k) root->output_count = k;
    cluster_or_block_sync(cluster, cluster_size);
    emit_global<1>(row, begin, end, root->threshold, out, k, root);
    cluster_or_block_sync(cluster, cluster_size);
    return;
  }

  collect_local(row, begin, end, root->threshold, cand_keys, cand_vals, st);
  cluster_or_block_sync(cluster, cluster_size);

  int merged_count;
  if (cluster_size > 1) {
    unsigned* root_merged_keys = cluster.map_shared_rank(merged_keys, 0);
    int* root_merged_vals = cluster.map_shared_rank(merged_vals, 0);
    if (rank == 0 && threadIdx.x == 0) root->output_count = 0;
    cluster_or_block_sync(cluster, cluster_size);
    int lane = threadIdx.x & 31;
    int count = st->local_count;
    for (int base = (threadIdx.x & ~31); base < count; base += blockDim.x) {
      int i = base + lane;
      unsigned mask = __ballot_sync(0xffffffffu, i < count);
      int dst = 0;
      if (lane == 0 && mask) dst = atomicAdd(&root->output_count, __popc(mask));
      dst = __shfl_sync(0xffffffffu, dst, 0);
      if (i < count) {
        int pos = dst + __popc(mask & ((1u << lane) - 1u));
        root_merged_keys[pos] = cand_keys[i];
        root_merged_vals[pos] = cand_vals[i];
      }
    }
    cluster_or_block_sync(cluster, cluster_size);
    merged_count = root->output_count;
  } else {
    merged_keys = cand_keys;
    merged_vals = cand_vals;
    merged_count = st->local_count;
  }

  if (rank == 0) {
    if (threadIdx.x == 0) {
      reinterpret_cast<unsigned*>(scratch)[0] = 0;
      scratch[1] = k;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = threadIdx.x; i < RADIX; i += blockDim.x) hist[i] = 0;
      __syncthreads();
      unsigned prefix = reinterpret_cast<unsigned*>(scratch)[0];
      for (int i = threadIdx.x; i < merged_count; i += blockDim.x) {
        unsigned key = merged_keys[i];
        bool active = shift == 24 || (key >> (shift + 8)) == prefix;
        if (active) atomicAdd(hist + ((key >> shift) & 255u), 1);
      }
      __syncthreads();
      warp_scan_256bin(hist, scratch);
      __syncthreads();
    }
    unsigned boundary_key = reinterpret_cast<unsigned*>(scratch)[0];
    int greater_count = k - scratch[1];
    if (threadIdx.x == 0) { root->output_count = 0; root->local_count = 0; }
    __syncthreads();
    emit_candidates_both(merged_keys, merged_vals, merged_count, boundary_key,
                         greater_count, out, k, root);
    __syncthreads();
  }
  cluster_or_block_sync(cluster, cluster_size);
}

}  // namespace

// One-time max-dynamic-SMEM registration. cudaFuncSetAttribute persists for the
// lifetime of the kernel function, so calling it on every launch adds a host-side
// driver round-trip to the timed path for nothing. The linear fit of our latency
// vs the reference (mine = 7.4us + 0.78*base) shows a fixed per-call overhead that
// dominates the small-n cells; hoisting these driver calls out of the hot path
// trims that fixed cost. We register each variant once at its maximum possible
// dynamic-SMEM footprint (actual launches always request <= this).
static void init_func_attrs() {
  const int cached_max = (int)(8192 * sizeof(unsigned) + RADIX * sizeof(int) +
                               3 * MAX_WARPS * sizeof(int) + sizeof(SharedState));
  const int sb_max = (int)(CAP * (sizeof(unsigned) + sizeof(int)) + RADIX * sizeof(int) +
                           3 * MAX_WARPS * sizeof(int) + sizeof(SharedState));
  const int cl_max = (int)(2 * CAP * (sizeof(unsigned) + sizeof(int)) + RADIX * sizeof(int) +
                           3 * MAX_WARPS * sizeof(int) + sizeof(SharedState));
  cudaFuncSetAttribute(sb_cached_radix_kernel<256, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, cached_max);
  cudaFuncSetAttribute(sb_cached_radix_kernel<512, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, cached_max);
  cudaFuncSetAttribute(sb_gvr_topk_kernel<1024, 1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sb_max);
  cudaFuncSetAttribute(sb_gvr_topk_kernel<1024, 1, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, sb_max);
  cudaFuncSetAttribute(sb_gvr_topk_kernel<512, 2, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, sb_max);
  cudaFuncSetAttribute(gvr_topk_kernel<1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, cl_max);
  cudaFuncSetAttribute(gvr_topk_kernel<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, cl_max);
}

void gvr_topk_launch(const float* logits, const int* pre_idx, int* indices,
                     int batch, int stride, int n, int k, cudaStream_t stream) {
  static bool attrs_ready = (init_func_attrs(), true);
  (void)attrs_ready;
  constexpr int sms = 148;
  if (n - k <= 32) {
    // Dense (n~=k) path. NCU on pro_4k_L12 (n=1027,k=1024,b=1, removal branch):
    // 14.4us, SM 0.07% busy, occupancy 12% -- a single latency-bound block doing
    // remove_count sequential block-reductions. Same profile as the small-n
    // secant cells: at low batch there is one block/row so occupancy is
    // irrelevant and more threads shorten each scan's serial chain. High batch
    // (b>sms, e.g. pro_4k_L14_bs1024 which we already WIN at 0.65x) keeps 256 so
    // many CTAs pack per SM.
    if (batch <= sms) {
      dense_gvr_topk_kernel<1024><<<batch, 1024, 0, stream>>>(
          logits, pre_idx, indices, stride, n, k);
    } else {
      dense_gvr_topk_kernel<256><<<batch, 256, 0, stream>>>(
          logits, pre_idx, indices, stride, n, k);
    }
    return;
  }
  int cs = 1;
#ifndef FORCE_CS1
  if (n >= 131072 && batch <= 4) cs = 8;
  else if (n >= 65536 && batch * 4 <= sms) cs = 4;
  else if (n >= 65536 && batch * 2 <= sms) cs = 2;
#endif
  int per_cta = (n + cs - 1) / cs;

  if (cs == 1) {
    bool high_batch = batch > sms;  // oversubscribes the device
    // Small-n: cache the entire row in SMEM once and run exact radix-select
    // directly against the cache (no repeated global re-reads). n_pad keys +
    // radix hist + reduction scratch + state. Threshold chosen so smem stays
    // small enough to pack multiple CTAs/SM in the oversubscribed regime.
    // (Empirically n>8192 regresses: 4-pass SMEM radix at 1 CTA/SM loses to the
    // fast-converging secant with L2-cached global reads.)
    if (n <= 8192) {
      int n_pad = (n + 3) & ~3;
      size_t smem = static_cast<size_t>(n_pad) * sizeof(unsigned) +
                    RADIX * sizeof(int) + 3 * MAX_WARPS * sizeof(int) + sizeof(SharedState);
      if (high_batch) {
        sb_cached_radix_kernel<256, 8><<<batch, 256, smem, stream>>>(
            logits, pre_idx, indices, stride, n, k);
      } else {
        sb_cached_radix_kernel<512, 2><<<batch, 512, smem, stream>>>(
            logits, pre_idx, indices, stride, n, k);
      }
      return;
    }
    // Runtime candidate cap: never exceed n (rounded to a float4).
    int cap_c = CAP;
    int n4 = ((n + 3) & ~3) + 4;
    if (n4 < cap_c) cap_c = n4;
    if (n < 65536 && k < 2048) {
      int tuned_cap = (k == 512) ? 4 * k : 2 * k;
      if (tuned_cap < cap_c) cap_c = tuned_cap;
    }
    size_t smem = static_cast<size_t>(cap_c) * (sizeof(unsigned) + sizeof(int)) +
                  RADIX * sizeof(int) + 3 * MAX_WARPS * sizeof(int) + sizeof(SharedState);
    // Threads / occupancy by regime. Low batch => minimize latency with more
    // threads (occupancy irrelevant at 1 block). High batch small-n => pack
    // many CTAs/SM with fewer threads.
    // DO_FUSE only for larger n: the fused gather-per-pass saves a post-loop
    // full-row rescan that is a real cost at n>=16384, but at small n the many
    // secant passes each pay the gather overhead and it regresses. The thread
    // split already routes n>=16384 (batch<=sms) to the 1024-thread branch, so
    // fuse there only; the 512-branch (small-n or high-batch) stays count-only.
    if (per_cta >= 16384 && batch <= sms) {
      sb_gvr_topk_kernel<1024, 1, 1><<<batch, 1024, smem, stream>>>(
          logits, pre_idx, indices, stride, n, k, cap_c);
    } else if (batch <= sms) {
      // Low-batch small-n (8192<n<16384, b=1): NCU shows this cell is a single
      // block, SM 0.15% busy, pure serial dependency-latency (row is L2-cached).
      // With one resident block, occupancy is irrelevant, so more threads halve
      // each scan's per-thread trip count on the latency-bound secant loop.
      // DO_FUSE stays 0 (fused gather regresses at small n).
      sb_gvr_topk_kernel<1024, 1, 0><<<batch, 1024, smem, stream>>>(
          logits, pre_idx, indices, stride, n, k, cap_c);
    } else {
      sb_gvr_topk_kernel<512, 2, 0><<<batch, 512, smem, stream>>>(
          logits, pre_idx, indices, stride, n, k, cap_c);
    }
    return;
  }

  int threads = (per_cta >= 16384 && batch <= sms) ? 1024 : 512;
  size_t smem = CAP * (sizeof(unsigned) + sizeof(int)) +
                RADIX * sizeof(int) + 3 * MAX_WARPS * sizeof(int) + sizeof(SharedState);
  smem += CAP * (sizeof(unsigned) + sizeof(int));
  auto kern = (threads == 1024) ? gvr_topk_kernel<1024> : gvr_topk_kernel<512>;

  {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(batch * cs, 1, 1);
    config.blockDim = dim3(threads, 1, 1);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = cs;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = &attr;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kern,
                       logits, pre_idx, indices, stride, n, k, cs);
  }
}
