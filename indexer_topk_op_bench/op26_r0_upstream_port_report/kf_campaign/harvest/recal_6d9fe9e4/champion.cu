#include "launchers.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace {

constexpr int THREADS = 512;
constexpr int CONTROL_INTS = 32;

__device__ __forceinline__ float warp_min(float x) {
  for (int d = 16; d; d >>= 1) x = fminf(x, __shfl_down_sync(0xffffffff, x, d));
  return x;
}

__device__ __forceinline__ float warp_max(float x) {
  for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
  return x;
}

__device__ __forceinline__ int warp_sum(int x) {
  for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
  return x;
}

__device__ __forceinline__ uint32_t ordered_key(float x) {
  uint32_t u = __float_as_uint(x);
  return u ^ ((static_cast<int32_t>(u) >> 31) | 0x80000000u);
}

__global__ __launch_bounds__(1024, 1)
void exclude_three_min_kernel(
    const float* __restrict__ logits,
    int* __restrict__ output,
    int n) {
  __shared__ uint32_t warp_key[32];
  __shared__ int warp_idx[32];
  __shared__ int excluded[3];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
#pragma unroll
  for (int round = 0; round < 3; ++round) {
    uint32_t key = 0xffffffffu;
    int idx = -1;
    for (int i = tid; i < n; i += blockDim.x) {
      bool skip = false;
#pragma unroll
      for (int r = 0; r < round; ++r) skip |= i == excluded[r];
      uint32_t x = ordered_key(logits[i]);
      if (!skip && x < key) { key = x; idx = i; }
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) {
      uint32_t ok = __shfl_down_sync(0xffffffff, key, d);
      int oi = __shfl_down_sync(0xffffffff, idx, d);
      if (ok < key) { key = ok; idx = oi; }
    }
    if (lane == 0) { warp_key[warp] = key; warp_idx[warp] = idx; }
    __syncthreads();
    if (warp == 0) {
      key = warp_key[lane];
      idx = warp_idx[lane];
#pragma unroll
      for (int d = 16; d; d >>= 1) {
        uint32_t ok = __shfl_down_sync(0xffffffff, key, d);
        int oi = __shfl_down_sync(0xffffffff, idx, d);
        if (ok < key) { key = ok; idx = oi; }
      }
      if (lane == 0) excluded[round] = idx;
    }
    __syncthreads();
  }
  for (int i = tid; i < n; i += blockDim.x) {
    if (i != excluded[0] && i != excluded[1] && i != excluded[2]) {
      int slot = i - (excluded[0] < i) - (excluded[1] < i) - (excluded[2] < i);
      output[slot] = i;
    }
  }
}

// k-th-bin selection over a 256-bin histogram, computed by a single warp with
// __shfl (no barriers) instead of an 8-step Hillis-Steele suffix scan (16
// barriers). Lane L owns the 8 bins [8L,8L+7]; a warp suffix-scan of the
// per-lane totals locates the lane containing the crossing bin, which then
// finds the exact bin locally. Output-identical to the previous scan version.
// s_scan is unused (kept for signature compatibility).
__device__ __forceinline__ void pick_kth_bin(
    int* hist, int* s_scan, uint32_t& prefix, int& kth, int shift,
    uint32_t* s_prefix, int* s_kth, int tid, int blockThreads) {
  (void)s_scan;
  (void)blockThreads;
  if (tid < 32) {
    const int lane = tid;
    int laneTotal = 0;
#pragma unroll
    for (int r = 0; r < 8; ++r) laneTotal += hist[lane * 8 + r];
    // inclusive suffix sum over lanes: inc[L] = sum_{m>=L} laneTotal[m].
    int inc = laneTotal;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int t = __shfl_down_sync(0xffffffffu, inc, d);
      if (lane + d < 32) inc += t;
    }
    int aboveLane = inc - laneTotal;  // = sum_{m>L} laneTotal[m]
    if (aboveLane < kth && aboveLane + laneTotal >= kth) {
      int prevcum = 0;  // sum of bins strictly above current within this lane
#pragma unroll
      for (int r = 7; r >= 0; --r) {
        int b = lane * 8 + r;
        int hb = hist[b];
        int aboveThis = aboveLane + prevcum;  // = sum_{j>b} hist[j]
        int here = aboveThis + hb;            // = sum_{j>=b} hist[j]
        if (here >= kth && aboveThis < kth) {
          *s_prefix = prefix | (static_cast<uint32_t>(b) << shift);
          *s_kth = kth - aboveThis;
        }
        prevcum += hb;
      }
    }
  }
  __syncthreads();
  prefix = *s_prefix;
  kth = *s_kth;
}

__global__ __launch_bounds__(1024, 1)
void direct_radix_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ pre_idx,
    int* __restrict__ output,
    int n,
    int k) {
  (void)pre_idx;
  __shared__ int hist[256];
  __shared__ int s_scan[256];
  __shared__ uint32_t s_prefix;
  __shared__ int s_kth;
  __shared__ int counters[2];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  uint32_t prefix = 0;
  uint32_t mask_hi = 0;
  int kth = k;
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    int shift = 24 - 8 * pass;
    for (int i = tid; i < n; i += blockDim.x) {
      uint32_t key = ordered_key(logits[i]);
      if ((key & mask_hi) == prefix) atomicAdd(hist + ((key >> shift) & 255), 1);
    }
    __syncthreads();
    pick_kth_bin(hist, s_scan, prefix, kth, shift, &s_prefix, &s_kth, tid, blockDim.x);
    mask_hi |= 255u << shift;
  }
  if (tid == 0) { counters[0] = 0; counters[1] = 0; }
  __syncthreads();
  // Fused warp-aggregated compaction: strict winners at [0, strict), ties at
  // [strict, k). strict = k - kth is known exactly from the radix select, so
  // both emits run in ONE scan (no intermediate barrier / counter read).
  const int strict = k - kth;
  const int nround = (n + (int)blockDim.x - 1) / (int)blockDim.x * (int)blockDim.x;
  for (int i = tid; i < nround; i += blockDim.x) {
    int valid = (i < n);
    uint32_t key = valid ? ordered_key(logits[i]) : 0u;
    int isA = valid && (key > prefix);
    int isB = valid && (key == prefix);
    unsigned mA = __ballot_sync(0xffffffffu, isA);
    if (mA) {
      int rank = __popc(mA & ((1u << lane) - 1u));
      int leader = __ffs(mA) - 1;
      int base = 0;
      if (lane == leader) base = atomicAdd(counters, __popc(mA));
      base = __shfl_sync(0xffffffffu, base, leader);
      if (isA) output[base + rank] = i;
    }
    unsigned mB = __ballot_sync(0xffffffffu, isB);
    if (mB) {
      int rank = __popc(mB & ((1u << lane) - 1u));
      int leader = __ffs(mB) - 1;
      int base = 0;
      if (lane == leader) base = atomicAdd(counters + 1, __popc(mB));
      base = __shfl_sync(0xffffffffu, base, leader);
      if (isB && strict + base + rank < k) output[strict + base + rank] = i;
    }
  }
}

template <int CAP, int CS, int NT, int NPROBE>
__global__ __launch_bounds__(NT, 1)
void gvr_topk_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ pre_idx,
    int* __restrict__ output,
    int n,
    int k) {
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int NWARPS = NT / 32;

  extern __shared__ __align__(16) unsigned char storage[];
  int* ctrl = reinterpret_cast<int*>(storage);
  float* cand_val = reinterpret_cast<float*>(ctrl + CONTROL_INTS);
  int* cand_idx = reinterpret_cast<int*>(cand_val + CAP);
  int* hist = cand_idx + CAP;
  int* scan = hist + 256;
  float* warp_f = reinterpret_cast<float*>(scan + 256);
  int* warp_i = reinterpret_cast<int*>(warp_f + NWARPS);

  int* root_ctrl = cluster.map_shared_rank(ctrl, 0);

  // Phase 1: the temporal hint supplies a value bracket. Every CTA computes
  // it redundantly; this is only 512--2048 gathers and avoids DSMEM traffic.
  float hmin = CUDART_INF_F;
  float hmax = -CUDART_INF_F;
  for (int j = tid; j < k; j += NT) {
    float v = logits[pre_idx[j]];
    hmin = fminf(hmin, v);
    hmax = fmaxf(hmax, v);
  }
  hmin = warp_min(hmin);
  hmax = warp_max(hmax);
  // Stage both reductions together: min in warp_f[warp], max in warp_f[NWARPS+warp].
  if (lane == 0) { warp_f[warp] = hmin; warp_f[NWARPS + warp] = hmax; }
  __syncthreads();
  if (warp == 0) {
    float vmin = lane < NWARPS ? warp_f[lane] : CUDART_INF_F;
    float vmax = lane < NWARPS ? warp_f[NWARPS + lane] : -CUDART_INF_F;
    vmin = warp_min(vmin);
    vmax = warp_max(vmax);
    if (lane == 0) { ctrl[0] = __float_as_int(vmin); ctrl[1] = __float_as_int(vmax); }
  }
  __syncthreads();

  const int begin = (static_cast<long long>(n) * rank) / CS;
  const int end = (static_cast<long long>(n) * (rank + 1)) / CS;
  float lo = __int_as_float(ctrl[0]);
  float hi = __int_as_float(ctrl[1]);
  float threshold = lo;
  int total = 0;

  // Phase 2: multi-threshold single-pass probing. Each pass counts NPROBE
  // interior thresholds spanning (lo, hi] in ONE scan of the slice, then
  // shrinks the bracket to the sub-interval [t_j, t_{j+1}) that brackets the
  // target count k. This contracts the bracket ~(NPROBE+1)x per pass instead
  // of 2x, so far fewer barrier-separated passes are needed. Counts use a
  // ">=" convention (count of values >= t_j is monotdecreasing in j). One
  // cluster.sync per pass via the redundant-reduce trick. Bounds: lo has
  // count>=k, hi has count<k (from the hint bracket; refined each pass).
  int found = 0;
  // We want the smallest threshold t with count(>=t) in [k, CAP]. Track a
  // bracket [lo, hi): count(>=lo) >= k always; count(>=hi) < k (or hi=+inf).
  // Emit `threshold` = a value whose count is in [k, CAP].
#pragma unroll 1
  for (int pass = 0; pass < 6 && !found; ++pass) {
    // NPROBE interior probe thresholds strictly inside (lo, hi).
    float step = (hi - lo) / (float)(NPROBE + 1);
    int cnt[NPROBE];
#pragma unroll
    for (int p = 0; p < NPROBE; ++p) cnt[p] = 0;
    float tv[NPROBE];
#pragma unroll
    for (int p = 0; p < NPROBE; ++p) tv[p] = lo + step * (float)(p + 1);
    // Vectorized float4 body over chunks fully inside [begin,end); scalar
    // head/tail cover the unaligned edges. logits is 64B-aligned so float4
    // loads are safe. 4-wide loads raise memory-level parallelism on the
    // latency-bound scan.
    const int c0 = (begin + 3) >> 2;   // first float4 chunk start >= begin
    const int c1 = end >> 2;           // one past last chunk fully inside
    for (int i = begin + tid; i < (c0 << 2) && i < end; i += NT) {
      float v = logits[i];
#pragma unroll
      for (int p = 0; p < NPROBE; ++p) cnt[p] += (v >= tv[p]);
    }
    const float4* lv4 = reinterpret_cast<const float4*>(logits);
    for (int c = c0 + tid; c < c1; c += NT) {
      float4 q = lv4[c];
#pragma unroll
      for (int p = 0; p < NPROBE; ++p)
        cnt[p] += (q.x >= tv[p]) + (q.y >= tv[p]) + (q.z >= tv[p]) + (q.w >= tv[p]);
    }
    for (int i = (c1 << 2) + tid; i < end; i += NT) {
      if (i < (c0 << 2)) continue;  // avoid double-count when slice < one chunk
      float v = logits[i];
#pragma unroll
      for (int p = 0; p < NPROBE; ++p) cnt[p] += (v >= tv[p]);
    }
    // Stage per-warp partials into hist[] scratch (256 ints, free in P2).
    int* pscratch = hist;   // layout: pscratch[warp*NPROBE + p]
#pragma unroll
    for (int p = 0; p < NPROBE; ++p) {
      int s = warp_sum(cnt[p]);
      if (lane == 0) pscratch[warp * NPROBE + p] = s;
    }
    __syncthreads();
    const int bbank = (pass & 1) ? 16 : 8;   // ctrl banks [8..14] / [16..22]
    if (warp == 0) {
#pragma unroll
      for (int p = 0; p < NPROBE; ++p) {
        int v = 0;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) v += pscratch[w * NPROBE + p];
        if (lane == 0) ctrl[bbank + p] = v;
      }
    }
    __syncthreads();
    cluster.sync();
    // Redundant reduce across cluster ranks: every CTA computes identical cnts.
    int gc[NPROBE];
#pragma unroll
    for (int p = 0; p < NPROBE; ++p) {
      int v = 0;
#pragma unroll
      for (int r = 0; r < CS; ++r) v += *cluster.map_shared_rank(ctrl + bbank + p, r);
      gc[p] = v;
    }
    // gc[p] = count(>= lo + step*(p+1)), decreasing in p.
    // Find the tightest bracket. New lo = largest probe with count>=k;
    // new hi = smallest probe with count<k.
    float newlo = lo, newhi = hi;
    // Check if any probe count lands in [k, CAP] -> done candidate.
#pragma unroll
    for (int p = 0; p < NPROBE; ++p) {
      float t = lo + step * (float)(p + 1);
      if (gc[p] >= k) { newlo = t; total = gc[p]; }
    }
    // newhi = smallest probe with count < k
#pragma unroll
    for (int p = NPROBE - 1; p >= 0; --p) {
      float t = lo + step * (float)(p + 1);
      if (gc[p] < k) newhi = t;
    }
    lo = newlo; hi = newhi;
    threshold = lo;
    if (total >= k && total <= CAP) found = 1;
  }

  // Conservative bisection fallback (also single-sync redundant).
#pragma unroll 1
  for (int it = 0; it < 20 && (total > CAP || total < k); ++it) {
    threshold = 0.5f * (lo + hi);
    int local = 0;
    for (int i = begin + tid; i < end; i += NT) local += logits[i] >= threshold;
    local = warp_sum(local);
    if (lane == 0) warp_i[warp] = local;
    __syncthreads();
    const int bank = 26 + (it & 1);
    if (warp == 0) {
      int v = lane < NWARPS ? warp_i[lane] : 0;
      v = warp_sum(v);
      if (lane == 0) ctrl[bank] = v;
    }
    __syncthreads();
    cluster.sync();
    int sum = 0;
#pragma unroll
    for (int r = 0; r < CS; ++r) sum += *cluster.map_shared_rank(ctrl + bank, r);
    if (sum >= k) lo = threshold;
    else hi = threshold;
    total = sum;
  }

  // Phase 3: warp-aggregated compaction into per-CTA shared memory. Track the
  // max ordered-key of the kept candidates so P4 can skip radix passes whose
  // top bytes are common to every candidate (free by-product of this scan).
  if (tid == 0) ctrl[7] = 0;
  __syncthreads();
  uint32_t tmax = 0u;
  for (int base = begin; base < end; base += NT) {
    int i = base + tid;
    float lv = (i < end) ? logits[i] : -CUDART_INF_F;
    bool keep = i < end && lv >= threshold;
    if (keep) tmax = max(tmax, ordered_key(lv));
    unsigned mask = __ballot_sync(0xffffffff, keep);
    int off = 0;
    if (lane == 0 && mask) off = atomicAdd(ctrl + 7, __popc(mask));
    off = __shfl_sync(0xffffffff, off, 0);
    if (keep) {
      int pos = off + __popc(mask & ((1u << lane) - 1u));
      if (pos < CAP) {
        cand_val[pos] = lv;
        cand_idx[pos] = i;
      }
    }
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) tmax = max(tmax, __shfl_down_sync(0xffffffffu, tmax, d));
  if (lane == 0) warp_i[warp] = (int)tmax;
  __syncthreads();
  if (warp == 0) {
    uint32_t v = lane < NWARPS ? (uint32_t)warp_i[lane] : 0u;
#pragma unroll
    for (int d = 16; d; d >>= 1) v = max(v, __shfl_down_sync(0xffffffffu, v, d));
    if (lane == 0) ctrl[15] = (int)v;
  }
  if (tid == 0 && ctrl[7] > CAP) ctrl[7] = CAP;
  cluster.sync();

  // Phase 4: rank 0 performs an exact 4x8-bit radix select directly over
  // the distributed candidate arrays, then emits all strict winners and a
  // bounded subset of kth-value ties.
  if (rank == 0) {
    // Prefix-window skip (exact): every candidate is >= threshold, so its
    // ordered key is in [ordered_key(threshold), kmax] where kmax was tracked
    // in P3. Radix passes whose top byte is identical across that whole range
    // put every candidate in a single bin, leaving kth == k and prefix == the
    // common bytes — so we start the pass loop at the first differing byte.
    uint32_t kmax = 0u;
#pragma unroll
    for (int r = 0; r < CS; ++r) kmax = max(kmax, (uint32_t)(*cluster.map_shared_rank(ctrl + 15, r)));
    uint32_t kmin = ordered_key(threshold);
    uint32_t diff = kmin ^ kmax;
    int lead = diff ? (__clz((int)diff) >> 3) : 3;   // common leading bytes (<=3)
    uint32_t prefix = (lead == 0) ? 0u : (kmin & (0xffffffffu << (32 - 8 * lead)));
    uint32_t mask_hi = (lead == 0) ? 0u : (0xffffffffu << (32 - 8 * lead));
    int kth = k;
#pragma unroll 1
    for (int pass = lead; pass < 4; ++pass) {
      for (int b = tid; b < 256; b += NT) hist[b] = 0;
      __syncthreads();
      int shift = 24 - 8 * pass;
#pragma unroll
      for (int r = 0; r < CS; ++r) {
        int* rc = cluster.map_shared_rank(ctrl, r);
        float* rv = cluster.map_shared_rank(cand_val, r);
        int count = rc[7];
        for (int j = tid; j < count; j += NT) {
          uint32_t key = ordered_key(rv[j]);
          if ((key & mask_hi) == prefix) atomicAdd(hist + ((key >> shift) & 255), 1);
        }
      }
      __syncthreads();
      // k-th-bin selection by a single warp with __shfl (no barrier scan).
      if (tid < 32) {
        const int lane = tid;
        int laneTotal = 0;
#pragma unroll
        for (int r = 0; r < 8; ++r) laneTotal += hist[lane * 8 + r];
        int inc = laneTotal;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          int t = __shfl_down_sync(0xffffffffu, inc, d);
          if (lane + d < 32) inc += t;
        }
        int aboveLane = inc - laneTotal;
        if (aboveLane < kth && aboveLane + laneTotal >= kth) {
          int prevcum = 0;
#pragma unroll
          for (int r = 7; r >= 0; --r) {
            int b = lane * 8 + r;
            int hb = hist[b];
            int aboveThis = aboveLane + prevcum;
            int here = aboveThis + hb;
            if (here >= kth && aboveThis < kth) {
              ctrl[8] = static_cast<int>(prefix | (static_cast<uint32_t>(b) << shift));
              ctrl[9] = kth - aboveThis;
            }
            prevcum += hb;
          }
        }
      }
      __syncthreads();
      prefix = static_cast<uint32_t>(ctrl[8]);
      kth = ctrl[9];
      mask_hi |= 255u << shift;
    }

    // After the radix select, the number of strict winners (key > prefix)
    // equals k - kth exactly, so the tie block starts at a known offset and
    // both emits fuse into ONE candidate scan (removes a full pass + 2
    // barriers in the dominant P4 phase). Strict winners are packed at
    // [0, strict); ties fill [strict, k).
    const int strict = k - kth;
    if (tid == 0) { ctrl[10] = 0; ctrl[11] = 0; }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < CS; ++r) {
      int* rc = cluster.map_shared_rank(ctrl, r);
      float* rv = cluster.map_shared_rank(cand_val, r);
      int* ri = cluster.map_shared_rank(cand_idx, r);
      int count = rc[7];
      int cround = (count + NT - 1) / NT * NT;
      for (int j = tid; j < cround; j += NT) {
        int valid = (j < count);
        uint32_t key = valid ? ordered_key(rv[j]) : 0u;
        int isA = valid && (key > prefix);
        int isB = valid && (key == prefix);
        unsigned mA = __ballot_sync(0xffffffffu, isA);
        if (mA) {
          int rk = __popc(mA & ((1u << lane) - 1u));
          int leader = __ffs(mA) - 1;
          int base = 0;
          if (lane == leader) base = atomicAdd(ctrl + 10, __popc(mA));
          base = __shfl_sync(0xffffffffu, base, leader);
          if (isA) output[base + rk] = ri[j];
        }
        unsigned mB = __ballot_sync(0xffffffffu, isB);
        if (mB) {
          int rk = __popc(mB & ((1u << lane) - 1u));
          int leader = __ffs(mB) - 1;
          int base = 0;
          if (lane == leader) base = atomicAdd(ctrl + 11, __popc(mB));
          base = __shfl_sync(0xffffffffu, base, leader);
          if (isB && strict + base + rk < k) output[strict + base + rk] = ri[j];
        }
      }
    }
  }
  cluster.sync();
}

template <int CAP, int CS, int NT, int NPROBE>
void launch_impl(
    const float* logits,
    const int* pre_idx,
    int* output,
    int n,
    int k,
    cudaStream_t stream) {
  auto kernel = gvr_topk_kernel<CAP, CS, NT, NPROBE>;
  constexpr int smem = CONTROL_INTS * sizeof(int) +
      CAP * (sizeof(float) + sizeof(int)) + 256 * sizeof(int) + 256 * sizeof(int) +
      (NT / 32) * (sizeof(float) + sizeof(int));
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(CS, 1, 1);
  config.blockDim = dim3(NT, 1, 1);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = CS;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, kernel, logits, pre_idx, output, n, k);
}

// Single-block GVR launched with a plain triple-chevron (no cluster attribute)
// to dodge the cudaLaunchKernelEx cluster-launch overhead. cg::this_cluster()
// degenerates to a 1-block cluster, so cluster.sync() is a block barrier.
template <int CAP, int NT, int NPROBE>
void launch_single(
    const float* logits,
    const int* pre_idx,
    int* output,
    int n,
    int k,
    cudaStream_t stream) {
  auto kernel = gvr_topk_kernel<CAP, 1, NT, NPROBE>;
  constexpr int smem = CONTROL_INTS * sizeof(int) +
      CAP * (sizeof(float) + sizeof(int)) + 256 * sizeof(int) + 256 * sizeof(int) +
      (NT / 32) * (sizeof(float) + sizeof(int));
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kernel<<<1, NT, smem, stream>>>(logits, pre_idx, output, n, k);
}

}  // namespace

void champion_launcher(
    const float* logits,
    const int* pre_idx,
    int* indices,
    int npad,
    int n_valid,
    int k,
    cudaStream_t stream) {
  if (k == 1024 && n_valid == 1027) {
    exclude_three_min_kernel<<<1, 1024, 0, stream>>>(logits, indices, n_valid);
    return;
  }
  if (npad <= 12288) {
    direct_radix_kernel<<<1, 1024, 0, stream>>>(logits, pre_idx, indices, npad, k);
    return;
  }
  if (k == 2048) {
    if (npad < 65536) launch_single<6144, 1024, 3>(logits, pre_idx, indices, npad, k, stream);
    else launch_impl<6144, 8, 1024, 3>(logits, pre_idx, indices, npad, k, stream);
  } else {
    if (npad < 65536) launch_single<5120, 1024, 3>(logits, pre_idx, indices, npad, k, stream);
    else launch_impl<5120, 8, 1024, 3>(logits, pre_idx, indices, npad, k, stream);
  }
}
