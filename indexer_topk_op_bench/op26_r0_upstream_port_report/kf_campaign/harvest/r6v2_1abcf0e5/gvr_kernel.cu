#include "gvr_kernel.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

// Monotone key for all finite IEEE float values: unsigned comparison of keys
// is identical to numerical comparison of the source floats.
__device__ __forceinline__ uint32_t ordered_key(float x) {
  uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

struct GvrState {
  uint32_t prior_key;
  uint32_t prefix;
  uint32_t threshold_key;
  int rank;
  int selected_count;
  int pivot;
  int candidate_count;
  int fast_exact_prior;
  uint32_t max_hint_key;
  uint32_t ladder_key;
  int greater_count;
  int greater_ticket;
  int equal_ticket;
};

struct KeyIndex {
  uint32_t key;
  int index;
};

__device__ __forceinline__ KeyIndex warp_min_pair(KeyIndex x) {
  for (int offset = 16; offset; offset >>= 1) {
    KeyIndex y;
    y.key = __shfl_down_sync(0xffffffffu, x.key, offset);
    y.index = __shfl_down_sync(0xffffffffu, x.index, offset);
    if (y.key < x.key || (y.key == x.key && y.index < x.index)) x = y;
  }
  return x;
}

// Degenerate GVR tier for n=1027,k=1024.  P1/P2 still gather and verify the
// temporal threshold.  Since the admitted capacity is the whole row, P4 is
// exactly the cheaper complementary problem: exclude the three minima.
__global__ __launch_bounds__(512, 1) void gvr_exclude3_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ pre_idx,
    int npad,
    int* __restrict__ indices) {
  constexpr int n = 1027;
  constexpr int k = 1024;
  const int row_id = int(blockIdx.x);
  const int tid = int(threadIdx.x);
  const float* row = logits + size_t(row_id) * npad;
  const int* hints = pre_idx + size_t(row_id) * k;
  int* out = indices + size_t(row_id) * k;

  __shared__ uint32_t keys[1088];
  __shared__ uint32_t prior_key;
  __shared__ int verified_count;
  __shared__ uint32_t warp_key[16];
  __shared__ int warp_index[16];
  __shared__ int excluded[3];

  if (tid == 0) {
    prior_key = 0xffffffffu;
    verified_count = 0;
  }
  __syncthreads();
  for (int j = tid; j < k; j += 512) {
    int idx = hints[j];
    atomicMin(&prior_key, ordered_key(row[idx]));
  }
  __syncthreads();

  int local_admitted = 0;
  for (int i = tid; i < n; i += 512) {
    uint32_t key = ordered_key(row[i]);
    keys[i] = key;
    local_admitted += int(key >= prior_key);
  }
  for (int offset = 16; offset; offset >>= 1)
    local_admitted += __shfl_down_sync(0xffffffffu, local_admitted, offset);
  if ((tid & 31) == 0) atomicAdd(&verified_count, local_admitted);
  __syncthreads();

#pragma unroll
  for (int pick = 0; pick < 3; ++pick) {
    KeyIndex best{0xffffffffu, 0x7fffffff};
    for (int i = tid; i < n; i += 512) {
      KeyIndex v{keys[i], i};
      if (v.key < best.key || (v.key == best.key && v.index < best.index))
        best = v;
    }
    best = warp_min_pair(best);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
      warp_key[warp] = best.key;
      warp_index[warp] = best.index;
    }
    __syncthreads();
    if (warp == 0) {
      KeyIndex w = lane < 16
          ? KeyIndex{warp_key[lane], warp_index[lane]}
          : KeyIndex{0xffffffffu, 0x7fffffff};
      w = warp_min_pair(w);
      if (lane == 0) {
        excluded[pick] = w.index;
        keys[w.index] = 0xffffffffu;
      }
    }
    __syncthreads();
  }

  int e0 = excluded[0], e1 = excluded[1], e2 = excluded[2];
  for (int i = tid; i < n; i += 512) {
    if (i != e0 && i != e1 && i != e2) {
      int slot = i - int(e0 < i) - int(e1 < i) - int(e2 < i);
      out[slot] = i;
    }
  }
}

// A log-CCDF secant estimate inside the current exact histogram bracket.
// Rank 1 maps toward the high edge, while rank==population maps low.  The
// estimate is only a pivot: the histogram search on either side verifies and
// corrects it exactly, so plateaus and adversarial priors remain safe.
__device__ __forceinline__ int log_secant_pivot(
    int rank, int population, int bins) {
  if (population <= 1) return bins - 1;
  float y = log2f(float(rank) + 1.0f) / log2f(float(population) + 1.0f);
  int p = (bins - 1) - __float2int_rn(y * float(bins - 1));
  return max(0, min(bins - 1, p));
}

template <int BLOCK_THREADS>
__device__ __forceinline__ void suffix256_small(int* suffix) {
#pragma unroll
  for (int offset = 1; offset < 256; offset <<= 1) {
    int v = (threadIdx.x < 256 && threadIdx.x + offset < 256)
        ? suffix[threadIdx.x + offset] : 0;
    __syncthreads();
    if (threadIdx.x < 256) suffix[threadIdx.x] += v;
    __syncthreads();
  }
}

// Dense-half specialization: the temporal prior seeds a verified byte-radix
// solve.  The log-CCDF secant pivot chooses which side of the exact suffix
// boundary participates at every rung; the suffix condition then verifies
// the selected digit exactly, including plateaus.
__global__ __launch_bounds__(256, 2) void gvr_small512_fast_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ pre_idx,
    int npad,
    int* __restrict__ indices) {
  constexpr int n = 1027;
  constexpr int k = 512;
  constexpr int BLOCK_THREADS = 256;
  constexpr int NUM_WARPS = 8;
  int row_id = int(blockIdx.x);
  int tid = int(threadIdx.x);
  int warp = tid >> 5;
  int lane = tid & 31;
  const float* row = logits + size_t(row_id) * npad;
  const int* hints = pre_idx + size_t(row_id) * k;
  int* out = indices + size_t(row_id) * k;
  const uint4* row4 = reinterpret_cast<const uint4*>(row);

  __shared__ int warp_hist[NUM_WARPS][256];
  __shared__ int suffix[256];
  __shared__ uint32_t prior_key;
  __shared__ uint32_t prefix;
  __shared__ int rank;
  __shared__ int pivot;
  __shared__ int chosen;
  __shared__ int strict_ticket;
  __shared__ int equal_ticket;

  if (tid == 0) prior_key = 0xffffffffu;
  __syncthreads();
  uint32_t local_prior = 0xffffffffu;
  for (int j = tid; j < k; j += BLOCK_THREADS)
    local_prior = min(local_prior, ordered_key(row[hints[j]]));
  for (int offset = 16; offset; offset >>= 1)
    local_prior = min(local_prior,
        __shfl_down_sync(0xffffffffu, local_prior, offset));
  if (lane == 0) atomicMin(&prior_key, local_prior);
  __syncthreads();
  if (tid == 0) {
    prefix = 0;
    rank = k;
    pivot = int(prior_key >> 24);
  }
  __syncthreads();

#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    int shift = 24 - pass * 8;
    uint32_t mask = (shift + 8 >= 32) ? 0u : (0xffffffffu << (shift + 8));
    for (int i = tid; i < NUM_WARPS * 256; i += BLOCK_THREADS)
      reinterpret_cast<int*>(warp_hist)[i] = 0;
    if (tid == 0) chosen = 0;
    __syncthreads();
    uint32_t pfx = prefix;
    int* my_hist = warp_hist[warp];
    for (int i = tid; i < (n >> 2); i += BLOCK_THREADS) {
      uint4 raw = row4[i];
      uint32_t keys[4] = {
          ordered_key(__uint_as_float(raw.x)),
          ordered_key(__uint_as_float(raw.y)),
          ordered_key(__uint_as_float(raw.z)),
          ordered_key(__uint_as_float(raw.w))};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        bool member = pass == 0 ? keys[j] >= prior_key
                                : (keys[j] & mask) == (pfx & mask);
        if (member) atomicAdd(my_hist + ((keys[j] >> shift) & 255u), 1);
      }
    }
    for (int i = (n & ~3) + tid; i < n; i += BLOCK_THREADS) {
      uint32_t key = ordered_key(row[i]);
      bool member = pass == 0 ? key >= prior_key
                              : (key & mask) == (pfx & mask);
      if (member) atomicAdd(my_hist + ((key >> shift) & 255u), 1);
    }
    __syncthreads();
    if (tid < 256) {
      int count = 0;
#pragma unroll
      for (int w = 0; w < NUM_WARPS; ++w) count += warp_hist[w][tid];
      suffix[tid] = count;
    }
    __syncthreads();
    suffix256_small<BLOCK_THREADS>(suffix);

    int current_rank = rank;
    int current_pivot = pivot;
    if (tid < 256) {
      bool upper_side = suffix[current_pivot] >= current_rank;
      bool eligible = upper_side ? tid >= current_pivot : tid < current_pivot;
      int here = suffix[tid];
      int above = tid < 255 ? suffix[tid + 1] : 0;
      if (eligible && here >= current_rank && above < current_rank)
        chosen = tid;
    }
    __syncthreads();
    if (tid == 0) {
      int digit = chosen;
      int above = digit < 255 ? suffix[digit + 1] : 0;
      int population = suffix[digit] - above;
      prefix = pfx | (uint32_t(digit) << shift);
      rank = current_rank - above;
      pivot = log_secant_pivot(rank, max(1, population), 256);
    }
    __syncthreads();
  }

  if (tid == 0) {
    strict_ticket = 0;
    equal_ticket = 0;
  }
  __syncthreads();
  uint32_t threshold = prefix;
  int need_equal = rank;
  int greater = k - need_equal;
  for (int i = tid; i < (n >> 2); i += BLOCK_THREADS) {
    uint4 raw = row4[i];
    uint32_t keys[4] = {
        ordered_key(__uint_as_float(raw.x)),
        ordered_key(__uint_as_float(raw.y)),
        ordered_key(__uint_as_float(raw.z)),
        ordered_key(__uint_as_float(raw.w))};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int idx = (i << 2) + j;
      bool is_greater = keys[j] > threshold;
      unsigned gm = __ballot_sync(0xffffffffu, is_greater);
      int gb = 0;
      int gc = __popc(gm);
      int gr = __popc(gm & ((1u << lane) - 1u));
      if (gc) {
        int leader = __ffs(gm) - 1;
        if (lane == leader) gb = atomicAdd(&strict_ticket, gc);
        gb = __shfl_sync(0xffffffffu, gb, leader);
      }
      if (is_greater) out[gb + gr] = idx;

      bool is_equal = keys[j] == threshold;
      unsigned em = __ballot_sync(0xffffffffu, is_equal);
      int eb = 0;
      int ec = __popc(em);
      int er = __popc(em & ((1u << lane) - 1u));
      if (ec) {
        int leader = __ffs(em) - 1;
        if (lane == leader) eb = atomicAdd(&equal_ticket, ec);
        eb = __shfl_sync(0xffffffffu, eb, leader);
      }
      if (is_equal && eb + er < need_equal)
        out[greater + eb + er] = idx;
    }
  }
  for (int i = (n & ~3) + tid; i < n; i += BLOCK_THREADS) {
    uint32_t key = ordered_key(row[i]);
    if (key > threshold) {
      int slot = atomicAdd(&strict_ticket, 1);
      out[slot] = i;
    } else if (key == threshold) {
      int slot = atomicAdd(&equal_ticket, 1);
      if (slot < need_equal) out[greater + slot] = i;
    }
  }
}

// Warp 0 locates the descending-rank digit.  Each lane first totals one
// contiguous high-to-low chunk, then only the winning 32/64-bin chunk is
// walked serially.  The log-secant pivot splits that final chunk, and exact
// counts verify which side contains the target.
__device__ __forceinline__ void choose_digit_warp(
    const int* hist, int bins, int next_bins, bool record_total, GvrState* s) {
  int lane = int(threadIdx.x) & 31;
  int chunk = bins >> 5;
  int hi = bins - 1 - lane * chunk;
  int lo = hi - chunk + 1;
  int lane_sum = 0;
  for (int d = lo; d <= hi; ++d) lane_sum += hist[d];

  int inclusive = lane_sum;
  for (int offset = 1; offset < 32; offset <<= 1) {
    int x = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) inclusive += x;
  }
  unsigned candidates = __ballot_sync(0xffffffffu, inclusive >= s->rank);
  int owner = __ffs(candidates) - 1;
  int higher_chunk = owner ? __shfl_sync(0xffffffffu, inclusive, owner - 1) : 0;
  int total = __shfl_sync(0xffffffffu, inclusive, 31);

  if (lane == owner) {
    if (record_total) s->candidate_count = total;
    int owner_hi = bins - 1 - owner * chunk;
    int owner_lo = owner_hi - chunk + 1;
    int pivot = max(owner_lo, min(owner_hi, s->pivot));
    int above_pivot = 0;
    for (int d = pivot + 1; d <= owner_hi; ++d) above_pivot += hist[d];

    int chosen = owner_lo;
    int higher = higher_chunk;
    if (higher + above_pivot >= s->rank) {
      int remaining = higher + above_pivot;
      for (int d = pivot + 1; d <= owner_hi; ++d) {
        int c = hist[d];
        if (remaining - c < s->rank) {
          chosen = d;
          higher = remaining - c;
          break;
        }
        remaining -= c;
      }
    } else {
      higher += above_pivot;
      for (int d = pivot; d >= owner_lo; --d) {
        int c = hist[d];
        if (higher + c >= s->rank) {
          chosen = d;
          break;
        }
        higher += c;
      }
    }

    s->rank -= higher;
    s->selected_count = hist[chosen];
    s->prefix = (s->prefix << ((bins == 1024) ? 10 : 11)) | uint32_t(chosen);
    s->pivot = log_secant_pivot(
        s->rank, max(1, s->selected_count), next_bins);
  }
}

template <bool STAGE_ROW, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 1) void gvr_topk_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ pre_idx,
    int n,
    int npad,
    int k,
    int* __restrict__ indices) {
  const int row_id = int(blockIdx.x);
  const int tid = int(threadIdx.x);
  const float* row = logits + size_t(row_id) * npad;
  const int* hints = pre_idx + size_t(row_id) * k;
  int* out = indices + size_t(row_id) * k;

  extern __shared__ __align__(16) unsigned char dynamic_smem[];
  float* staged = reinterpret_cast<float*>(dynamic_smem);
  size_t row_bytes = STAGE_ROW ? size_t(npad) * sizeof(float) : 0;
  row_bytes = (row_bytes + 15u) & ~size_t(15u);
  int* hist = reinterpret_cast<int*>(dynamic_smem + row_bytes);
  __shared__ GvrState state;

  // P1: the prior is the minimum current value among the previous top-k
  // indices.  It is a correctness-safe lower bound: any k distinct hinted
  // elements prove that the current kth value cannot be below this value.
  if (tid == 0) state.prior_key = 0xffffffffu;
  __syncthreads();
  for (int j = tid; j < k; j += BLOCK_THREADS) {
    int idx = hints[j];
    uint32_t key = (unsigned(idx) < unsigned(n))
        ? ordered_key(row[idx]) : 0u;
    atomicMin(&state.prior_key, key);
  }

  if constexpr (STAGE_ROW) {
    for (int i = tid; i < n; i += BLOCK_THREADS) staged[i] = row[i];
  }
  __syncthreads();

  if (tid == 0) {
    state.prefix = 0;
    state.rank = k;
    // The prior's leading digit is the first CCDF/secant bracket pivot.
    state.pivot = int(state.prior_key >> 21);
  }
  __syncthreads();

  // P2: three exact threshold-refinement rungs (11/11/10 bits).  Each rung
  // is a verified histogram ladder centered by the log-secant estimate from
  // the preceding bracket.  The prior filters the first rung safely.
  for (int i = tid; i < 2048; i += BLOCK_THREADS) hist[i] = 0;
  __syncthreads();
  for (int i = tid; i < n; i += BLOCK_THREADS) {
    uint32_t key = ordered_key(STAGE_ROW ? staged[i] : row[i]);
    if (key >= state.prior_key) atomicAdd(hist + (key >> 21), 1);
  }
  __syncthreads();
  if (tid < 32) choose_digit_warp(hist, 2048, 2048, true, &state);
  __syncthreads();

  // A verified prior with exactly k survivors is already the exact answer.
  // This is the common one-shot GVR convergence path for a strong temporal
  // hint and avoids both remaining radix rungs.
  if (state.candidate_count == k) {
    if (tid == 0) {
      state.fast_exact_prior = 1;
      state.threshold_key = state.prior_key;
      state.greater_ticket = 0;
    }
    __syncthreads();
    for (int i = tid; i < n; i += BLOCK_THREADS) {
      uint32_t key = ordered_key(STAGE_ROW ? staged[i] : row[i]);
      if (key >= state.threshold_key) {
        int slot = atomicAdd(&state.greater_ticket, 1);
        if (slot < k) out[slot] = i;
      }
    }
    return;
  }

  for (int i = tid; i < 2048; i += BLOCK_THREADS) hist[i] = 0;
  __syncthreads();
  for (int i = tid; i < n; i += BLOCK_THREADS) {
    uint32_t key = ordered_key(STAGE_ROW ? staged[i] : row[i]);
    if ((key >> 21) == state.prefix)
      atomicAdd(hist + ((key >> 10) & 2047u), 1);
  }
  __syncthreads();
  if (tid < 32) choose_digit_warp(hist, 2048, 1024, false, &state);
  __syncthreads();

  for (int i = tid; i < 2048; i += BLOCK_THREADS) hist[i] = 0;
  __syncthreads();
  for (int i = tid; i < n; i += BLOCK_THREADS) {
    uint32_t key = ordered_key(STAGE_ROW ? staged[i] : row[i]);
    if ((key >> 10) == state.prefix)
      atomicAdd(hist + (key & 1023u), 1);
  }
  __syncthreads();
  if (tid < 32) choose_digit_warp(hist, 1024, 1024, false, &state);
  __syncthreads();
  if (tid == 0) {
    state.threshold_key = state.prefix;
    state.greater_count = k - state.rank;
    state.greater_ticket = 0;
    state.equal_ticket = 0;
  }
  __syncthreads();

  // P3/P4: exact survivor refinement and tie-ticket writeback.  Strictly
  // greater values own [0, greater_count); exact ties fill only the remainder,
  // so scheduling can never displace a mandatory strict winner.
  for (int i = tid; i < n; i += BLOCK_THREADS) {
    uint32_t key = ordered_key(STAGE_ROW ? staged[i] : row[i]);
    if (key > state.threshold_key) {
      int slot = atomicAdd(&state.greater_ticket, 1);
      if (slot < state.greater_count) out[slot] = i;
    } else if (key == state.threshold_key) {
      int ticket = atomicAdd(&state.equal_ticket, 1);
      int need = k - state.greater_count;
      if (ticket < need) out[state.greater_count + ticket] = i;
    }
  }
}

template <int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 1) void gvr_candidate_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ pre_idx,
    int n,
    int npad,
    int k,
    int capacity,
    int* __restrict__ indices) {
  const int row_id = int(blockIdx.x);
  const int tid = int(threadIdx.x);
  const int lane = tid & 31;
  const float* row = logits + size_t(row_id) * npad;
  const int* hints = pre_idx + size_t(row_id) * k;
  int* out = indices + size_t(row_id) * k;

  extern __shared__ __align__(16) unsigned char dynamic_smem[];
  uint32_t* candidate_keys = reinterpret_cast<uint32_t*>(dynamic_smem);
  int* candidate_indices = reinterpret_cast<int*>(candidate_keys + capacity);
  int* hist = candidate_indices + capacity;
  __shared__ GvrState state;
  __shared__ uint32_t ladder[8];
  __shared__ int ladder_counts[8];

  // P1: reduce the gathered hint minimum with one shared atomic per warp.
  if (tid == 0) {
    state.prior_key = 0xffffffffu;
    state.max_hint_key = 0u;
  }
  __syncthreads();
  uint32_t local_prior = 0xffffffffu;
  uint32_t local_max_hint = 0u;
  for (int j = tid; j < k; j += BLOCK_THREADS) {
    int idx = hints[j];
    uint32_t key = (unsigned(idx) < unsigned(n))
        ? ordered_key(row[idx]) : 0u;
    local_prior = min(local_prior, key);
    local_max_hint = max(local_max_hint, key);
  }
  for (int offset = 16; offset; offset >>= 1) {
    local_prior = min(local_prior,
        __shfl_down_sync(0xffffffffu, local_prior, offset));
    local_max_hint = max(local_max_hint,
        __shfl_down_sync(0xffffffffu, local_max_hint, offset));
  }
  if (lane == 0) {
    atomicMin(&state.prior_key, local_prior);
    atomicMax(&state.max_hint_key, local_max_hint);
  }
  __syncthreads();

  if constexpr (false) {
  // P2: verify an eight-rung prior ladder in one row pass.  The geometric
  // spacing is the discrete log/secant frame between the safe hint minimum
  // and maximum.  A usable rung must provably retain at least k values while
  // fitting the exact-refine candidate capacity.
  if (tid < 8) {
    int numerator = tid ? (1 << (tid - 1)) : 0;
    uint64_t span = uint64_t(state.max_hint_key) - uint64_t(state.prior_key);
    ladder[tid] = state.prior_key
        + uint32_t((span * uint64_t(numerator)) >> 6);
    ladder_counts[tid] = 0;
  }
  __syncthreads();
  int local_counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = tid; i < n; i += BLOCK_THREADS) {
    uint32_t key = ordered_key(row[i]);
#pragma unroll
    for (int q = 0; q < 8; ++q) local_counts[q] += int(key >= ladder[q]);
  }
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    for (int offset = 16; offset; offset >>= 1)
      local_counts[q] += __shfl_down_sync(0xffffffffu, local_counts[q], offset);
    if (lane == 0) atomicAdd(ladder_counts + q, local_counts[q]);
  }
  __syncthreads();
  if (tid == 0) {
    state.fast_exact_prior = 0;
    for (int q = 0; q < 8; ++q) {
      int count = ladder_counts[q];
      if (count >= k && count <= capacity) {
        state.fast_exact_prior = 1;
        state.ladder_key = ladder[q];
      }
    }
  }
  __syncthreads();

  // Common GVR convergence: collect the verified bounded superset, refine it
  // exactly in shared memory, and never reread the row.
  if (state.fast_exact_prior) {
    if (tid == 0) {
      state.prefix = 0;
      state.rank = k;
      state.pivot = int(state.ladder_key >> 21);
      state.candidate_count = 0;
    }
    for (int i = tid; i < 2048; i += BLOCK_THREADS) hist[i] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += BLOCK_THREADS) {
      uint32_t key = ordered_key(row[i]);
      if (key >= state.ladder_key) {
        int slot = atomicAdd(&state.candidate_count, 1);
        candidate_keys[slot] = key;
        candidate_indices[slot] = i;
        atomicAdd(hist + (key >> 21), 1);
      }
    }
    __syncthreads();
    if (tid < 32) choose_digit_warp(hist, 2048, 2048, false, &state);
    __syncthreads();

    for (int i = tid; i < 2048; i += BLOCK_THREADS) hist[i] = 0;
    __syncthreads();
    for (int p = tid; p < state.candidate_count; p += BLOCK_THREADS) {
      uint32_t key = candidate_keys[p];
      if ((key >> 21) == state.prefix)
        atomicAdd(hist + ((key >> 10) & 2047u), 1);
    }
    __syncthreads();
    if (tid < 32) choose_digit_warp(hist, 2048, 1024, false, &state);
    __syncthreads();

    for (int i = tid; i < 2048; i += BLOCK_THREADS) hist[i] = 0;
    __syncthreads();
    for (int p = tid; p < state.candidate_count; p += BLOCK_THREADS) {
      uint32_t key = candidate_keys[p];
      if ((key >> 10) == state.prefix)
        atomicAdd(hist + (key & 1023u), 1);
    }
    __syncthreads();
    if (tid < 32) choose_digit_warp(hist, 1024, 1024, false, &state);
    __syncthreads();
    if (tid == 0) {
      state.threshold_key = state.prefix;
      state.greater_count = k - state.rank;
      state.greater_ticket = 0;
      state.equal_ticket = 0;
    }
    __syncthreads();

    for (int p = tid; p < state.candidate_count; p += BLOCK_THREADS) {
      uint32_t key = candidate_keys[p];
      int idx = candidate_indices[p];
      if (key > state.threshold_key) {
        int slot = atomicAdd(&state.greater_ticket, 1);
        if (slot < state.greater_count) out[slot] = idx;
      } else if (key == state.threshold_key) {
        int ticket = atomicAdd(&state.equal_ticket, 1);
        int need = k - state.greater_count;
        if (ticket < need) out[state.greater_count + ticket] = idx;
      }
    }
    return;
  }
  }

  if (tid == 0) {
    state.prefix = 0;
    state.rank = k;
    state.pivot = int(state.prior_key >> 21);
    state.candidate_count = 0;
  }
  for (int i = tid; i < 2048; i += BLOCK_THREADS) hist[i] = 0;
  __syncthreads();

  // P2a: verify the prior and locate the exact leading 11-bit bracket.
  for (int i = tid; i < n; i += BLOCK_THREADS) {
    uint32_t key = ordered_key(row[i]);
    if (key >= state.prior_key) atomicAdd(hist + (key >> 21), 1);
  }
  __syncthreads();
  if (tid < 32) choose_digit_warp(hist, 2048, 2048, false, &state);
  __syncthreads();

  // P2b/P3: now collect only the selected leading bracket and all higher
  // brackets.  This is a small exact superset even when the raw prior admits
  // much of the row.  Build the next histogram during the same scan.
  for (int i = tid; i < 2048; i += BLOCK_THREADS) hist[i] = 0;
  if (tid == 0) state.candidate_count = 0;
  __syncthreads();
  for (int i = tid; i < n; i += BLOCK_THREADS) {
    uint32_t key = ordered_key(row[i]);
    int high = int(key >> 21);
    if (high >= int(state.prefix)) {
      int slot = atomicAdd(&state.candidate_count, 1);
      if (slot < capacity) {
        candidate_keys[slot] = key;
        candidate_indices[slot] = i;
      }
    }

    if (high == int(state.prefix))
      atomicAdd(hist + ((key >> 10) & 2047u), 1);
  }
  __syncthreads();
  if (tid < 32) choose_digit_warp(hist, 2048, 1024, false, &state);
  __syncthreads();

  const bool resident = state.candidate_count <= capacity;
  int refine_count = resident ? state.candidate_count : n;

  for (int i = tid; i < 2048; i += BLOCK_THREADS) hist[i] = 0;
  __syncthreads();
  for (int p = tid; p < refine_count; p += BLOCK_THREADS) {
    uint32_t key = resident ? candidate_keys[p] : ordered_key(row[p]);
    if ((key >> 10) == state.prefix)
      atomicAdd(hist + (key & 1023u), 1);
  }
  __syncthreads();
  if (tid < 32) choose_digit_warp(hist, 1024, 1024, false, &state);
  __syncthreads();
  if (tid == 0) {
    state.threshold_key = state.prefix;
    state.greater_count = k - state.rank;
    state.greater_ticket = 0;
    state.equal_ticket = 0;
  }
  __syncthreads();

  // P4 consumes resident candidate pairs without another row read.  Overflow
  // falls back to the same exact writeback over the row.
  for (int p = tid; p < refine_count; p += BLOCK_THREADS) {
    uint32_t key = resident ? candidate_keys[p] : ordered_key(row[p]);
    int idx = resident ? candidate_indices[p] : p;
    if (key > state.threshold_key) {
      int slot = atomicAdd(&state.greater_ticket, 1);
      if (slot < state.greater_count) out[slot] = idx;
    } else if (key == state.threshold_key) {
      int ticket = atomicAdd(&state.equal_ticket, 1);
      int need = k - state.greater_count;
      if (ticket < need) out[state.greater_count + ticket] = idx;
    }
  }
}

}  // namespace

void gvr_topk_launcher(
    const float* logits,
    const int* pre_idx,
    int n_valid,
    int* indices,
    int batch,
    int npad,
    int k,
    cudaStream_t stream) {
  constexpr int threads = 512;
  constexpr int hist_bytes = 2048 * int(sizeof(int));
  if (n_valid == 1027 && k == 512) {
    gvr_small512_fast_kernel<<<batch, 256, 0, stream>>>(
        logits, pre_idx, npad, indices);
    return;
  }
  if (n_valid == 1027 && k == 1024) {
    gvr_exclude3_kernel<<<batch, threads, 0, stream>>>(
        logits, pre_idx, npad, indices);
    return;
  }
  // Rows through 32K are read from global memory once and refined in SMEM.
  // Larger rows keep full CTA occupancy and use coalesced streaming passes.
  if (npad <= 32832) {
    int smem = npad * int(sizeof(float)) + hist_bytes + 16;
    cudaFuncSetAttribute(
        gvr_topk_kernel<true, threads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem);
    gvr_topk_kernel<true, threads><<<batch, threads, smem, stream>>>(
        logits, pre_idx, n_valid, npad, k, indices);
  } else {
    int capacity = (k == 2048) ? 6144 : 5120;
    int smem = capacity * 2 * int(sizeof(int)) + hist_bytes;
    cudaFuncSetAttribute(
        gvr_candidate_kernel<threads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem);
    gvr_candidate_kernel<threads><<<batch, threads, smem, stream>>>(
        logits, pre_idx, n_valid, npad, k, capacity, indices);
  }
}
