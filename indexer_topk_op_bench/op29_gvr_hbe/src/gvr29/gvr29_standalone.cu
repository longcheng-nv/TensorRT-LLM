/// op29 GVR-HBE standalone driver — FORK of the vendored sglang v2 driver\n/// (ops/sglang_v2, Apache-2.0) + the HBE hint-boundary-exact streaming path\n/// (sgl_kernel/deepseek_v4/topk_hbe.cuh). Baseline arm stays the untouched\n/// ops/sglang_v2 build; this is the EXPERIMENT arm.\n///\n/// Original header:
/// vendored 2026-07-13 from sglang@main:
///   python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk_impl.cuh
///   python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh   (kernels verbatim)
/// for the indexer_topk_op_bench A/B (op #28).
///
/// The device code (topk_impl.cuh + the four __global__ kernels below) is
/// verbatim upstream. Only the tvm-ffi host layer (TopKKernel::plan/transform)
/// is replaced by an equivalent torch-extension host layer that reproduces the
/// SAME dispatch constants and LaunchKernel configs (PDL + cluster dims).
///
/// Like the op#11 top512 vendor: identity page table (page_table=[0],
/// page_bits=21 so page_to_indices(i)==i for every i < N <= 2^21), per-batch
/// page_table stride = 0.
///
/// Upstream dispatch reproduced in `transform_launch` (see topk_v2.cuh):
///   use_cluster = max_seq_len > cluster_floor && batch <= 512
///     cluster_floor = 65536 (32768 when batch <= 15)
///     batch <= 30  -> topk_small_batch_kernel            (1 fused kernel)
///     batch <= 512 -> topk_persistent_cluster_kernel     (2 stitched kernels,
///                     + topk_main_kernel<level=3>          PDL overlapped)
///   else level 0/1/2 -> topk_main_kernel                 (1 kernel)
/// plan (topk_plan, 1 block) picks cluster_threshold from the seq_len
/// distribution; exposed separately so the harness can run it timed or not.
#include <sgl_kernel/deepseek_v4/topk_impl.cuh>
#include <sgl_kernel/deepseek_v4/topk_hbe.cuh>

#include <cstdint>
#include <iterator>
#include <bit>

namespace {

namespace impl = device::topk;
using impl::TopKProblem;

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;
using Cluster = impl::TopKCluster<8>;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kClusterSize = Cluster::kClusterSize;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384

#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)
#define CLUSTER_TOPK_KERNEL TOPK_KERNEL __cluster_dims__(1, kClusterSize, 1)

constexpr uint32_t kClusterFloor = 65536;
constexpr uint32_t kClusterMaxBatch = 512;
constexpr uint32_t kNumPersistentClusters = 15 * kOccupancy;

/// Metadata tensor rows (each 8 B / 2 int32). Row 0 is the global plan result;
/// rows 1..N are the (batch_id, seq_len) of items routed to the cluster pool.
struct alignas(8) GlobalMetadata {
  uint32_t cluster_threshold;
  uint32_t num_cluster_items;  // N = number of items routed to the cluster pool
};
struct alignas(8) PlanItem {
  uint32_t batch_id;
  uint32_t seq_len;
};
static_assert(sizeof(GlobalMetadata) == 2 * sizeof(int32_t) && sizeof(PlanItem) == sizeof(GlobalMetadata));

struct TopKLaunchParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;      // optional raw (pre-transform) indices output; nullptr if unused
  const PlanItem* __restrict__ metadata;  // [0]=GlobalMetadata, [1+i]=PlanItem
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;
  uint32_t cluster_floor;  // seq_len > this routes to the cluster path (batch-aware, host-set)

  SGL_DEVICE const GlobalMetadata& global() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE uint32_t cluster_threshold() const {
    return global().cluster_threshold;
  }
  SGL_DEVICE const PlanItem& item(uint32_t i) const {
    return metadata[1 + i];
  }
  SGL_DEVICE int32_t* get_output_ptr(uint32_t batch_id) const {
    return page_indices + batch_id * static_cast<int64_t>(topk);
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id, uint32_t seq_len) const {
    const auto k = static_cast<int64_t>(topk);
    return TopKProblem{
        .in = scores + batch_id * score_stride,
        .out = page_indices + batch_id * k,
        .raw_out = raw_indices != nullptr ? raw_indices + batch_id * k : nullptr,
        .page_table = page_table + batch_id * page_table_stride,
        .topk = topk,
        .seq_len = seq_len,
        .page_bits = page_bits,
    };
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id) const {
    return this->problem(batch_id, static_cast<uint32_t>(seq_lens[batch_id]));
  }
};

/**
 * \brief Persistent cluster kernel for the long items. It will handle long inputs.
 * The short items are handled by the separate topk_kernel.
 */
template <bool kPDL>
CLUSTER_TOPK_KERNEL void topk_persistent_cluster_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  __shared__ impl::MaxSmem<Cluster::Smem> smem;
  const uint32_t num_cluster_items = params.global().num_cluster_items;
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
#pragma unroll 1
  for (uint32_t w = blockIdx.x; w < num_cluster_items; w += kNumPersistentClusters) {
    const auto it = params.item(w);
    const auto problem = params.problem(it.batch_id, it.seq_len);
    Cluster::forward<false>(problem, &smem);
    __syncthreads();
  }
}

template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kNumElems; ++i) {
    if (const auto tx = i * kBlockSize + threadIdx.x; tx < topk) {
      __builtin_assume(tx < kMaxTopK);
      f(tx, i);
    }
  }
}

template <bool kPDL>
SGL_DEVICE void trivial_transform(const TopKProblem& problem) {
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t) {
    problem.transform_output(tx, tx < problem.seq_len ? static_cast<int32_t>(tx) : -1);
  });
}

SGL_DEVICE void problem_transform(TopKProblem& problem, int32_t* output_ptr) {
  static_assert(kMaxTopK % kBlockSize == 0);
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
  int32_t source_index[kNumElems];
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { source_index[i] = problem.out[tx]; });
  problem.out = output_ptr;
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { problem.transform_output(tx, source_index[i]); });
}

/**
 * \brief Main kernel for the short items and epilogue of long items.
 * \tparam kPDL whether to use PDL to synchronize with the cluster kernel (if any)
 * \tparam kLevel:
 * - Level 0: max_seq_len <= 8192           -> trivial + register<2>
 * - Level 1: max_seq_len <= 16384          -> trivial + register<4>
 * - Level 2: max_seq_len <= cluster_floor  -> trivial + register<4> + streaming
 * - Level 3: max_seq_len > cluster_floor   -> + epilogue process of cluster path
 */
template <bool kPDL, int kLevel>
TOPK_KERNEL void topk_main_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDL>(problem);
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;

  constexpr bool kHandleCluster = (kLevel == 3);
  // non-trivial path: dispatch based on level and seq_len
  const auto cluster_threshold = kHandleCluster ? params.cluster_threshold() : kU32Max;
  if constexpr (kLevel == 0) {
    __builtin_assume(problem.seq_len <= kReg2MaxSeqLen);
    Register2::forward<kPDL>(problem, &smem);
  } else if constexpr (kLevel == 1) {
    __builtin_assume(problem.seq_len <= kReg4MaxSeqLen);
    Register4::forward<kPDL>(problem, &smem);  // max_seq_len <= 16384 guarantees seq <= 16384
  } else {
    static_assert(kLevel == 2 || kLevel == 3, "we only support level = 0,1,2,3 now");
    // if using cluster, we can delay the PDL wait
    constexpr bool kPDLEarly = kPDL && !kHandleCluster;
    constexpr bool kPDLFinal = kPDL && kHandleCluster;
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDLEarly>(problem, &smem);
    } else if (problem.seq_len <= cluster_threshold) {
      Streaming::forward<kPDLEarly>(problem, &smem);
    } else {  // cluster path do nothing here
      problem.out = params.get_output_ptr(blockIdx.x);
    }
    device::PDLWaitPrimary<kPDLFinal>();
  }

  // page-table transform pass (gathers kept out of the hot scatter loop),
  // then trigger the dependent kernel only after the full output is written.
  device::PDLTriggerSecondary<kPDL>();
  __syncthreads();
  problem_transform(problem, params.get_output_ptr(blockIdx.x));
}

using HbeStreaming = impl::TopKHbeStreaming;
using HbeCfg = impl::HbeConfig;

/// op29 HBE kernel: streaming regime only (max_seq_len > 16384, no cluster).
/// pre_idx: [B, topk] hint indices (per-row). Dynamic smem = candidate bufs.
template <bool kPDL>
TOPK_KERNEL void gvr29_hbe_kernel(const __grid_constant__ TopKLaunchParams params,
                                  const int32_t* __restrict__ pre_idx,
                                  impl::TieValue* __restrict__ spill) {
  // NOTE: no enable_smem_spilling() here — ptxas forbids the pragma in
  // kernels with dynamic SMEM (the HBE candidate buffers).
  auto problem = params.problem(blockIdx.x);
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  extern __shared__ uint8_t dyn_smem[];
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDL>(problem);
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;
  if (problem.seq_len <= kReg4MaxSeqLen) {
    // short rows keep the register path (already 1 read; hint gain marginal)
    if (problem.seq_len <= kReg2MaxSeqLen) {
      Register2::forward<kPDL>(problem, &smem);
    } else {
      Register4::forward<kPDL>(problem, &smem);
    }
  } else {
    const size_t spill_stride =
        HbeCfg::spill_bytes_per_row(problem.topk) / sizeof(impl::TieValue);
    HbeStreaming::forward<kPDL>(problem,
                                pre_idx + blockIdx.x * int64_t(problem.topk),
                                spill + blockIdx.x * spill_stride,
                                &smem, dyn_smem);
  }
  device::PDLTriggerSecondary<kPDL>();
  __syncthreads();
  problem_transform(problem, params.get_output_ptr(blockIdx.x));
}

template <bool kPDL>
CLUSTER_TOPK_KERNEL void topk_small_batch_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  __shared__ impl::MaxSmem<Streaming::Smem, Cluster::Smem> smem;
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDL>(problem);
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;

  // randomly elect one worker rank to avoid workload imbalance
  const auto worker_rank = blockIdx.x % kClusterSize;

  // for small batch, we will fuse in the cluster case
  if (problem.seq_len <= kReg4MaxSeqLen) {
    if (blockIdx.y == worker_rank) Register4::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= params.cluster_floor) {
    if (blockIdx.y == worker_rank) Streaming::forward<kPDL>(problem, &smem);
  } else {
    auto cluster = cooperative_groups::this_cluster();
    problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
    Cluster::forward<kPDL>(problem, &smem);  // write to peer's output shared memory
    cluster.sync();
  }

  device::PDLWaitPrimary<kPDL>();
  __syncthreads();
  if (blockIdx.y == worker_rank) problem_transform(problem, params.get_output_ptr(blockIdx.x));
}

// --- Plan: choose cluster_threshold from the seq_len distribution -----------
__global__ __launch_bounds__(kBlockSize, 1) void topk_plan(
    const uint32_t* __restrict__ seq_lens,
    PlanItem* __restrict__ metadata,  // [0]=GlobalMetadata, [1+i]=PlanItem
    const uint32_t batch_size,
    const uint32_t static_cluster_threshold) {
  // Candidate (threshold T_j, cap_j) pairs, T strictly increasing. The plan lowers
  // cluster_threshold to T_j while #(items with seq_len > T_j) <= cap_j, so cap_j
  // bounds how many long items go to the persistent pool. The pool runs N items in
  // ceil(N / kNumPersistentClusters) waves; the longer the seq the more waves pay
  // off (streaming a single block over a long item is very slow), so cap_j is the
  // measured cluster-vs-streaming crossover (B200, occ2) and GROWS with T -- a flat
  // cap = pool size only fits the shortest (~98K, one-wave) bucket. (Plan is tunable.)
  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  constexpr Pair kCandidates[] = {
      {65536, 30},    // (65536,98304]:    ~1 pool wave, streams beyond 30
      {98304, 48},    // (98304,131072]
      {131072, 60},   // (131072,196608]
      {196608, 80},   // (196608,262144]
      {262144, 112},  // (262144,393216]
      {393216, 128},  // (393216,inf):     longest -- worth many pool waves; a top
                      // threshold here lets overloaded ~280-393K batches still stream
  };
  constexpr uint32_t kNumCandidates = std::size(kCandidates);
  static_assert(kCandidates[0].threshold == kClusterFloor);

  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;
  __shared__ uint32_t s_count;

  const auto tx = threadIdx.x;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  if (tx == 0) s_count = 0;
  __syncthreads();

  if (static_cluster_threshold > 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else {
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (sl > kCandidates[j].threshold ? 1 : 0);
      }
      if (count > 0) atomicAdd(&s_counts[count - 1], 1);
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t accum = 0;
      uint32_t chosen = kCandidates[kNumCandidates - 1].threshold;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];  // # items with seq_len > kCandidates[j].threshold
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();
  const auto cluster_threshold = max(s_threshold, kClusterFloor);

  // Compact items with seq_len > threshold into metadata[1..N]: their batch ids
  // are the work list the persistent cluster pool fetches.
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t sl = seq_lens[i];
    if (sl > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, sl};
    }
  }
  __syncthreads();
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {.cluster_threshold = cluster_threshold, .num_cluster_items = s_count};
  }
}

}  // namespace

// ===========================================================================
// Torch-extension host layer (replaces upstream tvm-ffi TopKKernel struct;
// dispatch constants and LaunchKernel configs identical to topk_v2.cuh).
// ===========================================================================
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// plan: seq_lens [B] int32, metadata [B+1, 2] int32.
void topk_v2_plan(torch::Tensor seq_lens, torch::Tensor metadata,
                  int64_t static_cluster_threshold) {
  TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == torch::kInt32);
  TORCH_CHECK(metadata.scalar_type() == torch::kInt32 && metadata.size(1) == 2);
  const auto batch_size = static_cast<uint32_t>(seq_lens.size(0));
  TORCH_CHECK(metadata.size(0) == batch_size + 1, "invalid metadata shape");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  host::LaunchKernel(1, kBlockSize, stream)(
      topk_plan,
      reinterpret_cast<const uint32_t*>(seq_lens.data_ptr<int32_t>()),
      reinterpret_cast<PlanItem*>(metadata.data_ptr<int32_t>()),
      batch_size,
      static_cast<uint32_t>(static_cluster_threshold));
}

// transform: scores [B, C] fp32 (stride multiple of 4), seq_lens [B] int32,
// page_table [>=1] int32 (identity: [0]==0, stride 0), out [B, K] int32.
// Reproduces upstream TopKKernel::transform dispatch verbatim.
void topk_v2_transform(torch::Tensor scores, torch::Tensor seq_lens,
                       torch::Tensor page_table, torch::Tensor out,
                       torch::Tensor metadata, int64_t K, int64_t page_bits,
                       int64_t max_seq_len, torch::Tensor pre_idx,
                       bool use_hbe, torch::Tensor spill) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat32,
              "scores must be CUDA float32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
  TORCH_CHECK(out.scalar_type() == torch::kInt32);
  TORCH_CHECK(scores.stride(0) % 4 == 0,
              "score_stride must be a multiple of 4 (16-byte vectorized load)");
  const auto topk = static_cast<uint32_t>(K);
  TORCH_CHECK(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
  const auto batch_size = static_cast<uint32_t>(scores.size(0));

  // Upstream constants (topk_v2.cuh): B200 fits one wave of 15 8-block
  // clusters at occ2; batch <= 15 keeps the 32K small-batch floor.
  constexpr uint32_t kClusterFloorSmall = 32768;
  constexpr uint32_t kSmallBatchLowFloor = 15;
  const auto params = TopKLaunchParams{
      .scores = scores.data_ptr<float>(),
      .seq_lens = seq_lens.data_ptr<int32_t>(),
      .page_table = page_table.data_ptr<int32_t>(),
      .page_indices = out.data_ptr<int32_t>(),
      .raw_indices = nullptr,
      .metadata = reinterpret_cast<const PlanItem*>(metadata.data_ptr<int32_t>()),
      .score_stride = scores.stride(0),
      .page_table_stride = 0,  // identity page table shared by all rows
      .topk = topk,
      .page_bits = static_cast<uint32_t>(page_bits),
      .cluster_floor = (batch_size <= kSmallBatchLowFloor) ? kClusterFloorSmall : kClusterFloor,
  };

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // op29 HBE dispatch: streaming regime only (no cluster, rows > 16384).
  const bool cluster_eligible =
      (static_cast<uint32_t>(max_seq_len) > params.cluster_floor) && (batch_size <= kClusterMaxBatch);
  // iter10b guard REVERT to the proven domain (iter10 expansion falsified:
  // 65536x2048 0.63 [fixed per-CTA overheads vs short rows], K2048 0.56-0.88
  // [unattributed K-proportional cost — NCU next]): K <= 1024 && N >= 131072.
  if (use_hbe && !cluster_eligible && static_cast<int64_t>(K) <= 1024 &&
      static_cast<uint32_t>(max_seq_len) >= 131072) {
    TORCH_CHECK(pre_idx.scalar_type() == torch::kInt32 &&
                pre_idx.size(0) == batch_size &&
                pre_idx.size(1) == static_cast<int64_t>(topk),
                "pre_idx must be [B, K] int32");
    const size_t dyn = HbeCfg::dyn_smem_bytes(topk);
    static size_t configured_dyn = 0;
    if (dyn > configured_dyn) {
      cudaFuncSetAttribute(gvr29_hbe_kernel<true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(HbeCfg::dyn_smem_bytes(2048)));
      configured_dyn = HbeCfg::dyn_smem_bytes(2048);
    }
    TORCH_CHECK(spill.numel() * spill.element_size() >=
                    static_cast<int64_t>(batch_size) *
                        static_cast<int64_t>(HbeCfg::spill_bytes_per_row(topk)),
                "spill buffer too small");
    host::LaunchKernel(batch_size, kBlockSize, stream, dyn)
        .config({.use_pdl = true})
        .launch(gvr29_hbe_kernel<true>, params, pre_idx.data_ptr<int32_t>(),
                reinterpret_cast<impl::TieValue*>(spill.data_ptr()));
    return;
  }
  const bool use_cluster =
      (static_cast<uint32_t>(max_seq_len) > params.cluster_floor) && (batch_size <= kClusterMaxBatch);
  constexpr bool kUsePDL = true;
  if (use_cluster) {
    if (batch_size <= kNumPersistentClusters) {
      host::LaunchKernel({batch_size, kClusterSize}, kBlockSize, stream)
          .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
          .launch(topk_small_batch_kernel<kUsePDL>, params);
    } else {
      const uint32_t num_clusters = std::min(batch_size, kNumPersistentClusters);
      host::LaunchKernel({num_clusters, kClusterSize}, kBlockSize, stream)
          .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
          .launch(topk_persistent_cluster_kernel<kUsePDL>, params);
      host::LaunchKernel(batch_size, kBlockSize, stream)
          .config({.use_pdl = kUsePDL})
          .launch(topk_main_kernel<kUsePDL, /*kLevel=*/3>, params);
    }
  } else if (static_cast<uint32_t>(max_seq_len) <= kReg2MaxSeqLen) {
    host::LaunchKernel(batch_size, kBlockSize, stream)
        .config({.use_pdl = kUsePDL})
        .launch(topk_main_kernel<kUsePDL, /*kLevel=*/0>, params);
  } else if (static_cast<uint32_t>(max_seq_len) <= kReg4MaxSeqLen) {
    host::LaunchKernel(batch_size, kBlockSize, stream)
        .config({.use_pdl = kUsePDL})
        .launch(topk_main_kernel<kUsePDL, /*kLevel=*/1>, params);
  } else {
    host::LaunchKernel(batch_size, kBlockSize, stream)
        .config({.use_pdl = kUsePDL})
        .launch(topk_main_kernel<kUsePDL, /*kLevel=*/2>, params);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gvr29_plan", &topk_v2_plan,
        "SGLang v2 top-K plan kernel (cluster_threshold + long-item work list)");
  m.def("gvr29_transform", &topk_v2_transform,
        "SGLang v2 top-K transform (dispatch: register/streaming/cluster paths)");
}
