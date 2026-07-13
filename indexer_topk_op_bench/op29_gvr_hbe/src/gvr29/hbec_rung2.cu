// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// HBE-C rung-2 microbench (DESIGN_HBEC_HINT_LADDER.md §6): isolate the
// cluster serial-chain cost of the stock dense histogram all-reduce vs the
// HBE-C M×8-scalar reduce (+ sparse candidate mini-hist), at BS=1 (one
// 8-CTA cluster). Each mode runs `rounds` iterations of ONLY its reduce
// pattern; per-round latency = slope of t(rounds), so launch overhead and
// the persistent-pool entry cost cancel.
//
// Modes:
//   0  sync-only baseline: 2 cluster.sync per round
//   1  dense 1024-bin all-reduce (bit-faithful port of TopKCluster Phase
//      1.5: rank-partitioned bins, 1 bin/thread, DSMEM gather + warp
//      segment-reduce + scatter) + 2 cluster.sync
//   2  dense 4096-bin all-reduce (4 bins/thread variant) + 2 cluster.sync
//   3  M×8 scalar reduce (M=3 per-rung counts: 24 DSMEM reads on one warp,
//      segment-reduce, broadcast) + 2 cluster.sync
//   4  mode 3 + sparse candidate mini-hist: `cand_per_cta` DSMEM atomics
//      per CTA into the rank-0 4096-bin hist + 1 extra cluster.sync
//
// Build: torch cpp_extension, same flags as gvr29 (sm_100f).

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace cg = cooperative_groups;

#define RUNG2_KERNEL \
  __global__ __launch_bounds__(1024, 1) __cluster_dims__(1, 8, 1)

constexpr uint32_t kBlock = 1024;
constexpr uint32_t kCluster = 8;
constexpr uint32_t kDenseSmall = 1024;
constexpr uint32_t kDenseBig = 4096;
constexpr uint32_t kM = 3;

struct Smem {
  uint32_t hist[kDenseBig];
  uint32_t scalars[kM];
  uint32_t out;
};

template <uint32_t kWidth>
__device__ __forceinline__ uint32_t seg_reduce_sum(uint32_t v) {
#pragma unroll
  for (uint32_t d = kWidth / 2; d > 0; d /= 2)
    v += __shfl_xor_sync(0xffffffffu, v, d);
  return v;
}

RUNG2_KERNEL void rung2_kernel(int mode, int rounds, int cand_per_cta,
                               uint32_t* out) {
  __shared__ Smem smem;
  const auto cluster = cg::this_cluster();
  const uint32_t tx = threadIdx.x;
  const uint32_t rank = blockIdx.y;

  for (uint32_t i = tx; i < kDenseBig; i += kBlock) smem.hist[i] = tx ^ rank;
  if (tx < kM) smem.scalars[tx] = tx + rank;
  if (tx == 0) smem.out = 0;
  __syncthreads();
  cluster.sync();

  uint32_t acc = 0;
  for (int r = 0; r < rounds; ++r) {
    if (mode == 0) {
      cluster.sync();
      cluster.sync();
    } else if (mode == 1) {
      // TopKCluster Phase 1.5, kHist=1024: thread t owns
      // bin (rank*128 + t/8), peer (t%8)
      cluster.sync();
      constexpr uint32_t kPart = kDenseSmall / kCluster;
      const uint32_t which = rank * kPart + tx / kCluster;
      const uint32_t peer = tx % kCluster;
      auto addr = cluster.map_shared_rank(&smem.hist[which], peer);
      const uint32_t v = *addr;
      *addr = seg_reduce_sum<kCluster>(v);
      cluster.sync();
      acc += smem.hist[tx & (kDenseSmall - 1)];
    } else if (mode == 2) {
      // 4096-bin variant: 4 owned bins per thread
      cluster.sync();
      constexpr uint32_t kPart = kDenseBig / kCluster;  // 512 bins/rank
#pragma unroll
      for (uint32_t j = 0; j < kDenseBig / kDenseSmall; ++j) {
        const uint32_t slot = j * kBlock + tx;
        const uint32_t which = rank * kPart + slot / kCluster;
        const uint32_t peer = slot % kCluster;
        auto addr = cluster.map_shared_rank(&smem.hist[which], peer);
        const uint32_t v = *addr;
        *addr = seg_reduce_sum<kCluster>(v);
      }
      cluster.sync();
      acc += smem.hist[tx & (kDenseBig - 1)];
    } else if (mode >= 3) {
      // HBE-C C2: M×8 scalar gather on warp 0 of every CTA (redundant,
      // keeps every CTA's copy consistent with zero extra latency)
      cluster.sync();
      if (tx < 32) {  // full warp participates (shfl mask validity);
        const uint32_t m = tx / kCluster;      // segment 3 = padding read
        const uint32_t peer = tx % kCluster;
        auto addr = cluster.map_shared_rank(
            &smem.scalars[m < kM ? m : 0], peer);
        const uint32_t v = *addr;
        const uint32_t s = seg_reduce_sum<kCluster>(v);
        if (peer == 0 && m < kM) smem.hist[kDenseBig - 1 - m] = s;
      }
      cluster.sync();
      if (mode == 4) {
        // sparse candidate mini-hist: cand_per_cta atomics from each CTA
        // into rank-0's 4096-bin hist (worst-case: all remote for ranks>0)
        auto* h0 = cluster.map_shared_rank(&smem.hist[0], 0);
        for (uint32_t i = tx; i < (uint32_t)cand_per_cta; i += kBlock) {
          atomicAdd(h0 + ((i * 2654435761u + rank * 97u + r) &
                          (kDenseBig - 1)), 1u);
        }
        cluster.sync();
      } else if (mode == 5) {
        // REVISED C2: per-CTA LOCAL mini-hist build (cand_per_cta local
        // smem atomics, 1024 bins) + stock-style dense 1024-bin all-reduce
        for (uint32_t i = tx; i < (uint32_t)cand_per_cta; i += kBlock) {
          atomicAdd(&smem.hist[(i * 2654435761u + rank * 97u + r) &
                               (kDenseSmall - 1)], 1u);
        }
        cluster.sync();
        constexpr uint32_t kPart = kDenseSmall / kCluster;
        const uint32_t which = rank * kPart + tx / kCluster;
        const uint32_t peer = tx % kCluster;
        auto addr = cluster.map_shared_rank(&smem.hist[which], peer);
        const uint32_t v = *addr;
        *addr = seg_reduce_sum<kCluster>(v);
        cluster.sync();
      }
      acc += smem.hist[kDenseBig - 1];
    }
  }
  if (tx == 0) atomicAdd(&smem.out, acc + 1);
  __syncthreads();
  if (tx == 0 && rank == 0) out[blockIdx.x] = smem.out;
  cluster.sync();
}

void rung2_run(int mode, int rounds, int cand_per_cta, torch::Tensor out) {
  dim3 grid(1, kCluster, 1), block(kBlock, 1, 1);
  rung2_kernel<<<grid, block, 0,
                 at::cuda::getCurrentCUDAStream()>>>(
      mode, rounds, cand_per_cta, out.data_ptr<uint32_t>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &rung2_run, "rung2 microbench single launch");
}
