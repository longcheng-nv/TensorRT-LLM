/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/heuristicTopKDecode.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/envUtils.h"

// Import gvrTopKJob (__device__ __noinline__, the GVR micro-kernel) and
// all helpers. gvrTopKJob is independently optimized by ptxas, matching standalone
// SASS quality regardless of the caller's prologue code.
#include "tensorrt_llm/kernels/heuristic_topk.cuh"

#include <cfloat>
#include <cooperative_groups.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mutex>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

namespace cg = cooperative_groups;

using heuristic_topk::BLOCK_SIZE;
using heuristic_topk::GvrDtypeTraits;
using heuristic_topk::GvrParams;
using heuristic_topk::gvrTopKJob;
using heuristic_topk::gvrTopKJobDtype;
using heuristic_topk::KernelSmemTplK;
// MC kernel helpers
using heuristic_topk::blockCountGE;
using heuristic_topk::blockCountGEDtype;
using heuristic_topk::blockFusedSnapIter;
using heuristic_topk::blockFusedSnapIterDtype;
using heuristic_topk::MAX_REFINE_ITERS;
using heuristic_topk::NUM_WARPS;
using heuristic_topk::warpReduceMax;
using heuristic_topk::warpReduceMin;
using heuristic_topk::WARP_SIZE;

// Templated on TopK so the launcher can dispatch K=512/1024/2048 to the
// same kernel template. Smem layout is derived from GvrParams<float, TopK>
// at compile time.
template <int TopK>
__global__ void __launch_bounds__(BLOCK_SIZE) heuristicTopKMultiRowKernel(float const* __restrict__ logits,
    int const* __restrict__ seqLens, int const* __restrict__ preIdx, float* __restrict__ scratchValues,
    int* __restrict__ outIndices, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount)
{
    using SmemT = KernelSmemTplK<float, GvrParams<float, TopK>::kC, GvrParams<float, TopK>::kNumBins>;

    int const rowIdx = blockIdx.x;
    int const seq_len = seqLens[rowIdx / next_n];
    int const N = seq_len - next_n + (rowIdx % next_n) + 1;

    float const* __restrict__ input = logits + static_cast<int64_t>(rowIdx) * stride0;
    int const* __restrict__ rowPreIdx = preIdx + static_cast<int64_t>(rowIdx / next_n) * preIdxStride;
    float* __restrict__ outputValues = scratchValues + static_cast<int64_t>(rowIdx) * topK;
    int* __restrict__ outputIndices = outIndices + static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<SmemT*>(smem_raw);

    if (N <= topK)
    {
        int const tid = threadIdx.x;
        for (int i = tid; i < N; i += BLOCK_SIZE)
        {
            outputValues[i] = input[i];
            outputIndices[i] = i;
        }
        for (int i = N + tid; i < topK; i += BLOCK_SIZE)
        {
            outputValues[i] = -FLT_MAX;
            outputIndices[i] = -1;
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    // +1 accounts for the temporal shift: prev_topk indices were computed at
    // seq_len-1, but the current step has one additional KV token appended.
    int const preIdxOffset = (rowIdx % next_n) + 1;
    gvrTopKJob<TopK>(input, N, rowPreIdx, preIdxCount, topK, outputValues, outputIndices, smem, preIdxOffset);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ============================================================================
// Multi-dtype path (bf16 / fp16)
// ============================================================================
// Mirrors heuristicTopKMultiRowKernel for bf16/fp16 inputs. The kernel body
// is structurally identical; only the input/output dtype, the smem-key
// dtype, and the GVR job (gvrTopKJobDtype<InputT>) differ.

// Templated on (InputT, TopK). Smem layout is derived from
// GvrParams<InputT, TopK>.
template <typename InputT, int TopK>
__global__ void __launch_bounds__(BLOCK_SIZE) heuristicTopKMultiRowKernelDtype(InputT const* __restrict__ logits,
    int const* __restrict__ seqLens, int const* __restrict__ preIdx, InputT* __restrict__ scratchValues,
    int* __restrict__ outIndices, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount)
{
    // dtype path uses fp32 keys[] in smem (down-conversion deferred to writeback).
    using SmemT = KernelSmemTplK<float, GvrParams<InputT, TopK>::kC, GvrParams<InputT, TopK>::kNumBins>;

    int const rowIdx = blockIdx.x;
    int const seq_len = seqLens[rowIdx / next_n];
    int const N = seq_len - next_n + (rowIdx % next_n) + 1;

    InputT const* __restrict__ input = logits + static_cast<int64_t>(rowIdx) * stride0;
    int const* __restrict__ rowPreIdx = preIdx + static_cast<int64_t>(rowIdx / next_n) * preIdxStride;
    InputT* __restrict__ outputValues = scratchValues + static_cast<int64_t>(rowIdx) * topK;
    int* __restrict__ outputIndices = outIndices + static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<SmemT*>(smem_raw);

    if (N <= topK)
    {
        int const tid = threadIdx.x;
        for (int i = tid; i < N; i += BLOCK_SIZE)
        {
            outputValues[i] = input[i];
            outputIndices[i] = i;
        }
        InputT const neg_max = GvrDtypeTraits<InputT>::from_fp32(-FLT_MAX);
        for (int i = N + tid; i < topK; i += BLOCK_SIZE)
        {
            outputValues[i] = neg_max;
            outputIndices[i] = -1;
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    int const preIdxOffset = (rowIdx % next_n) + 1;
    gvrTopKJobDtype<InputT, TopK>(
        input, N, rowPreIdx, preIdxCount, topK, outputValues, outputIndices, smem, preIdxOffset);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Explicit instantiations — 6 (dtype × K) combos. Launchers dispatch on
// runtime topK via switch, so all 6 must be available at link time.
template __global__ void heuristicTopKMultiRowKernelDtype<__nv_bfloat16, 512>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__nv_bfloat16, 1024>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__nv_bfloat16, 2048>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__half, 512>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__half, 1024>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__half, 2048>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernel<512>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernel<1024>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernel<2048>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int);

// Dispatch on topK at runtime — each TopK-instantiation gets its own smem
// size (driven by GvrParams<InputT, TopK>::kC/kNumBins) and own kfn pointer
// (cudaFuncSetAttribute / cudaLaunchKernelEx target the right kernel).
//
// fp32 routes to heuristicTopKMultiRowKernel<TopK>; bf16/fp16 route to
// heuristicTopKMultiRowKernelDtype<InputT, TopK>. Vector-load alignment
// requirement is 4 elements for fp32 (float4) and 8 elements for bf16/fp16
// (int4 of 16-bit). In TRT-LLM the logits stride is always a multiple of
// tokens_per_block (≥64), so the alignment check is never hit at runtime
// — it's an assert against caller misuse.
template <typename InputT>
void launchHeuristicTopKDecodeImpl(InputT const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    InputT* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        topK == 512 || topK == 1024 || topK == 2048, "heuristicTopKDecode requires topK ∈ {512, 1024, 2048}");

    constexpr int kAlign = std::is_same_v<InputT, float> ? 4 : 8;
    TLLM_CHECK_WITH_INFO(stride0 % kAlign == 0 || numRows <= 1,
        "heuristicTopKDecode requires logits stride0 divisible by %d for multi-row launch", kAlign);

    auto launchOne = [&]<int TopK>()
    {
        // bf16/fp16 path also uses fp32 keys[] in smem (down-conversion deferred).
        using SmemT = KernelSmemTplK<float, GvrParams<InputT, TopK>::kC, GvrParams<InputT, TopK>::kNumBins>;
        size_t const smemSize = sizeof(SmemT);

        auto kfn = []()
        {
            if constexpr (std::is_same_v<InputT, float>)
                return heuristicTopKMultiRowKernel<TopK>;
            else
                return heuristicTopKMultiRowKernelDtype<InputT, TopK>;
        }();

        if (smemSize > 48u * 1024u)
        {
            cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
        }

        cudaLaunchConfig_t config;
        config.gridDim = numRows;
        config.blockDim = BLOCK_SIZE;
        config.dynamicSmemBytes = smemSize;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(&config, kfn, logits, seqLens, preIdx, scratchValues, outIndices, stride0, next_n, topK,
            preIdxStride, preIdxCount);
    };

    switch (topK)
    {
    case 512: launchOne.template operator()<512>(); break;
    case 1024: launchOne.template operator()<1024>(); break;
    case 2048: launchOne.template operator()<2048>(); break;
    default: TLLM_THROW("heuristicTopKDecode: topK validated above; unreachable");
    }
}

// ============================================================================
// Multi-CTA path (Q12, 2026-05-10) — Phase 1 first cut
// ============================================================================
// Three-kernel design:
//   Stage A: heuristicTopKStageAKernel<T,K> — gridDim = numRows × clusterSize.
//            Each block runs gvrTopKJob{,Dtype} on its chunk (chunk_N = N/clusterSize)
//            and writes top-K per chunk to staging[BS × clusterSize × K], with
//            indices shifted to row-global.
//   Stage B: heuristicTopKMergeKernel<T,K> — gridDim = numRows.
//            Each block reads staging[clusterSize × K] for its row, runs P4
//            (histogram + snap + 2-pass emit) on aggregated candidates,
//            writes final top-K to output. Indices already row-global from Stage A.
//
// Partition policy: chunk_min = 8 × K_topK (sweet spot per
// 10_multi_cta_v1/REPORT.md §3.4); cluster_max = 4 (sm_100 cluster supports up
// to 8 but 4 keeps wave_cap reasonable for BS≥1); smem_cap = kCC / K_topK to
// keep cluster × K candidates within Stage B's smem keys[] buffer.
//
// Reuses gvrTopKJob{,Dtype} unchanged — V2e LOCK byte-identity preserved for
// the SC production path (heuristicTopKMultiRowKernel{,Dtype}).
//
// Activation: TRTLLM_GVR_MC=1 env (default off; preserves SC dispatch).
//             TRTLLM_GVR_MC_CLUSTER=N override (0=auto, 1/2/4=force).
// ============================================================================

constexpr int kMcClusterMax = 4;

inline int computeMcClusterSize(int numRows, int numColumns, int topK, int kCC, int nCtaPerSm)
{
    char const* envOverride = std::getenv("TRTLLM_GVR_MC_CLUSTER");
    if (envOverride && envOverride[0] != '\0' && envOverride[0] != '0')
    {
        int v = atoi(envOverride);
        if (v >= 1 && v <= 8)
            return v;
    }
    constexpr int kSmCount = 148; // B200 sm_100
    int const totalCtaSlots = kSmCount * nCtaPerSm;
    int const waveCap = std::max(1, totalCtaSlots / std::max(numRows, 1));
    int const chunkMin = 8 * topK;
    int const chunkCap = std::max(1, numColumns / chunkMin);
    int const smemCap = std::max(1, kCC / topK);
    int cluster = std::min({waveCap, chunkCap, smemCap, kMcClusterMax});
    // Round down to power of 2.
    int p = 1;
    while ((p * 2) <= cluster)
        p *= 2;
    return std::max(1, p);
}

template <typename InputT, int TopK>
__global__ void __launch_bounds__(BLOCK_SIZE)
    heuristicTopKStageAKernel(InputT const* __restrict__ logits, int const* __restrict__ seqLens,
        int const* __restrict__ preIdx, InputT* __restrict__ stagingValues, int* __restrict__ stagingIndices,
        int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int clusterSize)
{
    using Params = GvrParams<InputT, TopK>;
    using SmemT = KernelSmemTplK<float, Params::kC, Params::kNumBins>;

    int const blkIdx = blockIdx.x;
    int const rowIdx = blkIdx / clusterSize;
    int const chunkIdx = blkIdx % clusterSize;
    int const tid = threadIdx.x;

    int const seq_len = seqLens[rowIdx / next_n];
    int const N = seq_len - next_n + (rowIdx % next_n) + 1;

    InputT const* row_input = logits + static_cast<int64_t>(rowIdx) * stride0;
    int const* row_preIdx = preIdx + static_cast<int64_t>(rowIdx / next_n) * preIdxStride;
    InputT* my_staging_v = stagingValues + static_cast<int64_t>(blkIdx) * topK;
    int* my_staging_i = stagingIndices + static_cast<int64_t>(blkIdx) * topK;

    InputT neg_max;
    if constexpr (std::is_same_v<InputT, float>)
        neg_max = -FLT_MAX;
    else
        neg_max = GvrDtypeTraits<InputT>::from_fp32(-FLT_MAX);

    if (N <= topK)
    {
        // Trivial passthrough: only chunk 0 emits real data; others emit -inf
        if (chunkIdx == 0)
        {
            for (int i = tid; i < N; i += BLOCK_SIZE)
            {
                my_staging_v[i] = row_input[i];
                my_staging_i[i] = i;
            }
            for (int i = N + tid; i < topK; i += BLOCK_SIZE)
            {
                my_staging_v[i] = neg_max;
                my_staging_i[i] = -1;
            }
        }
        else
        {
            for (int i = tid; i < topK; i += BLOCK_SIZE)
            {
                my_staging_v[i] = neg_max;
                my_staging_i[i] = -1;
            }
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    // Chunk boundaries — chunk_size rounded up to a multiple of 8 to preserve
    // 16-byte alignment for vector loads (float4 needs % 4 = 0; int4 of bf16/fp16
    // needs % 8 = 0; using 8 satisfies both). Last chunk gets the residual.
    constexpr int kAlign = 8;
    int const chunk_size_raw = (N + clusterSize - 1) / clusterSize;
    int const chunk_size = ((chunk_size_raw + kAlign - 1) / kAlign) * kAlign;
    int const chunk_start = chunkIdx * chunk_size;
    int const chunk_end = (chunk_start + chunk_size < N) ? (chunk_start + chunk_size) : N;
    int const chunk_N = (chunk_end > chunk_start) ? (chunk_end - chunk_start) : 0;

    if (chunk_N <= 0)
    {
        // Empty chunk (last chunk over-shoots); emit -inf
        for (int i = tid; i < topK; i += BLOCK_SIZE)
        {
            my_staging_v[i] = neg_max;
            my_staging_i[i] = -1;
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    if (chunk_N <= topK)
    {
        // Chunk smaller than topK: emit all chunk elements + pad
        for (int i = tid; i < chunk_N; i += BLOCK_SIZE)
        {
            my_staging_v[i] = row_input[chunk_start + i];
            my_staging_i[i] = chunk_start + i;
        }
        for (int i = chunk_N + tid; i < topK; i += BLOCK_SIZE)
        {
            my_staging_v[i] = neg_max;
            my_staging_i[i] = -1;
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    // V3.2 decode preIdxOffset, shifted to chunk-local frame
    int const preIdxOffsetChunk = (rowIdx % next_n) + 1 - chunk_start;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<SmemT*>(smem_raw);

    // Run gvrTopKJob{,Dtype} on the chunk → my_staging_v / my_staging_i with
    // chunk-relative indices in [0, chunk_N)
    if constexpr (std::is_same_v<InputT, float>)
    {
        gvrTopKJob<TopK>(row_input + chunk_start, chunk_N, row_preIdx, preIdxCount, topK, my_staging_v, my_staging_i,
            smem, preIdxOffsetChunk);
    }
    else
    {
        gvrTopKJobDtype<InputT, TopK>(row_input + chunk_start, chunk_N, row_preIdx, preIdxCount, topK, my_staging_v,
            my_staging_i, smem, preIdxOffsetChunk);
    }

    __syncthreads();

    // Shift indices to row-global (chunk-relative + chunk_start)
    for (int i = tid; i < topK; i += BLOCK_SIZE)
    {
        int const idx = my_staging_i[i];
        if (idx >= 0)
            my_staging_i[i] = idx + chunk_start;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputT, int TopK>
__global__ void __launch_bounds__(BLOCK_SIZE)
    heuristicTopKMergeKernel(InputT const* __restrict__ stagingValues, int const* __restrict__ stagingIndices,
        int clusterSize, InputT* __restrict__ outputValues, int* __restrict__ outputIndices, int topK)
{
    using Params = GvrParams<InputT, TopK>;
    using SmemT = KernelSmemTplK<float, Params::kC, Params::kNumBins>;

    constexpr int kK = TopK;
    constexpr int kCC = Params::kC;
    constexpr int kBins = Params::kNumBins;

    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid & (WARP_SIZE - 1);
    unsigned const full_mask = 0xffffffffu;
    int const rowIdx = blockIdx.x;

    int const M = clusterSize * topK; // total candidates for this row
    InputT const* row_staging_v = stagingValues + static_cast<int64_t>(rowIdx) * clusterSize * topK;
    int const* row_staging_i = stagingIndices + static_cast<int64_t>(rowIdx) * clusterSize * topK;
    InputT* row_out_v = outputValues + static_cast<int64_t>(rowIdx) * topK;
    int* row_out_i = outputIndices + static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<SmemT*>(smem_raw);

    // Load staging into smem keys[] (fp32) + vals[] (row-global indices)
    if constexpr (std::is_same_v<InputT, float>)
    {
        for (int i = tid; i < M; i += BLOCK_SIZE)
        {
            smem->keys[i] = row_staging_v[i];
            smem->vals[i] = row_staging_i[i];
        }
    }
    else
    {
        for (int i = tid; i < M; i += BLOCK_SIZE)
        {
            smem->keys[i] = GvrDtypeTraits<InputT>::to_fp32(row_staging_v[i]);
            smem->vals[i] = row_staging_i[i];
        }
    }
    __syncthreads();

    int const cand_count = M;
    InputT neg_max;
    if constexpr (std::is_same_v<InputT, float>)
        neg_max = -FLT_MAX;
    else
        neg_max = GvrDtypeTraits<InputT>::from_fp32(-FLT_MAX);

    // Trivial paths
    if (cand_count == kK)
    {
        for (int i = tid; i < kK; i += BLOCK_SIZE)
        {
            if constexpr (std::is_same_v<InputT, float>)
                row_out_v[i] = smem->keys[i];
            else
                row_out_v[i] = GvrDtypeTraits<InputT>::from_fp32(smem->keys[i]);
            row_out_i[i] = smem->vals[i];
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }
    if (cand_count < kK)
    {
        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            if constexpr (std::is_same_v<InputT, float>)
                row_out_v[i] = smem->keys[i];
            else
                row_out_v[i] = GvrDtypeTraits<InputT>::from_fp32(smem->keys[i]);
            row_out_i[i] = smem->vals[i];
        }
        for (int i = cand_count + tid; i < kK; i += BLOCK_SIZE)
        {
            row_out_v[i] = neg_max;
            row_out_i[i] = -1;
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    // cand_count > kK: histogram-based selection (mirrors gvrTopKJob P4 lines 932-1115)
    {
        float cmin = FLT_MAX, cmax = -FLT_MAX;
        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            float v = smem->keys[i];
            cmin = fminf(cmin, v);
            cmax = fmaxf(cmax, v);
        }
        cmin = warpReduceMin(cmin);
        cmax = warpReduceMax(cmax);
        if (lane == 0)
        {
            smem->warp_counts[warp_id] = __float_as_int(cmin);
            smem->histogram[warp_id] = __float_as_int(cmax);
        }
        __syncthreads();

        float block_min = FLT_MAX, block_max = -FLT_MAX;
        for (int w = 0; w < NUM_WARPS; w++)
        {
            block_min = fminf(block_min, __int_as_float(smem->warp_counts[w]));
            block_max = fmaxf(block_max, __int_as_float(smem->histogram[w]));
        }
        if (block_max <= block_min)
            block_max = block_min + 1e-6f;

        for (int i = tid; i < kBins; i += BLOCK_SIZE)
            smem->histogram[i] = 0;
        __syncthreads();

        float range1 = block_max - block_min;
        float inv1 = (range1 > 0.0f) ? ((float) (kBins - 1) + 0.99f) / range1 : 0.0f;

        for (int i = tid; i < cand_count; i += BLOCK_SIZE)
        {
            int bin = (int) ((smem->keys[i] - block_min) * inv1);
            bin = min(max(bin, 0), kBins - 1);
            atomicAdd(&smem->histogram[bin], 1);
        }
        __syncthreads();

        // K-th bin search (mirrors gvrTopKJob 3-step search)
        {
            constexpr int BINS_PER_WARP = kBins / NUM_WARPS;
            static_assert(kBins % NUM_WARPS == 0, "kBins must be divisible by NUM_WARPS");
            int warp_bin_sum = 0;
            for (int j = 0; j < BINS_PER_WARP; j++)
                warp_bin_sum += smem->histogram[kBins - 1 - warp_id * BINS_PER_WARP - j];
            if (lane == 0)
                smem->warp_counts[warp_id] = warp_bin_sum;
        }
        __syncthreads();

        if (tid == 0)
        {
            int cum = 0, tw = NUM_WARPS - 1;
            for (int w = 0; w < NUM_WARPS; w++)
            {
                cum += smem->warp_counts[w];
                if (cum >= kK)
                {
                    tw = w;
                    break;
                }
            }
            cum = 0;
            for (int w = 0; w < tw; w++)
                cum += smem->warp_counts[w];
            smem->cnt_lo = cum;
            smem->cnt_hi = tw;
        }
        __syncthreads();

        if (warp_id == smem->cnt_hi && lane == 0)
        {
            constexpr int BINS_PER_WARP = kBins / NUM_WARPS;
            int base_cum = smem->cnt_lo;
            float thr = block_min;
            for (int j = 0; j < BINS_PER_WARP; j++)
            {
                int b = kBins - 1 - smem->cnt_hi * BINS_PER_WARP - j;
                base_cum += smem->histogram[b];
                if (base_cum >= kK)
                {
                    thr = block_min + (float) b * range1 / (float) kBins;
                    break;
                }
            }
            smem->threshold = thr;
        }
        __syncthreads();

        bool snap_converged = false;
        int snap_limit = (cand_count > 128 ? cand_count / 4 : 32);
        for (int si = 0; si < snap_limit; si++)
        {
            blockFusedSnapIter<TopK>(smem, cand_count, tid, warp_id, lane);
            int cge = smem->cnt_lo;
            int cgt = smem->cnt_hi;
            if (cgt < kK && cge >= kK)
            {
                snap_converged = true;
                break;
            }
        }
        (void) snap_converged;

        float sel_thr = smem->threshold;
        if (tid == 0)
            smem->out_count = 0;
        __syncthreads();

        // Pass 1: strictly greater than sel_thr
        for (int base = warp_id * WARP_SIZE; base < cand_count; base += BLOCK_SIZE)
        {
            int i = base + lane;
            float v = (i < cand_count) ? smem->keys[i] : -FLT_MAX;
            bool emit_gt = (i < cand_count) && (v > sel_thr);
            unsigned mask_gt = __ballot_sync(full_mask, emit_gt);
            if (mask_gt)
            {
                int cnt = __popc(mask_gt);
                int moff = __popc(mask_gt & ((1u << lane) - 1u));
                int bp = 0;
                if (lane == 0)
                    bp = atomicAdd(&smem->out_count, cnt);
                bp = __shfl_sync(full_mask, bp, 0);
                if (emit_gt && bp + moff < kK)
                {
                    if constexpr (std::is_same_v<InputT, float>)
                        row_out_v[bp + moff] = v;
                    else
                        row_out_v[bp + moff] = GvrDtypeTraits<InputT>::from_fp32(v);
                    row_out_i[bp + moff] = smem->vals[i]; // already row-global
                }
            }
        }
        __syncthreads();

        // Pass 2: equal to sel_thr (fills remaining slots)
        for (int base = warp_id * WARP_SIZE; base < cand_count; base += BLOCK_SIZE)
        {
            int i = base + lane;
            float v = (i < cand_count) ? smem->keys[i] : -FLT_MAX;
            bool emit_eq = (i < cand_count) && (v == sel_thr);
            unsigned mask_eq = __ballot_sync(full_mask, emit_eq);
            if (mask_eq)
            {
                int cnt = __popc(mask_eq);
                int moff = __popc(mask_eq & ((1u << lane) - 1u));
                int bp = 0;
                if (lane == 0)
                    bp = atomicAdd(&smem->out_count, cnt);
                bp = __shfl_sync(full_mask, bp, 0);
                if (emit_eq && bp + moff < kK)
                {
                    if constexpr (std::is_same_v<InputT, float>)
                        row_out_v[bp + moff] = v;
                    else
                        row_out_v[bp + moff] = GvrDtypeTraits<InputT>::from_fp32(v);
                    row_out_i[bp + moff] = smem->vals[i];
                }
            }
        }
        __syncthreads();

        int filled = min(smem->out_count, kK);
        for (int i = filled + tid; i < kK; i += BLOCK_SIZE)
        {
            row_out_v[i] = neg_max;
            row_out_i[i] = -1;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ============================================================================
// Cooperative Multi-CTA path (Q12 v2.1, 2026-05-10) — single-kernel cluster design
// ============================================================================
// Differs from v2 (Stage A + Stage B) in three structural axes:
//   1. **One kernel launch** per row-batch instead of two — saves ~5 µs of
//      cudaEvent-measured launch overhead at small BS.
//   2. **Cooperative P2 secant** — all CTAs in a cluster contribute to a global
//      count via cluster atomicAdd; rank-0 leader updates the threshold; peers
//      broadcast-read it for the next iter. This *fixes* the v2 catastrophic
//      regression on long-tail layers (L42, KS=0.0245) where chunk-local
//      secant converges to chunk's K-th value instead of row's K-th value
//      → cand_count blows up → P4 snap runs full limit → 600+ µs.
//   3. **DSM-staged candidate buffer** — P3 collect scatters into rank-0's
//      keys[]/vals[] via cluster-scope atomic write_base; eliminates the v2
//      global staging buffer + HBM round-trip.
//
// Activation: TRTLLM_GVR_MC_COOP=1 env (default off; preserves SC dispatch).
//             TRTLLM_GVR_MC_COOP_CLUSTER=N override (0=auto, 2/4=force).
//
// Cluster size selection (auto): same chunk_min=8·topK / smem_cap=kCC/topK
// rules as v2; cluster ∈ {2, 4}. K=512/1024 → cluster=4 (smem_cap=10/5),
// K=2048 → cluster=2 (smem_cap=2-3).
//
// Smem layout (per CTA):
//   [SmemT (~35-59 KB)][ClusterScratch (16 B)]
// Only rank-0's ClusterScratch is read/written via DSM by all CTAs; peers'
// ClusterScratch is unused but allocated for layout uniformity.
// ============================================================================

struct ClusterScratch
{
    int g_total_cnt;  // P2: cluster-wide blockCountGE total
    int g_write_base; // P3: cluster-wide collect cursor
    int g_done;       // P2: convergence broadcast
    int g_pad;        // alignment
};

// Helper: cluster atomicAdd compiles to atom.shared::cluster.add on sm_100+
// when the address is DSM-resident.

template <typename InputT, int TopK, int kClusterSize>
__global__ void __cluster_dims__(kClusterSize, 1, 1) __launch_bounds__(BLOCK_SIZE)
    heuristicTopKCoopKernel(InputT const* __restrict__ logits, int const* __restrict__ seqLens,
        int const* __restrict__ preIdx, InputT* __restrict__ outputValues, int* __restrict__ outIndices, int stride0,
        int next_n, int topK, int preIdxStride, int preIdxCount)
{
    using Trait = GvrDtypeTraits<InputT>;
    using Params = GvrParams<InputT, TopK>;
    // bf16/fp16 path also stores fp32 keys[] (down-conversion deferred to writeback).
    using SmemT = KernelSmemTplK<float, Params::kC, Params::kNumBins>;
    constexpr int kK = TopK;
    constexpr int kCC = Params::kC;
    constexpr int kBins = Params::kNumBins;
    constexpr int kFTarget = Params::kFTarget;
    constexpr bool kIsFp32 = std::is_same_v<InputT, float>;

    cg::cluster_group cluster = cg::this_cluster();
    int const rank = cluster.block_rank();
    int const tid = threadIdx.x;
    int const warp_id = tid / WARP_SIZE;
    int const lane = tid & (WARP_SIZE - 1);
    unsigned const full_mask = 0xffffffffu;

    // gridDim.x = numRows × kClusterSize; row id = blockIdx.x / kClusterSize.
    int const blkIdx = blockIdx.x;
    int const rowIdx = blkIdx / kClusterSize;
    int const seq_len = seqLens[rowIdx / next_n];
    int const N = seq_len - next_n + (rowIdx % next_n) + 1;

    InputT const* __restrict__ row_input = logits + static_cast<int64_t>(rowIdx) * stride0;
    int const* __restrict__ row_preIdx = preIdx + static_cast<int64_t>(rowIdx / next_n) * preIdxStride;
    InputT* __restrict__ row_out_v = outputValues + static_cast<int64_t>(rowIdx) * topK;
    int* __restrict__ row_out_i = outIndices + static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    SmemT* smem = reinterpret_cast<SmemT*>(smem_raw);
    ClusterScratch* my_cs = reinterpret_cast<ClusterScratch*>(smem_raw + sizeof(SmemT));
    // DSM views of rank-0's smem + scratch (peers redirect their reads/writes here).
    SmemT* dsm0 = cluster.map_shared_rank(smem, 0);
    ClusterScratch* cs0 = cluster.map_shared_rank(my_cs, 0);

    InputT neg_max;
    if constexpr (kIsFp32)
        neg_max = -FLT_MAX;
    else
        neg_max = Trait::from_fp32(-FLT_MAX);

    // ---- Trivial passthrough (N <= topK): rank-0 emits, peers idle ----
    if (N <= topK)
    {
        if (rank == 0)
        {
            for (int i = tid; i < N; i += BLOCK_SIZE)
            {
                row_out_v[i] = row_input[i];
                row_out_i[i] = i;
            }
            for (int i = N + tid; i < topK; i += BLOCK_SIZE)
            {
                row_out_v[i] = neg_max;
                row_out_i[i] = -1;
            }
        }
        cluster.sync();
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    // V3.2 decode preIdxOffset (global frame; same value across all CTAs).
    int const preIdxOffset = (rowIdx % next_n) + 1;

    // Chunk boundaries — chunk_size rounded to 8 for vector-load alignment.
    constexpr int kAlign = 8;
    int const chunk_size_raw = (N + kClusterSize - 1) / kClusterSize;
    int const chunk_size = ((chunk_size_raw + kAlign - 1) / kAlign) * kAlign;
    int const chunk_start = rank * chunk_size;
    int const chunk_end = (chunk_start + chunk_size < N) ? (chunk_start + chunk_size) : N;
    int const chunk_N = (chunk_end > chunk_start) ? (chunk_end - chunk_start) : 0;

    // ================================================================
    // Phase 1 — preIdx Min/Max/Mean (redundant per CTA; M=K small,
    // cheaper than DSM broadcast). Each CTA computes the same threshold
    // independently from preIdx[]. After cluster.sync, all CTAs have the
    // same initial threshold value in their local smem; rank-0's value is
    // the canonical state for P2 cooperative state (val_lo/val_hi/cnt_lo/
    // cnt_hi/threshold).
    // ================================================================
    {
        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;
        float local_sum = 0.0f;
        int local_cnt = 0;
        for (int i = tid; i < preIdxCount; i += BLOCK_SIZE)
        {
            int idx = __ldg(&row_preIdx[i]) + preIdxOffset;
            if (idx >= 0 && idx < N)
            {
                float v;
                if constexpr (kIsFp32)
                    v = __ldg(&row_input[idx]);
                else
                    v = Trait::to_fp32(__ldg(&row_input[idx]));
                local_min = fminf(local_min, v);
                local_max = fmaxf(local_max, v);
                local_sum += v;
                local_cnt++;
            }
        }

        float wmin = warpReduceMin(local_min);
        float wmax = warpReduceMax(local_max);
        float wsum = local_sum;
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            wsum += __shfl_down_sync(full_mask, wsum, off);
        int wcnt;
#if __CUDA_ARCH__ >= 800
        wcnt = __reduce_add_sync(full_mask, local_cnt);
#else
        wcnt = local_cnt;
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            wcnt += __shfl_down_sync(full_mask, wcnt, off);
#endif
        if (lane == 0)
        {
            smem->histogram[warp_id] = __float_as_int(wmin);
            smem->histogram[NUM_WARPS + warp_id] = __float_as_int(wmax);
            smem->histogram[NUM_WARPS * 2 + warp_id] = __float_as_int(wsum);
            smem->histogram[NUM_WARPS * 3 + warp_id] = wcnt;
        }
        __syncthreads();

        if (tid == 0)
        {
            float pmin = FLT_MAX, pmax = -FLT_MAX, psum = 0.0f;
            int pcnt = 0;
            for (int w = 0; w < NUM_WARPS; w++)
            {
                pmin = fminf(pmin, __int_as_float(smem->histogram[w]));
                pmax = fmaxf(pmax, __int_as_float(smem->histogram[NUM_WARPS + w]));
                psum += __int_as_float(smem->histogram[NUM_WARPS * 2 + w]);
                pcnt += smem->histogram[NUM_WARPS * 3 + w];
            }
            float pmean = (pcnt > 0) ? psum / (float) pcnt : (pmin + pmax) * 0.5f;

            smem->pmax_saved = pmax;
            smem->threshold = pmean;
            smem->val_lo = pmin;
            smem->val_hi = pmax;
            smem->cnt_lo = preIdxCount + preIdxCount / 4;
            smem->cnt_hi = 1;
            smem->done = 0;
            // rank-0 also primes ClusterScratch
            if (rank == 0)
            {
                my_cs->g_total_cnt = 0;
                my_cs->g_write_base = 0;
                my_cs->g_done = 0;
            }
        }
        __syncthreads();
    }

    // Degenerate input check (val_hi == -inf or val_lo >= val_hi):
    // rank-0 emits trivial output; peers wait.
    if (smem->val_hi <= -FLT_MAX || smem->val_lo >= smem->val_hi)
    {
        if (rank == 0 && tid == 0)
        {
            for (int i = 0; i < topK && i < N; i++)
            {
                row_out_i[i] = i;
                if constexpr (kIsFp32)
                    row_out_v[i] = row_input[i];
                else
                    row_out_v[i] = row_input[i];
            }
        }
        cluster.sync();
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    cluster.sync(); // [SYNC-P1→P2] P1 results + g_total_cnt=0 visible to all CTAs

    // ================================================================
    // Phase 2 — Cooperative secant threshold search (v2.2 sync-merged)
    //
    // v2.2 reduces sync count vs v2.1:
    //   - Pre-loop: 1 sync (cnt only); SYNC-P2-init removed (P1 already
    //     reset g_total_cnt and SYNC-P1→P2 propagated it)
    //   - Per loop iter: 2 sync (broadcast + cnt); SYNC-P2-leader merged
    //     into next iter's leader-compute (done==1 → leader skips compute,
    //     peers see done after broadcast sync)
    //
    // Per Q5e (CCDF simulation, 18 216 rows): P2 mean iter=2.13, max=6.
    // Sync count: v2.1 ~9.4 → v2.2 ~5.3, saves ~4 cluster.sync × 0.255 µs
    // = ~1.0 µs / 6% wall on BS=1 bf16 K=512.
    // ================================================================

    // Initial blockCountGE on chunk_N before secant loop.
    // (g_total_cnt was zeroed in P1; no extra reset needed.)

    if (chunk_N > 0)
    {
        if constexpr (kIsFp32)
            blockCountGE<SmemT>(row_input + chunk_start, chunk_N, smem->threshold, smem, tid, warp_id, lane);
        else
            blockCountGEDtype<InputT, SmemT>(
                row_input + chunk_start, chunk_N, smem->threshold, smem, tid, warp_id, lane);
    }
    else
    {
        // Empty chunk (last chunk over-shoots); per_thread_counts cleared, cand_count=0.
        smem->per_thread_counts[tid] = 0;
        if (lane == 0)
            smem->warp_counts[warp_id] = 0;
        __syncthreads();
        if (tid == 0)
            smem->cand_count = 0;
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(&cs0->g_total_cnt, smem->cand_count);
    cluster.sync(); // [SYNC-P2-cnt-pre] leader sees initial total

    // Pre-loop state update — leader only; no sync (peers will see done after
    // first iter's broadcast sync).
    if (rank == 0 && tid == 0)
    {
        int total = my_cs->g_total_cnt;
        if (total >= kK && total <= kCC)
        {
            smem->done = 1;
            my_cs->g_done = 1;
        }
        else if (total > kCC)
        {
            smem->val_lo = smem->threshold;
            smem->cnt_lo = total;
        }
        else
        {
            smem->val_hi = smem->threshold;
            smem->cnt_hi = total;
        }
    }

    for (int iter = 0; iter < MAX_REFINE_ITERS; iter++)
    {
        // (1) Leader computes new threshold (only if not yet done).
        if (rank == 0 && tid == 0 && smem->done == 0)
        {
            float vlo = smem->val_lo, vhi = smem->val_hi;
            int clo = smem->cnt_lo, chi = smem->cnt_hi;
            float range = vhi - vlo;
            float nv;
            if (clo > chi && range > 1e-10f)
            {
                float f = (float) (clo - kFTarget) / (float) (clo - chi);
                f = fmaxf(0.05f, fminf(0.95f, f));
                if (iter == 0)
                    f = fminf(f, 0.50f);
                nv = vlo + range * f;
            }
            else
                nv = (vlo + vhi) * 0.5f;
            if (nv <= vlo)
                nv = vlo + range * 0.05f;
            if (nv >= vhi)
                nv = vhi - range * 0.05f;
            if (nv == vlo || nv == vhi)
            {
                nv = (vlo + vhi) * 0.5f;
                if (nv == vlo || nv == vhi)
                {
                    smem->threshold = vlo;
                    smem->done = 2;
                    my_cs->g_done = 2;
                }
                else
                    smem->threshold = nv;
            }
            else
                smem->threshold = nv;
            // reset counter for next blockCountGE
            my_cs->g_total_cnt = 0;
        }
        cluster.sync(); // [SYNC-P2-broadcast] peers see done flag + new threshold

        // (2) All CTAs check done (peers via DSM).
        if ((rank == 0 ? smem->done : dsm0->done) != 0)
            break;

        // (3) Peers copy threshold for next blockCountGE
        if (rank != 0 && tid == 0)
            smem->threshold = dsm0->threshold;
        __syncthreads();

        // (4) blockCountGE on chunk_N with new threshold
        if (chunk_N > 0)
        {
            if constexpr (kIsFp32)
                blockCountGE<SmemT>(row_input + chunk_start, chunk_N, smem->threshold, smem, tid, warp_id, lane);
            else
                blockCountGEDtype<InputT, SmemT>(
                    row_input + chunk_start, chunk_N, smem->threshold, smem, tid, warp_id, lane);
        }
        else
        {
            smem->per_thread_counts[tid] = 0;
            if (lane == 0)
                smem->warp_counts[warp_id] = 0;
            __syncthreads();
            if (tid == 0)
                smem->cand_count = 0;
            __syncthreads();
        }
        if (tid == 0)
            atomicAdd(&cs0->g_total_cnt, smem->cand_count);
        cluster.sync(); // [SYNC-P2-cnt-iter] leader sees iter total

        // (5) Leader state update — no sync (next iter's broadcast covers it,
        // OR loop exits and post-loop fixup handles it).
        if (rank == 0 && tid == 0)
        {
            int total = my_cs->g_total_cnt;
            if (total >= kK && total <= kCC)
            {
                smem->done = 1;
                my_cs->g_done = 1;
            }
            else if (total > kCC)
            {
                smem->val_lo = smem->threshold;
                smem->cnt_lo = total;
            }
            else
            {
                smem->val_hi = smem->threshold;
                smem->cnt_hi = total;
            }
        }
    }

    // Force-converge fallback (mirrors SC behaviour)
    if (rank == 0 && tid == 0 && !smem->done)
    {
        if (smem->cnt_lo <= kCC * 2)
            smem->threshold = smem->val_lo;
        else
            smem->threshold = smem->val_hi;
        smem->done = 2;
        my_cs->g_done = 2;
    }
    cluster.sync(); // [SYNC-P2→P3]

    // Peers sync threshold for P3
    if (rank != 0 && tid == 0)
        smem->threshold = dsm0->threshold;
    __syncthreads();

    // ================================================================
    // Phase 3 — Cooperative collect into rank-0's keys[]/vals[]
    // Each CTA scans chunk_N, computes per-thread qualifier counts (using
    // cached per_thread_counts from last blockCountGE), block-prefix-sums,
    // claims a global write base via cluster atomicAdd on cs0->g_write_base,
    // then each thread DSM-scatters its qualifying values to dsm0->keys[]
    // and dsm0->vals[].
    // ================================================================

    // The last blockCountGE was at smem->threshold during P2 (or P2-fallback).
    // If P2 ended with smem->done==1 (count in [kK, kCC]) we can skip recount.
    // If P2 ended with done==2 (force-converge), the count may exceed kCC at
    // smem->val_lo; rank-0 may need a binary-bisect retry. For simplicity v2.1
    // does the recount only if necessary, mirroring SC's done-skip optimization.

    // Re-count at final threshold (one extra blockCountGE if done==2 path).
    bool need_recount = false;
    if (rank == 0)
        need_recount = (smem->done == 2);
    if (rank != 0)
        need_recount = (dsm0->done == 2);
    if (need_recount)
    {
        if (rank == 0 && tid == 0)
            my_cs->g_total_cnt = 0;
        cluster.sync();
        if (chunk_N > 0)
        {
            if constexpr (kIsFp32)
                blockCountGE<SmemT>(row_input + chunk_start, chunk_N, smem->threshold, smem, tid, warp_id, lane);
            else
                blockCountGEDtype<InputT, SmemT>(
                    row_input + chunk_start, chunk_N, smem->threshold, smem, tid, warp_id, lane);
        }
        else
        {
            smem->per_thread_counts[tid] = 0;
            if (lane == 0)
                smem->warp_counts[warp_id] = 0;
            __syncthreads();
            if (tid == 0)
                smem->cand_count = 0;
            __syncthreads();
        }
        if (tid == 0)
            atomicAdd(&cs0->g_total_cnt, smem->cand_count);
        cluster.sync();
        // If overflow, rank-0 raises threshold to val_lo (guaranteed cnt > kCC)
        // and re-runs once. Bounded retry to avoid infinite loop.
        for (int retry = 0; retry < 3; retry++)
        {
            int total = (rank == 0) ? my_cs->g_total_cnt : cs0->g_total_cnt;
            if (total <= kCC)
                break;
            if (rank == 0 && tid == 0)
            {
                float lo = smem->val_lo, hi = smem->val_hi;
                float mid = (lo + hi) * 0.5f;
                if (mid == lo)
                    mid = hi;
                smem->threshold = mid;
                my_cs->g_total_cnt = 0;
            }
            cluster.sync();
            if (rank != 0 && tid == 0)
                smem->threshold = dsm0->threshold;
            __syncthreads();
            if (chunk_N > 0)
            {
                if constexpr (kIsFp32)
                    blockCountGE<SmemT>(row_input + chunk_start, chunk_N, smem->threshold, smem, tid, warp_id, lane);
                else
                    blockCountGEDtype<InputT, SmemT>(
                        row_input + chunk_start, chunk_N, smem->threshold, smem, tid, warp_id, lane);
            }
            else
            {
                smem->per_thread_counts[tid] = 0;
                if (lane == 0)
                    smem->warp_counts[warp_id] = 0;
                __syncthreads();
                if (tid == 0)
                    smem->cand_count = 0;
                __syncthreads();
            }
            if (tid == 0)
                atomicAdd(&cs0->g_total_cnt, smem->cand_count);
            cluster.sync();
        }
    }

    // Per-thread qualifier count from cached per_thread_counts (last blockCountGE).
    int my_total_qual = (chunk_N > 0) ? smem->per_thread_counts[tid] : 0;

    // Warp prefix scan
    int thread_prefix = my_total_qual;
#pragma unroll
    for (int off = 1; off < WARP_SIZE; off *= 2)
    {
        int other = __shfl_up_sync(full_mask, thread_prefix, off);
        if (lane >= off)
            thread_prefix += other;
    }
    int my_excl_offset = thread_prefix - my_total_qual;
    int warp_total_qual = __shfl_sync(full_mask, thread_prefix, WARP_SIZE - 1);

    if (lane == 0)
        smem->warp_counts[warp_id] = warp_total_qual;
    __syncthreads();

    // CTA-local prefix over warps; tid=0 also claims a cluster-wide base.
    int my_cta_total = 0;
    if (tid == 0)
    {
        int total = 0;
        for (int w = 0; w < NUM_WARPS; w++)
        {
            int cnt = smem->warp_counts[w];
            smem->warp_counts[w] = total;
            total += cnt;
        }
        my_cta_total = total;
        // Claim cluster-wide write base (atomic returns pre-add value).
        int my_base = atomicAdd(&cs0->g_write_base, total);
        smem->cand_count = my_base; // overload: store CTA's base offset here
    }
    __syncthreads();
    int my_cta_base = smem->cand_count;
    int my_write_pos_global = my_cta_base + smem->warp_counts[warp_id] + my_excl_offset;

    // Scatter to rank-0's keys[]/vals[] via DSM.
    {
        float const thr = smem->threshold;
        if (chunk_N > 0)
        {
            if constexpr (kIsFp32)
            {
                int const ci_end = chunk_start + (chunk_N & ~3);
                for (int i = chunk_start + tid * 4; i + 3 < ci_end; i += BLOCK_SIZE * 4)
                {
                    float4 v4 = __ldg(reinterpret_cast<float4 const*>(row_input + i));
#pragma unroll
                    for (int j = 0; j < 4; j++)
                    {
                        float val = (&v4.x)[j];
                        if (val >= thr && my_write_pos_global < kCC)
                        {
                            dsm0->keys[my_write_pos_global] = val;
                            dsm0->vals[my_write_pos_global] = i + j;
                            my_write_pos_global++;
                        }
                    }
                }
                for (int i = ci_end + tid; i < chunk_start + chunk_N; i += BLOCK_SIZE)
                {
                    float val = __ldg(&row_input[i]);
                    if (val >= thr && my_write_pos_global < kCC)
                    {
                        dsm0->keys[my_write_pos_global] = val;
                        dsm0->vals[my_write_pos_global] = i;
                        my_write_pos_global++;
                    }
                }
            }
            else
            {
                // bf16/fp16 path: 8-wide vector load, up-cast to fp32 for compare,
                // store fp32 in dsm0->keys[].
                int const ci_end = chunk_start + (chunk_N & ~7);
                for (int i = chunk_start + tid * 8; i + 7 < ci_end; i += BLOCK_SIZE * 8)
                {
                    int4 raw = __ldg(reinterpret_cast<int4 const*>(row_input + i));
                    float v[8];
                    Trait::unpack8(raw, v);
#pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        if (v[j] >= thr && my_write_pos_global < kCC)
                        {
                            dsm0->keys[my_write_pos_global] = v[j];
                            dsm0->vals[my_write_pos_global] = i + j;
                            my_write_pos_global++;
                        }
                    }
                }
                for (int i = ci_end + tid; i < chunk_start + chunk_N; i += BLOCK_SIZE)
                {
                    float val = Trait::to_fp32(__ldg(&row_input[i]));
                    if (val >= thr && my_write_pos_global < kCC)
                    {
                        dsm0->keys[my_write_pos_global] = val;
                        dsm0->vals[my_write_pos_global] = i;
                        my_write_pos_global++;
                    }
                }
            }
        }
    }
    cluster.sync(); // [SYNC-P3→P4] all candidates in rank-0's keys[]

    // ================================================================
    // Phase 4 — Leader-only histogram + snap + 2-pass emit on rank-0's keys[]
    // Other CTAs do nothing (idle wait). The aggregate cand_count is in
    // cs0->g_write_base after all CTAs have finished P3 scatter.
    // ================================================================

    int const cand_count_global_pre = (rank == 0) ? my_cs->g_write_base : cs0->g_write_base;
    int const cand_count = (cand_count_global_pre < kCC) ? cand_count_global_pre : kCC;

    if (rank == 0)
    {
        // Trivial paths
        if (cand_count == kK)
        {
            for (int i = tid; i < kK; i += BLOCK_SIZE)
            {
                if constexpr (kIsFp32)
                    row_out_v[i] = smem->keys[i];
                else
                    row_out_v[i] = Trait::from_fp32(smem->keys[i]);
                row_out_i[i] = smem->vals[i];
            }
        }
        else if (cand_count < kK)
        {
            for (int i = tid; i < cand_count; i += BLOCK_SIZE)
            {
                if constexpr (kIsFp32)
                    row_out_v[i] = smem->keys[i];
                else
                    row_out_v[i] = Trait::from_fp32(smem->keys[i]);
                row_out_i[i] = smem->vals[i];
            }
            for (int i = cand_count + tid; i < kK; i += BLOCK_SIZE)
            {
                row_out_v[i] = neg_max;
                row_out_i[i] = -1;
            }
        }
        else
        {
            // cand_count > kK: full P4 (histogram + snap + 2-pass emit)
            float cmin = FLT_MAX, cmax = -FLT_MAX;
            for (int i = tid; i < cand_count; i += BLOCK_SIZE)
            {
                float v = smem->keys[i];
                cmin = fminf(cmin, v);
                cmax = fmaxf(cmax, v);
            }
            cmin = warpReduceMin(cmin);
            cmax = warpReduceMax(cmax);
            if (lane == 0)
            {
                smem->warp_counts[warp_id] = __float_as_int(cmin);
                smem->histogram[warp_id] = __float_as_int(cmax);
            }
            __syncthreads();

            float block_min = FLT_MAX, block_max = -FLT_MAX;
            for (int w = 0; w < NUM_WARPS; w++)
            {
                block_min = fminf(block_min, __int_as_float(smem->warp_counts[w]));
                block_max = fmaxf(block_max, __int_as_float(smem->histogram[w]));
            }
            if (block_max <= block_min)
                block_max = block_min + 1e-6f;

            for (int i = tid; i < kBins; i += BLOCK_SIZE)
                smem->histogram[i] = 0;
            __syncthreads();

            float range1 = block_max - block_min;
            float inv1 = (range1 > 0.0f) ? ((float) (kBins - 1) + 0.99f) / range1 : 0.0f;

            for (int i = tid; i < cand_count; i += BLOCK_SIZE)
            {
                int bin = (int) ((smem->keys[i] - block_min) * inv1);
                bin = min(max(bin, 0), kBins - 1);
                atomicAdd(&smem->histogram[bin], 1);
            }
            __syncthreads();

            // K-th bin search (3-step parallel)
            {
                constexpr int BINS_PER_WARP = kBins / NUM_WARPS;
                static_assert(kBins % NUM_WARPS == 0, "kBins must be divisible by NUM_WARPS");
                int warp_bin_sum = 0;
                for (int j = 0; j < BINS_PER_WARP; j++)
                    warp_bin_sum += smem->histogram[kBins - 1 - warp_id * BINS_PER_WARP - j];
                if (lane == 0)
                    smem->warp_counts[warp_id] = warp_bin_sum;
            }
            __syncthreads();

            if (tid == 0)
            {
                int cum = 0, tw = NUM_WARPS - 1;
                for (int w = 0; w < NUM_WARPS; w++)
                {
                    cum += smem->warp_counts[w];
                    if (cum >= kK)
                    {
                        tw = w;
                        break;
                    }
                }
                cum = 0;
                for (int w = 0; w < tw; w++)
                    cum += smem->warp_counts[w];
                smem->cnt_lo = cum;
                smem->cnt_hi = tw;
            }
            __syncthreads();

            if (warp_id == smem->cnt_hi && lane == 0)
            {
                constexpr int BINS_PER_WARP = kBins / NUM_WARPS;
                int base_cum = smem->cnt_lo;
                float thr = block_min;
                for (int j = 0; j < BINS_PER_WARP; j++)
                {
                    int b = kBins - 1 - smem->cnt_hi * BINS_PER_WARP - j;
                    base_cum += smem->histogram[b];
                    if (base_cum >= kK)
                    {
                        thr = block_min + (float) b * range1 / (float) kBins;
                        break;
                    }
                }
                smem->threshold = thr;
            }
            __syncthreads();

            // Snap iters: reuses existing blockFusedSnapIter helper since
            // it operates on smem fields not on dsm.
            int snap_limit = (cand_count > 128 ? cand_count / 4 : 32);
            for (int si = 0; si < snap_limit; si++)
            {
                if constexpr (kIsFp32)
                    blockFusedSnapIter<TopK, SmemT>(smem, cand_count, tid, warp_id, lane);
                else
                    blockFusedSnapIterDtype<float, TopK, SmemT>(smem, cand_count, tid, warp_id, lane);
                int cge = smem->cnt_lo;
                int cgt = smem->cnt_hi;
                if (cgt < kK && cge >= kK)
                    break;
            }

            float sel_thr = smem->threshold;
            if (tid == 0)
                smem->out_count = 0;
            __syncthreads();

            // Pass 1: strictly greater than sel_thr
            for (int base = warp_id * WARP_SIZE; base < cand_count; base += BLOCK_SIZE)
            {
                int i = base + lane;
                float v = (i < cand_count) ? smem->keys[i] : -FLT_MAX;
                bool emit_gt = (i < cand_count) && (v > sel_thr);
                unsigned mask_gt = __ballot_sync(full_mask, emit_gt);
                if (mask_gt)
                {
                    int cnt = __popc(mask_gt);
                    int moff = __popc(mask_gt & ((1u << lane) - 1u));
                    int bp = 0;
                    if (lane == 0)
                        bp = atomicAdd(&smem->out_count, cnt);
                    bp = __shfl_sync(full_mask, bp, 0);
                    if (emit_gt && bp + moff < kK)
                    {
                        if constexpr (kIsFp32)
                            row_out_v[bp + moff] = v;
                        else
                            row_out_v[bp + moff] = Trait::from_fp32(v);
                        row_out_i[bp + moff] = smem->vals[i];
                    }
                }
            }
            __syncthreads();

            // Pass 2: equal to sel_thr (fills remaining)
            for (int base = warp_id * WARP_SIZE; base < cand_count; base += BLOCK_SIZE)
            {
                int i = base + lane;
                float v = (i < cand_count) ? smem->keys[i] : -FLT_MAX;
                bool emit_eq = (i < cand_count) && (v == sel_thr);
                unsigned mask_eq = __ballot_sync(full_mask, emit_eq);
                if (mask_eq)
                {
                    int cnt = __popc(mask_eq);
                    int moff = __popc(mask_eq & ((1u << lane) - 1u));
                    int bp = 0;
                    if (lane == 0)
                        bp = atomicAdd(&smem->out_count, cnt);
                    bp = __shfl_sync(full_mask, bp, 0);
                    if (emit_eq && bp + moff < kK)
                    {
                        if constexpr (kIsFp32)
                            row_out_v[bp + moff] = v;
                        else
                            row_out_v[bp + moff] = Trait::from_fp32(v);
                        row_out_i[bp + moff] = smem->vals[i];
                    }
                }
            }
            __syncthreads();

            int filled = min(smem->out_count, kK);
            for (int i = filled + tid; i < kK; i += BLOCK_SIZE)
            {
                row_out_v[i] = neg_max;
                row_out_i[i] = -1;
            }
        }
    }
    cluster.sync(); // [SYNC-end] keep peers alive while leader writes outputs
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Explicit instantiations: 9 (T, K) cells × 2 cluster sizes (2, 4) = 18 total.
#define INSTANTIATE_COOP(InputT, K, CS)                                                                                \
    template __global__ void heuristicTopKCoopKernel<InputT, K, CS>(                                                   \
        InputT const*, int const*, int const*, InputT*, int*, int, int, int, int, int);

INSTANTIATE_COOP(float, 512, 2)
INSTANTIATE_COOP(float, 512, 4)
INSTANTIATE_COOP(float, 1024, 2)
INSTANTIATE_COOP(float, 1024, 4)
INSTANTIATE_COOP(float, 2048, 2)
INSTANTIATE_COOP(float, 2048, 4)
INSTANTIATE_COOP(__nv_bfloat16, 512, 2)
INSTANTIATE_COOP(__nv_bfloat16, 512, 4)
INSTANTIATE_COOP(__nv_bfloat16, 1024, 2)
INSTANTIATE_COOP(__nv_bfloat16, 1024, 4)
INSTANTIATE_COOP(__nv_bfloat16, 2048, 2)
INSTANTIATE_COOP(__nv_bfloat16, 2048, 4)
INSTANTIATE_COOP(__half, 512, 2)
INSTANTIATE_COOP(__half, 512, 4)
INSTANTIATE_COOP(__half, 1024, 2)
INSTANTIATE_COOP(__half, 1024, 4)
INSTANTIATE_COOP(__half, 2048, 2)
INSTANTIATE_COOP(__half, 2048, 4)

#undef INSTANTIATE_COOP

// Explicit instantiations for Stage A + Stage B kernels
template __global__ void heuristicTopKStageAKernel<float, 512>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKStageAKernel<float, 1024>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKStageAKernel<float, 2048>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKStageAKernel<__nv_bfloat16, 512>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKStageAKernel<__nv_bfloat16, 1024>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKStageAKernel<__nv_bfloat16, 2048>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKStageAKernel<__half, 512>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKStageAKernel<__half, 1024>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKStageAKernel<__half, 2048>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int, int);

template __global__ void heuristicTopKMergeKernel<float, 512>(float const*, int const*, int, float*, int*, int);
template __global__ void heuristicTopKMergeKernel<float, 1024>(float const*, int const*, int, float*, int*, int);
template __global__ void heuristicTopKMergeKernel<float, 2048>(float const*, int const*, int, float*, int*, int);
template __global__ void heuristicTopKMergeKernel<__nv_bfloat16, 512>(
    __nv_bfloat16 const*, int const*, int, __nv_bfloat16*, int*, int);
template __global__ void heuristicTopKMergeKernel<__nv_bfloat16, 1024>(
    __nv_bfloat16 const*, int const*, int, __nv_bfloat16*, int*, int);
template __global__ void heuristicTopKMergeKernel<__nv_bfloat16, 2048>(
    __nv_bfloat16 const*, int const*, int, __nv_bfloat16*, int*, int);
template __global__ void heuristicTopKMergeKernel<__half, 512>(__half const*, int const*, int, __half*, int*, int);
template __global__ void heuristicTopKMergeKernel<__half, 1024>(__half const*, int const*, int, __half*, int*, int);
template __global__ void heuristicTopKMergeKernel<__half, 2048>(__half const*, int const*, int, __half*, int*, int);

// Lazy-allocated staging buffer (Phase 1 first cut; v2.1 should integrate with
// TRT-LLM workspace allocator). One buffer max-sized for fp32 (4B per element);
// reinterpret_cast for bf16/fp16. Index buffer is always int32.
namespace mc_staging
{
static void* sValuesBuf = nullptr;
static int* sIndicesBuf = nullptr;
static size_t sCapacityElems = 0; // capacity in fp32-sized elements
static std::mutex sMutex;
} // namespace mc_staging

inline void ensureMcStaging(size_t numElems)
{
    std::lock_guard<std::mutex> lock(mc_staging::sMutex);
    if (numElems > mc_staging::sCapacityElems)
    {
        if (mc_staging::sValuesBuf)
            cudaFree(mc_staging::sValuesBuf);
        if (mc_staging::sIndicesBuf)
            cudaFree(mc_staging::sIndicesBuf);
        cudaMalloc(&mc_staging::sValuesBuf, numElems * sizeof(float));
        cudaMalloc(&mc_staging::sIndicesBuf, numElems * sizeof(int));
        mc_staging::sCapacityElems = numElems;
    }
}

template <typename InputT, int TopK>
void launchHeuristicTopKDecodeMCImpl(InputT const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    InputT* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int clusterSize, cudaStream_t stream)
{
    using Params = GvrParams<InputT, TopK>;
    using SmemT = KernelSmemTplK<float, Params::kC, Params::kNumBins>;
    size_t const smemSize = sizeof(SmemT);

    size_t const stagingElems = static_cast<size_t>(numRows) * clusterSize * topK;
    ensureMcStaging(stagingElems);
    InputT* stagingV = reinterpret_cast<InputT*>(mc_staging::sValuesBuf);
    int* stagingI = mc_staging::sIndicesBuf;

    bool const enablePDL = tensorrt_llm::common::getEnvEnablePDL();

    // Stage A: numRows × clusterSize CTAs
    {
        auto kfn = heuristicTopKStageAKernel<InputT, TopK>;
        if (smemSize > 48u * 1024u)
        {
            cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
        }
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(static_cast<unsigned>(numRows * clusterSize));
        config.blockDim = dim3(BLOCK_SIZE);
        config.dynamicSmemBytes = smemSize;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = enablePDL ? 1 : 0;
        config.attrs = attrs;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config, kfn, logits, seqLens, preIdx, stagingV, stagingI, stride0, next_n, topK,
            preIdxStride, preIdxCount, clusterSize);
    }

    // Stage B: numRows CTAs (1 per row, merge clusterSize × K cands → top K)
    {
        auto kfn = heuristicTopKMergeKernel<InputT, TopK>;
        if (smemSize > 48u * 1024u)
        {
            cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
        }
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(static_cast<unsigned>(numRows));
        config.blockDim = dim3(BLOCK_SIZE);
        config.dynamicSmemBytes = smemSize;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = enablePDL ? 1 : 0;
        config.attrs = attrs;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config, kfn, stagingV, stagingI, clusterSize, scratchValues, outIndices, topK);
    }
}

template <typename InputT>
void launchHeuristicTopKDecodeMCDispatch(InputT const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    InputT* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int numColumns, cudaStream_t stream)
{
    int kCC, nCtaPerSm;
    if (topK == 512)
    {
        kCC = GvrParams<InputT, 512>::kC;
        nCtaPerSm = std::is_same_v<InputT, float> ? 4 : 4;
    }
    else if (topK == 1024)
    {
        kCC = GvrParams<InputT, 1024>::kC;
        nCtaPerSm = 3;
    }
    else
    { // 2048
        kCC = GvrParams<InputT, 2048>::kC;
        nCtaPerSm = std::is_same_v<InputT, float> ? 3 : 2;
    }
    int const clusterSize = computeMcClusterSize(numRows, numColumns, topK, kCC, nCtaPerSm);

    if (clusterSize <= 1)
    {
        // Fall back to SC
        launchHeuristicTopKDecodeImpl<InputT>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n, topK,
            preIdxStride, preIdxCount, numRows, stream);
        return;
    }

    switch (topK)
    {
    case 512:
        launchHeuristicTopKDecodeMCImpl<InputT, 512>(logits, seqLens, preIdx, outIndices, scratchValues, stride0,
            next_n, topK, preIdxStride, preIdxCount, numRows, clusterSize, stream);
        break;
    case 1024:
        launchHeuristicTopKDecodeMCImpl<InputT, 1024>(logits, seqLens, preIdx, outIndices, scratchValues, stride0,
            next_n, topK, preIdxStride, preIdxCount, numRows, clusterSize, stream);
        break;
    case 2048:
        launchHeuristicTopKDecodeMCImpl<InputT, 2048>(logits, seqLens, preIdx, outIndices, scratchValues, stride0,
            next_n, topK, preIdxStride, preIdxCount, numRows, clusterSize, stream);
        break;
    default: TLLM_THROW("heuristicTopKDecodeMC: topK must be 512/1024/2048");
    }
}

// ============================================================================
// Coop launcher (v2.1, single cooperative kernel)
// ============================================================================

template <typename InputT, int TopK, int kClusterSize>
void launchHeuristicTopKDecodeCoopImpl(InputT const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    InputT* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream)
{
    using Params = GvrParams<InputT, TopK>;
    using SmemT = KernelSmemTplK<float, Params::kC, Params::kNumBins>;
    // Coop kernel uses [SmemT][ClusterScratch] layout per CTA.
    size_t const smemSize = sizeof(SmemT) + sizeof(ClusterScratch);

    auto kfn = heuristicTopKCoopKernel<InputT, TopK, kClusterSize>;
    if (smemSize > 48u * 1024u)
    {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
    }

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned>(numRows * kClusterSize));
    config.blockDim = dim3(BLOCK_SIZE);
    config.dynamicSmemBytes = smemSize;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim = {static_cast<unsigned>(kClusterSize), 1u, 1u};
    config.attrs = attrs;
    config.numAttrs = 2;

    cudaLaunchKernelEx(&config, kfn, logits, seqLens, preIdx, scratchValues, outIndices, stride0, next_n, topK,
        preIdxStride, preIdxCount);
}

template <typename InputT>
void launchHeuristicTopKDecodeCoopDispatch(InputT const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    InputT* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int numColumns, cudaStream_t stream)
{
    int kCC, nCtaPerSm;
    if (topK == 512)
    {
        kCC = GvrParams<InputT, 512>::kC;
        nCtaPerSm = 4;
    }
    else if (topK == 1024)
    {
        kCC = GvrParams<InputT, 1024>::kC;
        nCtaPerSm = 3;
    }
    else
    {
        kCC = GvrParams<InputT, 2048>::kC;
        nCtaPerSm = std::is_same_v<InputT, float> ? 3 : 2;
    }
    int clusterSize = computeMcClusterSize(numRows, numColumns, topK, kCC, nCtaPerSm);
    if (clusterSize < 2)
        clusterSize = 2; // Coop path requires cluster ≥ 2; SC fallback handled by gate.
    if (clusterSize > 4)
        clusterSize = 4; // Coop path templated on {2, 4} only.
    if (clusterSize == 3)
        clusterSize = 2; // round to power of 2

#define DISPATCH_COOP_K(K)                                                                                             \
    case K:                                                                                                            \
        if (clusterSize == 2)                                                                                          \
            launchHeuristicTopKDecodeCoopImpl<InputT, K, 2>(logits, seqLens, preIdx, outIndices, scratchValues,        \
                stride0, next_n, topK, preIdxStride, preIdxCount, numRows, stream);                                    \
        else                                                                                                           \
            launchHeuristicTopKDecodeCoopImpl<InputT, K, 4>(logits, seqLens, preIdx, outIndices, scratchValues,        \
                stride0, next_n, topK, preIdxStride, preIdxCount, numRows, stream);                                    \
        break;

    switch (topK)
    {
        DISPATCH_COOP_K(512)
        DISPATCH_COOP_K(1024)
        DISPATCH_COOP_K(2048)
    default: TLLM_THROW("heuristicTopKDecodeCoop: topK must be 512/1024/2048");
    }

#undef DISPATCH_COOP_K
}

} // anonymous namespace

// MC route gate: caller passes numColumns; if numColumns >= 8*topK and
// TRTLLM_GVR_MC=1 (default off — preserves current production), try MC.
template <typename InputT>
inline bool shouldUseMC(int numRows, int numColumns, int topK)
{
    if (numColumns < 8 * topK)
        return false;
    char const* envMc = std::getenv("TRTLLM_GVR_MC");
    if (!(envMc && envMc[0] == '1' && envMc[1] == '\0'))
        return false;
    int kCC, nCtaPerSm;
    if (topK == 512)
    {
        kCC = GvrParams<InputT, 512>::kC;
        nCtaPerSm = 4;
    }
    else if (topK == 1024)
    {
        kCC = GvrParams<InputT, 1024>::kC;
        nCtaPerSm = 3;
    }
    else
    { // 2048
        kCC = GvrParams<InputT, 2048>::kC;
        nCtaPerSm = std::is_same_v<InputT, float> ? 3 : 2;
    }
    return computeMcClusterSize(numRows, numColumns, topK, kCC, nCtaPerSm) > 1;
}

// Coop route gate: env-gated independently (TRTLLM_GVR_MC_COOP=1). Takes
// precedence over MC v2 (Stage A+B). Same partition policy: requires
// numColumns ≥ 8*topK and computed cluster_size ≥ 2.
template <typename InputT>
inline bool shouldUseCoop(int numRows, int numColumns, int topK)
{
    if (numColumns < 8 * topK)
        return false;
    char const* envCoop = std::getenv("TRTLLM_GVR_MC_COOP");
    if (!(envCoop && envCoop[0] == '1' && envCoop[1] == '\0'))
        return false;
    int kCC, nCtaPerSm;
    if (topK == 512)
    {
        kCC = GvrParams<InputT, 512>::kC;
        nCtaPerSm = 4;
    }
    else if (topK == 1024)
    {
        kCC = GvrParams<InputT, 1024>::kC;
        nCtaPerSm = 3;
    }
    else
    {
        kCC = GvrParams<InputT, 2048>::kC;
        nCtaPerSm = std::is_same_v<InputT, float> ? 3 : 2;
    }
    return computeMcClusterSize(numRows, numColumns, topK, kCC, nCtaPerSm) >= 2;
}

void launchHeuristicTopKDecode(float const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    float* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream, int numColumns)
{
    if (shouldUseCoop<float>(numRows, numColumns, topK))
    {
        launchHeuristicTopKDecodeCoopDispatch<float>(logits, seqLens, preIdx, outIndices, scratchValues, stride0,
            next_n, topK, preIdxStride, preIdxCount, numRows, numColumns, stream);
        return;
    }
    if (shouldUseMC<float>(numRows, numColumns, topK))
    {
        launchHeuristicTopKDecodeMCDispatch<float>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n,
            topK, preIdxStride, preIdxCount, numRows, numColumns, stream);
        return;
    }
    launchHeuristicTopKDecodeImpl<float>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n, topK,
        preIdxStride, preIdxCount, numRows, stream);
}

void launchHeuristicTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __nv_bfloat16* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream, int numColumns)
{
    if (shouldUseCoop<__nv_bfloat16>(numRows, numColumns, topK))
    {
        launchHeuristicTopKDecodeCoopDispatch<__nv_bfloat16>(logits, seqLens, preIdx, outIndices, scratchValues,
            stride0, next_n, topK, preIdxStride, preIdxCount, numRows, numColumns, stream);
        return;
    }
    if (shouldUseMC<__nv_bfloat16>(numRows, numColumns, topK))
    {
        launchHeuristicTopKDecodeMCDispatch<__nv_bfloat16>(logits, seqLens, preIdx, outIndices, scratchValues, stride0,
            next_n, topK, preIdxStride, preIdxCount, numRows, numColumns, stream);
        return;
    }
    launchHeuristicTopKDecodeImpl<__nv_bfloat16>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n,
        topK, preIdxStride, preIdxCount, numRows, stream);
}

void launchHeuristicTopKDecode(__half const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __half* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream, int numColumns)
{
    if (shouldUseCoop<__half>(numRows, numColumns, topK))
    {
        launchHeuristicTopKDecodeCoopDispatch<__half>(logits, seqLens, preIdx, outIndices, scratchValues, stride0,
            next_n, topK, preIdxStride, preIdxCount, numRows, numColumns, stream);
        return;
    }
    if (shouldUseMC<__half>(numRows, numColumns, topK))
    {
        launchHeuristicTopKDecodeMCDispatch<__half>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n,
            topK, preIdxStride, preIdxCount, numRows, numColumns, stream);
        return;
    }
    launchHeuristicTopKDecodeImpl<__half>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n, topK,
        preIdxStride, preIdxCount, numRows, stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
