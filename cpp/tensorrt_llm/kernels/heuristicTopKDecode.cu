/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

using heuristic_topk::BLOCK_SIZE;
using heuristic_topk::GvrDtypeTraits;
using heuristic_topk::GvrParams;
using heuristic_topk::GvrPreIdxMode;
using heuristic_topk::gvrTopKJob;
using heuristic_topk::gvrTopKJobDtype;
using heuristic_topk::KernelSmemTplK;

// Templated on TopK so the launcher can dispatch K=512/1024/2048 to the
// same kernel template. Smem layout is derived from GvrParams<float, TopK>
// at compile time.
template <int TopK>
__global__ void __launch_bounds__(BLOCK_SIZE)
    heuristicTopKMultiRowKernel(float const* __restrict__ logits, int const* __restrict__ seqLens,
        int const* __restrict__ preIdx, float* __restrict__ scratchValues, int* __restrict__ outIndices, int stride0,
        int next_n, int topK, int preIdxStride, int preIdxCount, int compressRatio)
{
    using SmemT = KernelSmemTplK<float, GvrParams<float, TopK>::kC, GvrParams<float, TopK>::kNumBins>;

    int const rowIdx = blockIdx.x;
    int const seq_len = seqLens[rowIdx / next_n];
    // seqLens is in uncompressed token space; the logits/preIdx live in
    // compressed-index space when compressRatio > 1 (DSv4 indexer).
    int const actual_kv_len = seq_len - next_n + (rowIdx % next_n) + 1;
    int const N = actual_kv_len / compressRatio;

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

    // Temporal-shift offset to map prev-step's top-K indices into this step's
    // KV index space.
    //   compressRatio == 1 (DSv3.2):   +1 — KV grew by exactly 1 token per
    //     decode step; prev indices were at seq_len-1 so a uniform +1 maps
    //     them to the equivalent positions under the indexer's "newest-first"
    //     layout. The (rowIdx % next_n) addend extends this to MTP windows.
    //   compressRatio == 4 (DSv4):     0 — in compressed-index space new
    //     compressed entries are appended at the end; prev indices in
    //     [0, c_prev-1] remain valid as-is. Per-row Δc varies (0 or 1) with
    //     prev kv_len mod 4 alignment, but a uniform offset of 0 stays
    //     within-bounds for all rows and preserves the temporal-correlation
    //     hint (vertical top-K consistency validated offline).
    int const preIdxOffset = (compressRatio == 1) ? ((rowIdx % next_n) + 1) : 0;
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
__global__ void __launch_bounds__(BLOCK_SIZE)
    heuristicTopKMultiRowKernelDtype(InputT const* __restrict__ logits, int const* __restrict__ seqLens,
        int const* __restrict__ preIdx, InputT* __restrict__ scratchValues, int* __restrict__ outIndices, int stride0,
        int next_n, int topK, int preIdxStride, int preIdxCount, int compressRatio)
{
    // dtype path uses fp32 keys[] in smem (down-conversion deferred to writeback).
    using SmemT = KernelSmemTplK<float, GvrParams<InputT, TopK>::kC, GvrParams<InputT, TopK>::kNumBins>;

    int const rowIdx = blockIdx.x;
    int const seq_len = seqLens[rowIdx / next_n];
    int const actual_kv_len = seq_len - next_n + (rowIdx % next_n) + 1;
    int const N = actual_kv_len / compressRatio;

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

    // See fp32 path: cr==1 → (rowIdx % next_n)+1; cr!=1 (DSv4) → 0.
    int const preIdxOffset = (compressRatio == 1) ? ((rowIdx % next_n) + 1) : 0;
    gvrTopKJobDtype<InputT, TopK>(
        input, N, rowPreIdx, preIdxCount, topK, outputValues, outputIndices, smem, preIdxOffset);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Explicit instantiations — 6 (dtype × K) combos. Launchers dispatch on
// runtime topK via switch, so all 6 must be available at link time.
// Trailing `int` is the compressRatio parameter (1 = V3.2, 4 = V4 indexer).
template __global__ void heuristicTopKMultiRowKernelDtype<__nv_bfloat16, 512>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__nv_bfloat16, 1024>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__nv_bfloat16, 2048>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__half, 512>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__half, 1024>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__half, 2048>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernel<512>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernel<1024>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernel<2048>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int, int);

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
    int compressRatio, cudaStream_t stream)
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
            preIdxStride, preIdxCount, compressRatio);
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
// Prefill assembly kernel (mirror of decode, micro-kernel shared)
// ============================================================================
// Derives `N_r = rowEnds[r] - rowStarts[r]` per row (chunked-prefill row
// geometry — supports nonzero `rowStart` for `cu_seqlen_ks > 0`) and
// dispatches the same `gvrTopKJob<TopK, PreIdxMode>` micro-kernel as decode.
// preIdx is synthesized in Phase 1 via `PreIdxMode`, not read from a tensor:
//   ConstIdentity (V4 cr=4) : idx = i               → [0..K-1]
//   BaseShift     (V3.2 cr=1): idx = (N_r-K) + i     → [N_r-K..N_r-1]
// Region-A (`N_r <= topK`) short-circuits to identity output before calling
// the micro-kernel, matching the decode assembly's early-exit and the radix-
// based prefill behavior for short rows.
//
// dtype-templated. The fp32 specialization preserves the original fp32-only
// production path (radix-based prefill is also fp32 because DeepGEMM's
// `fp8_mqa_logits` returns fp32). bf16/fp16 specializations are added for
// API symmetry with the decode op (which accepts fp32/bf16/fp16) and for
// callers that pass non-fp32 logits directly (e.g., manual experiments,
// future MQA logits variants).
template <typename InputT, int TopK, GvrPreIdxMode PreIdxMode>
__global__ void __launch_bounds__(BLOCK_SIZE)
    heuristicTopKMultiRowKernelPrefillDtype(InputT const* __restrict__ logits, int const* __restrict__ rowStarts,
        int const* __restrict__ rowEnds, InputT* __restrict__ scratchValues, int* __restrict__ outIndices, int stride0,
        int topK)
{
    // dtype path uses fp32 keys[] in smem (down-conversion deferred to writeback)
    using SmemT = KernelSmemTplK<float, GvrParams<InputT, TopK>::kC, GvrParams<InputT, TopK>::kNumBins>;

    int const rowIdx = blockIdx.x;
    int const rowStart = rowStarts[rowIdx];
    int const rowEnd = rowEnds[rowIdx];
    int const N = rowEnd - rowStart;

    // rowStart is folded into the input pointer; the micro-kernel uses local
    // [0, N) indexing and emits row-local output indices, identical to the
    // existing radix-based prefill behavior (no Python-side `-= cu_seqlen_ks`
    // is required by callers).
    InputT const* __restrict__ input = logits + static_cast<int64_t>(rowIdx) * stride0 + rowStart;
    // outputValues is a kernel-internal write-only output store: the
    // micro-kernel's Phase 4 final writeback and Region-A early-exit write
    // top-K logit values here, but NO code path inside the kernel reads them
    // back, and Python downstream (`heuristic_prefill_scratch_values`) is
    // only allocated as a caller-owned scratch — never read after the op
    // returns. So all CTAs in the prefill launch share the same [topK] slot.
    // The cross-CTA write race is benign (last-write-wins, no reader). See
    // the corresponding comment in the fp32 path for the full rationale.
    InputT* __restrict__ outputValues = scratchValues;
    int* __restrict__ outputIndices = outIndices + static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<SmemT*>(smem_raw);

    if (N <= topK)
    {
        // Region-A: dense degeneracy. Causal-mask ensures all `N` valid
        // positions are retained; rest is -FLT_MAX / -1 padded.
        int const tid = threadIdx.x;
        if constexpr (std::is_same_v<InputT, float>)
        {
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
        }
        else
        {
            InputT const neg_max = GvrDtypeTraits<InputT>::from_fp32(-FLT_MAX);
            for (int i = tid; i < N; i += BLOCK_SIZE)
            {
                outputValues[i] = input[i];
                outputIndices[i] = i;
            }
            for (int i = N + tid; i < topK; i += BLOCK_SIZE)
            {
                outputValues[i] = neg_max;
                outputIndices[i] = -1;
            }
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    // Region-B: N > topK. Synthesize preIdx via PreIdxMode + preIdxOffset.
    //   ConstIdentity (V4 cr=4)  → preIdxOffset unused; idx = i  ∈ [0, K)
    //   BaseShift     (V3.2 cr=1) → preIdxOffset = base = N - K; idx = base + i
    int const preIdxOffsetForMode = (PreIdxMode == GvrPreIdxMode::BaseShift) ? (N - topK) : 0;

    // Pass nullptr for preIdx — the micro-kernel does not dereference it in
    // ConstIdentity / BaseShift modes (Phase 1 takes the synthesized branch).
    //
    // Aligned=false: the per-row input pointer `logits + r*stride0 + rowStart`
    // may be misaligned for fp32 when `rowStart % 4 != 0` and for bf16/fp16
    // when `rowStart % 8 != 0` (multi-request packed chunks and batch_size > 1
    // single-pass prefill produce arbitrary cumulative rowStart values).
    // The misalignment-safe path in gvrTopKJob{,Dtype} adds a scalar prologue
    // to blockCountGE and Phase-3 collect, keeping vals indices in the
    // original [0, N) frame.
    if constexpr (std::is_same_v<InputT, float>)
    {
        gvrTopKJob<TopK, PreIdxMode, /*Aligned=*/false>(
            input, N, /*preIdx=*/nullptr, /*M=*/topK, topK, outputValues, outputIndices, smem, preIdxOffsetForMode);
    }
    else
    {
        gvrTopKJobDtype<InputT, TopK, PreIdxMode, /*Aligned=*/false>(
            input, N, /*preIdx=*/nullptr, /*M=*/topK, topK, outputValues, outputIndices, smem, preIdxOffsetForMode);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Explicit instantiations — {fp32, bf16, fp16} × {K=512, 1024, 2048} ×
// {ConstIdentity, BaseShift} = 18 combos. Launcher dispatches on (dtype,
// compressRatio, topK) at runtime via switch, so all 18 must be available
// at link time.
//
// fp32 path serves the production radix-replacement (DeepGEMM
// `fp8_mqa_logits` returns fp32). bf16/fp16 specializations exist for API
// symmetry with the decode op and for callers that pass non-fp32 logits
// (manual experiments, future MQA logits variants).
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<float, 512, GvrPreIdxMode::ConstIdentity>(
    float const*, int const*, int const*, float*, int*, int, int); // V4 Flash fp32
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<float, 1024, GvrPreIdxMode::ConstIdentity>(
    float const*, int const*, int const*, float*, int*, int, int); // V4 Pro fp32
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<float, 2048, GvrPreIdxMode::ConstIdentity>(
    float const*, int const*, int const*, float*, int*, int, int); // future V4 K=2048 fp32
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<float, 512, GvrPreIdxMode::BaseShift>(
    float const*, int const*, int const*, float*, int*, int, int); // V3.2 small K fp32
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<float, 1024, GvrPreIdxMode::BaseShift>(
    float const*, int const*, int const*, float*, int*, int, int); // V3.2 mid K fp32
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<float, 2048, GvrPreIdxMode::BaseShift>(
    float const*, int const*, int const*, float*, int*, int, int); // V3.2 production K fp32
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__nv_bfloat16, 512, GvrPreIdxMode::ConstIdentity>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int); // V4 Flash bf16
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__nv_bfloat16, 1024, GvrPreIdxMode::ConstIdentity>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int); // V4 Pro bf16
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__nv_bfloat16, 2048, GvrPreIdxMode::ConstIdentity>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int);
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__nv_bfloat16, 512, GvrPreIdxMode::BaseShift>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int);
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__nv_bfloat16, 1024, GvrPreIdxMode::BaseShift>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int);
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__nv_bfloat16, 2048, GvrPreIdxMode::BaseShift>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int);
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__half, 512, GvrPreIdxMode::ConstIdentity>(
    __half const*, int const*, int const*, __half*, int*, int, int); // V4 Flash fp16
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__half, 1024, GvrPreIdxMode::ConstIdentity>(
    __half const*, int const*, int const*, __half*, int*, int, int); // V4 Pro fp16
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__half, 2048, GvrPreIdxMode::ConstIdentity>(
    __half const*, int const*, int const*, __half*, int*, int, int);
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__half, 512, GvrPreIdxMode::BaseShift>(
    __half const*, int const*, int const*, __half*, int*, int, int);
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__half, 1024, GvrPreIdxMode::BaseShift>(
    __half const*, int const*, int const*, __half*, int*, int, int);
template __global__ void heuristicTopKMultiRowKernelPrefillDtype<__half, 2048, GvrPreIdxMode::BaseShift>(
    __half const*, int const*, int const*, __half*, int*, int, int);

template <typename InputT>
void launchHeuristicTopKPrefillImpl(InputT const* logits, int const* rowStarts, int const* rowEnds, int* outIndices,
    InputT* scratchValues, int stride0, int topK, int numRows, int compressRatio, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        topK == 512 || topK == 1024 || topK == 2048, "heuristicTopKPrefill requires topK ∈ {512, 1024, 2048}");
    TLLM_CHECK_WITH_INFO(compressRatio == 1 || compressRatio == 4,
        "heuristicTopKPrefill requires compressRatio ∈ {1, 4}; got %d", compressRatio);
    // fp32: float4 vector load needs 4-element alignment.
    // bf16/fp16: int4 vector load needs 8-element alignment.
    constexpr int kAlign = std::is_same_v<InputT, float> ? 4 : 8;
    TLLM_CHECK_WITH_INFO(stride0 % kAlign == 0 || numRows <= 1,
        "heuristicTopKPrefill requires logits stride0 divisible by %d (vector-load alignment)", kAlign);

    auto launchOne = [&]<int TopK, GvrPreIdxMode PreIdxMode>()
    {
        using SmemT = KernelSmemTplK<float, GvrParams<InputT, TopK>::kC, GvrParams<InputT, TopK>::kNumBins>;
        size_t const smemSize = sizeof(SmemT);

        auto* kfn = heuristicTopKMultiRowKernelPrefillDtype<InputT, TopK, PreIdxMode>;
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

        cudaLaunchKernelEx(&config, kfn, logits, rowStarts, rowEnds, scratchValues, outIndices, stride0, topK);
    };

    if (compressRatio == 4)
    {
        switch (topK)
        {
        case 512: launchOne.template operator()<512, GvrPreIdxMode::ConstIdentity>(); break;
        case 1024: launchOne.template operator()<1024, GvrPreIdxMode::ConstIdentity>(); break;
        case 2048: launchOne.template operator()<2048, GvrPreIdxMode::ConstIdentity>(); break;
        default: TLLM_THROW("heuristicTopKPrefill: topK validated above; unreachable");
        }
    }
    else // compressRatio == 1
    {
        switch (topK)
        {
        case 512: launchOne.template operator()<512, GvrPreIdxMode::BaseShift>(); break;
        case 1024: launchOne.template operator()<1024, GvrPreIdxMode::BaseShift>(); break;
        case 2048: launchOne.template operator()<2048, GvrPreIdxMode::BaseShift>(); break;
        default: TLLM_THROW("heuristicTopKPrefill: topK validated above; unreachable");
        }
    }
}

} // anonymous namespace

void launchHeuristicTopKDecode(float const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    float* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int compressRatio, cudaStream_t stream)
{
    launchHeuristicTopKDecodeImpl<float>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n, topK,
        preIdxStride, preIdxCount, numRows, compressRatio, stream);
}

void launchHeuristicTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __nv_bfloat16* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int compressRatio, cudaStream_t stream)
{
    launchHeuristicTopKDecodeImpl<__nv_bfloat16>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n,
        topK, preIdxStride, preIdxCount, numRows, compressRatio, stream);
}

void launchHeuristicTopKDecode(__half const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __half* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int compressRatio, cudaStream_t stream)
{
    launchHeuristicTopKDecodeImpl<__half>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n, topK,
        preIdxStride, preIdxCount, numRows, compressRatio, stream);
}

void launchHeuristicTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* outIndices,
    float* scratchValues, int stride0, int topK, int numRows, int compressRatio, cudaStream_t stream)
{
    launchHeuristicTopKPrefillImpl<float>(
        logits, rowStarts, rowEnds, outIndices, scratchValues, stride0, topK, numRows, compressRatio, stream);
}

void launchHeuristicTopKPrefill(__nv_bfloat16 const* logits, int const* rowStarts, int const* rowEnds, int* outIndices,
    __nv_bfloat16* scratchValues, int stride0, int topK, int numRows, int compressRatio, cudaStream_t stream)
{
    launchHeuristicTopKPrefillImpl<__nv_bfloat16>(
        logits, rowStarts, rowEnds, outIndices, scratchValues, stride0, topK, numRows, compressRatio, stream);
}

void launchHeuristicTopKPrefill(__half const* logits, int const* rowStarts, int const* rowEnds, int* outIndices,
    __half* scratchValues, int stride0, int topK, int numRows, int compressRatio, cudaStream_t stream)
{
    launchHeuristicTopKPrefillImpl<__half>(
        logits, rowStarts, rowEnds, outIndices, scratchValues, stride0, topK, numRows, compressRatio, stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
