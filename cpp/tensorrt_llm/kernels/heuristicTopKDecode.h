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

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

inline constexpr int kHeuristicTopK = 2048;
inline constexpr int kHeuristicSize = 2048;

/// Launch heuristic TopK decode kernel — fp32 input.
/// @param scratchValues Caller-owned buffer of size [numRows * topK] floats.
///        Required for CUDA Graph compatibility — must have a stable device address.
/// @param compressRatio  KV compression ratio (1 = V3.2 indexer; 4 = V4 indexer
///        whose logits/preIdx live in compressed-token-index space). For
///        compressRatio != 1, preIdxOffset is forced to 0 (append-at-end in
///        compressed space → prev-step indices remain valid as-is); the
///        existing (rowIdx % next_n)+1 shift is used only when compressRatio==1.
void launchHeuristicTopKDecode(float const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    float* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int compressRatio, cudaStream_t stream);

/// Launch heuristic TopK decode kernel — bf16 input.
/// scratchValues is [numRows * topK] of bf16 (matches input dtype).
/// @param compressRatio  See fp32 overload.
void launchHeuristicTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __nv_bfloat16* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int compressRatio, cudaStream_t stream);

/// Launch heuristic TopK decode kernel — fp16 input.
/// scratchValues is [numRows * topK] of fp16 (matches input dtype).
/// @param compressRatio  See fp32 overload.
void launchHeuristicTopKDecode(__half const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __half* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int compressRatio, cudaStream_t stream);

/// Launch heuristic TopK prefill kernel — fp32 input. Production path
/// (DeepGEMM's `fp8_mqa_logits` returns fp32 logits for prefill).
///
/// Dispatches to `heuristicTopKMultiRowKernelPrefillDtype<float, ...>` which
/// derives per-row `N_r = rowEnds[r] - rowStarts[r]` (chunked-prefill row
/// geometry, supports nonzero rowStart for cu_seqlen_ks > 0) and calls the
/// same `gvrTopKJob<TopK, PreIdxMode, /*Aligned=*/false>` micro-kernel as
/// the bf16/fp16 paths.
///
/// preIdx is synthesized inside the kernel from row geometry, NOT passed as a
/// tensor:
///   compressRatio == 1 (V3.2): BaseShift mode, idx = (N_r - K) + i — the
///     "most recent K positions" causal-diagonal pattern.
///   compressRatio == 4 (V4 Flash / Pro): ConstIdentity mode, idx = i — the
///     constant `[0..K-1]` pattern in compressed-token-index space.
/// Region-A rows where `N_r <= topK` short-circuit to identity output before
/// calling the micro-kernel (matches the decode launcher's early-exit and
/// the radix-based prefill behavior for short rows).
///
/// @param scratchValues Caller-owned buffer of size `topK` fp32 values
///        (shared across all CTAs in the launch; kernel writes are race-but-
///        no-reader, see `heuristicTopKMultiRowKernelPrefillDtype` comment).
///        Must have a stable device address for CUDA-Graph compatibility.
void launchHeuristicTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* outIndices,
    float* scratchValues, int stride0, int topK, int numRows, int compressRatio, cudaStream_t stream);

/// Launch heuristic TopK prefill kernel — bf16 input. API symmetry with the
/// decode op (which accepts fp32/bf16/fp16). Not the production path today
/// (DeepGEMM's fp8_mqa_logits returns fp32), but available for callers that
/// pass bf16 logits directly. `scratchValues` must be bf16 and sized `topK`.
void launchHeuristicTopKPrefill(__nv_bfloat16 const* logits, int const* rowStarts, int const* rowEnds, int* outIndices,
    __nv_bfloat16* scratchValues, int stride0, int topK, int numRows, int compressRatio, cudaStream_t stream);

/// Launch heuristic TopK prefill kernel — fp16 input. See bf16 overload.
void launchHeuristicTopKPrefill(__half const* logits, int const* rowStarts, int const* rowEnds, int* outIndices,
    __half* scratchValues, int stride0, int topK, int numRows, int compressRatio, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
