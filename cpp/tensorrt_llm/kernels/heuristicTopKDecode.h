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

#pragma once

#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// ============================================================================
// V4 K=M=512 ABLATION CONFIG BLOCK (branch feat/gvr-v4-K512-ablation)
// ----------------------------------------------------------------------------
// This block is the SINGLE SOURCE OF TRUTH for all 4 GVR-pipeline macros.
// Both indexerTopK.cu (uses kHeuristicTopK / kHeuristicSize constants) and
// heuristic_topk.cuh (uses TOP_K, F_TARGET, MAX_CANDIDATES constexpr derived
// from these macros) include this header — the include sequencing of macro
// processing in CUDA TUs forced this consolidation.
// Edit MAX_CANDIDATES per build to sweep (f, C). Comment the whole block to
// restore V3.2 production defaults (K=M=2048, f=3072, C=6144).
// Experiment dir: ablation_study/gvr_phase_timing/06_preidx_deep_dive/
//                 08_v4_K512_realbench/
// ============================================================================
#define HEURISTIC_TOP_K 512
#define HEURISTIC_SIZE_M 512
#define HEURISTIC_F_TARGET 2560
#define HEURISTIC_MAX_CANDIDATES 4096 // sweep: {6144, 5120, 4096}
// ============================================================================

inline constexpr int kHeuristicTopK = HEURISTIC_TOP_K;
inline constexpr int kHeuristicSize = HEURISTIC_SIZE_M;

/// Launch heuristic TopK decode kernel.
/// @param scratchValues Caller-owned buffer of size [numRows * topK] floats.
///        Required for CUDA Graph compatibility — must have a stable device address.
void launchHeuristicTopKDecode(float const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    float* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
