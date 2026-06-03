/*
 * Standalone GVR Heuristic Top-K decode launcher header.
 * Adapted verbatim from TensorRT-LLM's
 *   cpp/tensorrt_llm/kernels/heuristicTopKDecode.h
 * with only the namespace macro source swapped for `trtllm_stubs.h`.
 */
#pragma once

#include "trtllm_stubs.h"

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
/// See heuristicTopKDecode.cu / heuristic_topk.cuh for full algorithm doc.
void launchHeuristicTopKDecode(float const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    float* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int compressRatio, cudaStream_t stream);

/// Launch heuristic TopK decode kernel — bf16 input.
void launchHeuristicTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __nv_bfloat16* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int compressRatio, cudaStream_t stream);

/// Launch heuristic TopK decode kernel — fp16 input.
void launchHeuristicTopKDecode(__half const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __half* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    int compressRatio, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
