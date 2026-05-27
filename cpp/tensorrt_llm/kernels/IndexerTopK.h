/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
// Number of blocks-per-row used by the multi-block split + merge dispatch path of
// invokeIndexerTopKDecode. Returns 1 when the single-block path is preferred.
// Callers that allocate aux buffers must use this same helper to size them, and
// must pass the same splitWorkThreshold they will pass to invokeIndexerTopKDecode
// (a value <= 0 selects the internal default).
int computeIndexerTopKDecodeBlocksPerRow(int numRows, int numColumns, int splitWorkThreshold = 0);

/// fp32 indexer TopK decode — L2-aware BS-threshold dispatcher with four
/// fallback tiers:
///   - GVR Heuristic    (preIdx provided, kSeqSmall ≤ N < splitWork, BS < kBsLarge, K ∈ {512,1024,2048})
///   - Insertion sort   (N < kSortingAlgorithmThreshold)
///   - Radix sort       (kSortingAlgorithmThreshold ≤ N < splitWork)
///   - Radix split-work (N ≥ splitWork — uses outLogitsAux / outIndicesAux)
void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, float* heuristicScratch = nullptr, int const compressRatio = 1,
    cudaStream_t const stream = 0);

/// bf16 indexer TopK decode — same dispatch axes as the fp32 entry, except
/// kBsL2 uses sizeof(__nv_bfloat16) bytes/elem (L2 footprint is half) and
/// the split-work tier is unsupported (the bf16/fp16 entry does not expose
/// the float aux buffers required for split-work). Insertion + radix tiers
/// share topKPerRowDecode with fp32 — histogram and sort run on float keys
/// after a static_cast<float>(InputT) at HBM-read sites.
///
/// Aborts with TLLM_CHECK if numColumns ≥ splitWorkThreshold; callers in
/// that regime must use the fp32 entry.
void invokeIndexerTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int* indices,
    int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const next_n, int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0,
    int const preIdxCount = 0, __nv_bfloat16* heuristicScratch = nullptr, int const compressRatio = 1,
    cudaStream_t const stream = 0);

/// fp16 indexer TopK decode — see bf16 overload for dispatcher contract.
void invokeIndexerTopKDecode(__half const* logits, int const* seqLens, int* indices, int const splitWorkThreshold,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const next_n,
    int const topK = 2048, int const* preIdx = nullptr, int const preIdxStride = 0, int const preIdxCount = 0,
    __half* heuristicScratch = nullptr, int const compressRatio = 1, cudaStream_t const stream = 0);

/// Indexer TopK prefill — fp32 dispatcher. Dispatches to GVR Heuristic when
/// a caller-owned heuristicScratch is provided and `(topK, compressRatio,
/// stride1, numColumns, numRows)` satisfy the same dispatch envelope as
/// decode; otherwise falls back to the existing Radix / Insertion path
/// (Radix is fp32-only).
///
/// GVR prefill synthesizes preIdx inside the kernel from `(N_r, K, compressRatio)`:
///   compressRatio == 1 (V3.2)         → BaseShift mode, idx = (N_r - K) + i
///   compressRatio == 4 (V4 Flash/Pro) → ConstIdentity mode, idx = i
/// No external preIdx tensor is required from the caller (in contrast to
/// decode where preIdx is the previous step's top-K output).
///
/// @param heuristicScratch  Caller-owned [topK] fp32 buffer shared across
///                          all CTAs (kernel-internal write-only, no reader).
///                          Mandatory for GVR path, CUDA-Graph-safe stable
///                          address. Pass nullptr to force the Radix
///                          fallback regardless of other gating.
/// @param compressRatio     KV compression ratio: 1 = V3.2 (raw KV space),
///                          4 = V4 Flash / Pro (compressed token-index space).
void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    float* heuristicScratch, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const topK = 2048, int const compressRatio = 1, cudaStream_t const stream = 0);

/// bf16 indexer TopK prefill — GVR-Heuristic only (no Radix fallback; the
/// existing `topKPerRowPrefill` path is fp32-only). API symmetry with the
/// decode dispatcher (which also has bf16/fp16 entries). Aborts with
/// TLLM_CHECK if `heuristicScratch == nullptr` or the GVR envelope is not
/// satisfied. Use the fp32 overload for callers that want a Radix fallback.
void invokeIndexerTopKPrefill(__nv_bfloat16 const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    __nv_bfloat16* heuristicScratch, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const topK = 2048, int const compressRatio = 1, cudaStream_t const stream = 0);

/// fp16 indexer TopK prefill — see bf16 overload for dispatcher contract.
void invokeIndexerTopKPrefill(__half const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    __half* heuristicScratch, int const numRows, int const numColumns, int const stride0, int const stride1,
    int const topK = 2048, int const compressRatio = 1, cudaStream_t const stream = 0);

/// Returns true iff invokeIndexerTopKDecode would route to the GVR Heuristic
/// kernel for this (numRows, numColumns, topK) triple, assuming valid preIdx
/// is provided and stride1 == 1. Useful for callers that need to provision a
/// preIdx tensor or heuristicScratch buffer only when GVR will be selected.
///
/// Mirrors the gating logic of the dispatcher: K ∈ {512, 1024, 2048},
/// numColumns ∈ [kSeqSmall, splitWorkThreshold), numRows < kBsLarge, where
/// kBsLarge = min(kBsWave, kBsL2) and kBsL2 scales with bytesPerElem.
///
/// @param numRows         logits rows (batch · next_n)
/// @param numColumns      logits columns (max sequence length)
/// @param topK            requested output size
/// @param bytesPerElem    element size of logits (4 for fp32, 2 for bf16/fp16)
bool canIndexerTopKDecodeUseGvr(int numRows, int numColumns, int topK, int bytesPerElem = 4);

/// Returns true iff invokeIndexerTopKPrefill would route to the GVR Heuristic
/// kernel for this (numRows, numColumns, topK, compressRatio) tuple, assuming
/// a valid heuristicScratch is provided and stride1 == 1. Mirrors the gating
/// logic of the prefill dispatcher: K ∈ {512, 1024, 2048}, compressRatio ∈
/// {1, 4}, numColumns ∈ [kSeqSmall, splitWorkThreshold), numRows < kBsLarge.
bool canIndexerTopKPrefillUseGvr(int numRows, int numColumns, int topK, int compressRatio);

} // namespace kernels

TRTLLM_NAMESPACE_END
