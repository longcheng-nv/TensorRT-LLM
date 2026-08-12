/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuda_runtime.h>
#include <cstddef>

/* B2: bytes of caller-provided (zero-initialised, 8B-aligned) global workspace
   for the multi-CTA SPLIT slab. One workspace per concurrent stream. */
size_t gvr_topk_workspace_bytes();

/* Launch-or-prior-async error is returned (B1); cudaSuccess on clean launch. */
cudaError_t gvr_topk_launch(const float* logits, const int* pre_idx, int* out,
                            int b, int n, int npad, int k, void* workspace,
                            cudaStream_t stream);
