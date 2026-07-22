// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Batched host wrapper for the KF R3 ship kernel (r3_compB, BS=1 single-row
// contract): loops rows of a [BS, W] logits tensor, one topk_launch per row
// on the SAME stream. Kernel source untouched (kernel.cu == harvest/r3_compB).
// Sequential same-stream launches are REQUIRED: the large-n path shares one
// static scratch buffer + generation counter across launches (see kernel.cu),
// so concurrent per-row streams would race on it.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void topk_launch(const float* logits, int n, int k, int* out,
                 cudaStream_t stream);

void run_batch(torch::Tensor logits, int64_t n_valid, torch::Tensor indices) {
    TORCH_CHECK(logits.is_contiguous() && indices.is_contiguous());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t bs = logits.size(0);
    const int64_t w = logits.size(1);
    const int64_t k = indices.size(1);
    const float* lp = logits.data_ptr<float>();
    int* op = indices.data_ptr<int>();
    for (int64_t r = 0; r < bs; ++r) {
        topk_launch(lp + r * w, (int)n_valid, (int)k, op + r * k,
                    stream.stream());
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_batch", &run_batch, "compB exact top-k, per-row loop over BS");
}
