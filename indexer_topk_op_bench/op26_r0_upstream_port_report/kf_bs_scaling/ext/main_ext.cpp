// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Batched host wrapper for the compB BS>1 extension (kernel_ext.cu):
// one call = one batch. Small tiers batch via grid.y; large-n batches via
// single-wave row teams (chunked into waves when BS*team exceeds the
// co-residency cap). See R3_LEDGER "BS>1 extension design analysis".
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

void topk_launch_ext(const float* logits, long W, int n, int k, int* out,
                     int BS, cudaStream_t stream);
void topk_ext_info(int n, int k, int BS, int info[5]);

void run_batch_ext(torch::Tensor logits, int64_t n_valid,
                   torch::Tensor indices) {
    TORCH_CHECK(logits.is_contiguous() && indices.is_contiguous());
    auto stream = at::cuda::getCurrentCUDAStream();
    topk_launch_ext(logits.data_ptr<float>(), (long)logits.size(1),
                    (int)n_valid, (int)indices.size(1),
                    indices.data_ptr<int>(), (int)logits.size(0),
                    stream.stream());
}

std::vector<int64_t> ext_info(int64_t n, int64_t k, int64_t bs) {
    int info[5] = {0, 0, 0, 0, 0};
    topk_ext_info((int)n, (int)k, (int)bs, info);
    return {info[0], info[1], info[2], info[3], info[4]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_batch_ext", &run_batch_ext,
          "compB BS>1 extension: batched exact top-k, one call per batch");
    m.def("ext_info", &ext_info,
          "(path, team, cap, rows_per_wave, waves) for (n, k, BS)");
}
