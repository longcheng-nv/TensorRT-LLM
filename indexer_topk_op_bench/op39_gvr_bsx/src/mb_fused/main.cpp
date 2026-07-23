// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" void mb_fused_launch(const float*, const float*, float*, int*, int*, int*,
                                int*, int, int, int, int, cudaStream_t);

void run(torch::Tensor logits, torch::Tensor thr, torch::Tensor cand_val,
         torch::Tensor cand_idx, torch::Tensor cnt, torch::Tensor done,
         torch::Tensor out, int64_t K, int64_t chunks) {
  mb_fused_launch(logits.data_ptr<float>(), thr.data_ptr<float>(),
                  cand_val.data_ptr<float>(), cand_idx.data_ptr<int>(),
                  cnt.data_ptr<int>(), done.data_ptr<int>(), out.data_ptr<int>(),
                  (int)logits.size(1), (int)K, (int)logits.size(0), (int)chunks,
                  at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "op39 iter2: fused 1-pass collect + last-CTA reduce");
}
