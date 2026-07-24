#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "kernel.h"

void run(
    const torch::Tensor& logits,
    const torch::Tensor& pre_idx,
    int64_t n_valid,
    torch::Tensor indices) {
  gvr_topk_launcher(
      logits.data_ptr<float>(),
      pre_idx.data_ptr<int>(),
      static_cast<int>(n_valid),
      indices.data_ptr<int>(),
      static_cast<int>(logits.size(0)),
      static_cast<int>(logits.size(1)),
      static_cast<int>(pre_idx.size(1)),
      at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "Prior-seeded exact GVR top-k (CUDA)");
}

