#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "launchers.h"

void run(
    const torch::Tensor& logits,
    const torch::Tensor& pre_idx,
    int64_t n_valid,
    torch::Tensor& indices) {
  const int n = static_cast<int>(n_valid);
  const int k = static_cast<int>(pre_idx.size(1));
  const int npad = static_cast<int>(logits.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (npad >= 8192) {
    histogram_launcher(
        logits.data_ptr<float>(), pre_idx.data_ptr<int>(), npad, k,
        indices.data_ptr<int>(), stream);
  } else {
    champion_launcher(
        logits.data_ptr<float>(), pre_idx.data_ptr<int>(),
        indices.data_ptr<int>(), npad, n, k, stream);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "Hybrid hint-seeded exact decode top-k");
}
