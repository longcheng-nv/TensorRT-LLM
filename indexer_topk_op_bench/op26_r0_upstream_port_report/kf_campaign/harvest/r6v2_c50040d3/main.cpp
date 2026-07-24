#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

void run(const torch::Tensor& logits, const torch::Tensor& pre_idx, int64_t n_valid, torch::Tensor& indices){
    int b = logits.size(0);
    int npad = logits.size(1);
    int k = indices.size(1);
    topk_launcher(
        logits.data_ptr<float>(),
        indices.data_ptr<int>(),
        b, (int)n_valid, npad, k,
        at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("run", &run, "Exact tie-robust batched top-k (radix-select, CUDA)");
}
