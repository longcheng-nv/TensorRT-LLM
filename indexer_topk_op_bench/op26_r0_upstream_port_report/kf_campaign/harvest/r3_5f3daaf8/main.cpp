#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void topk_launch(const float* logits, const int* pre_idx, int n, int k, int* out,
                 cudaStream_t stream);

void run(torch::Tensor logits, torch::Tensor pre_idx, int64_t n_valid,
         torch::Tensor indices) {
    topk_launch(logits.data_ptr<float>(), pre_idx.data_ptr<int>(), (int)n_valid,
                (int)indices.size(1), indices.data_ptr<int>(),
                at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Exact CUDA radix top-k");
}
