#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void topk_launch(const float* logits, int n, int k, int* out,
                 cudaStream_t stream);

void run(const torch::Tensor& logits, const torch::Tensor& pre_idx,
         int64_t n_valid, const torch::Tensor& indices) {
    topk_launch(logits.data_ptr<float>(), (int)n_valid,
                (int)indices.size(1), indices.data_ptr<int>(),
                at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "exact fp32 radix top-k");
}
