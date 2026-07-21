#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void topk_launch(const float* logits, int n, int k, int* out, cudaStream_t stream);

void run(torch::Tensor logits, torch::Tensor pre_idx, int64_t n_valid, torch::Tensor indices) {
    int n = (int)n_valid;
    int k = (int)indices.size(1);
    topk_launch(logits.data_ptr<float>(), n, k, indices.data_ptr<int>(),
                at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "top-k indexer decode (CUDA radix-select)");
}
