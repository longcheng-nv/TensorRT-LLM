#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void topk_launcher(const float* logits, int* indices,
                   int b, int npad, int n_valid, int k,
                   cudaStream_t stream);

void run(const torch::Tensor& logits,
         const torch::Tensor& pre_idx,
         int64_t n_valid,
         torch::Tensor& indices) {
    int b    = (int)logits.size(0);
    int npad = (int)logits.size(1);
    int k    = (int)indices.size(1);
    topk_launcher(
        logits.data_ptr<float>(),
        indices.data_ptr<int>(),
        b, npad, (int)n_valid, k,
        at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "indexer topk decode (CUDA)");
}
