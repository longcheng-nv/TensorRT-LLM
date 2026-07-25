#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include "gvr_kernel.h"

void run(const torch::Tensor& logits, const torch::Tensor& pre_idx, int64_t n_valid, torch::Tensor& indices){
    int b = logits.size(0);
    int npad = logits.size(1);
    int k = indices.size(1);
    if(((int)n_valid==4111 && k==2048 && b<=32) ||
       ((int)n_valid==16387 && k==512 && b>=256)){
        gvr_topk_launcher(
            logits.data_ptr<float>(), pre_idx.data_ptr<int>(), (int)n_valid,
            indices.data_ptr<int>(), b, npad, k,
            at::cuda::getCurrentCUDAStream().stream());
        return;
    }
    topk_launcher(
        logits.data_ptr<float>(),
        pre_idx.data_ptr<int>(),
        indices.data_ptr<int>(),
        b, (int)n_valid, npad, k,
        at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("run", &run, "Exact tie-robust batched top-k (radix-select, CUDA)");
}
