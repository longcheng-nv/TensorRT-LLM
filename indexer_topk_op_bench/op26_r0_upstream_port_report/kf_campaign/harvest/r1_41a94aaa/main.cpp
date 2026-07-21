#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" void topk_launch(const void* logits, void* out, int npad, int n,
                            int k, cudaStream_t stream);

void run(const torch::Tensor& logits, const torch::Tensor& pre_idx,
         int64_t n_valid, const torch::Tensor& indices) {
  const int npad = (int)logits.size(1);
  const int k = (int)indices.size(1);
  topk_launch(logits.data_ptr(), indices.data_ptr(), npad, (int)n_valid, k,
              at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "indexer top-k decode");
}
