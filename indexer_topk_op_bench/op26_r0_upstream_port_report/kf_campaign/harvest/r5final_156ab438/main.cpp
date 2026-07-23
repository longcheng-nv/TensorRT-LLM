#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" void topk_launch(const float* logits, const int* pre_idx, int* out, int b, int npad,
                            int n, int K, cudaStream_t stream);

void run(torch::Tensor logits, torch::Tensor pre_idx, int64_t n_valid, torch::Tensor indices) {
  int b = (int)logits.size(0);
  int npad = (int)logits.size(1);
  int K = (int)pre_idx.size(1);
  topk_launch(logits.data_ptr<float>(), pre_idx.data_ptr<int>(), indices.data_ptr<int>(), b,
              npad, (int)n_valid, K, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }




