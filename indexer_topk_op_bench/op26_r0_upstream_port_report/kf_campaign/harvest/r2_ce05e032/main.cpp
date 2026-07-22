#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void gvr_topk_launch(const float* logits, const int* pre_idx, int* out, int npad, int K,
                     cudaStream_t stream);

void run(torch::Tensor logits, torch::Tensor pre_idx, int64_t n_valid, torch::Tensor indices) {
  (void)n_valid;  // pad is the lowest float and can never enter the top-k; kernel scans npad
  gvr_topk_launch(logits.data_ptr<float>(), pre_idx.data_ptr<int>(), indices.data_ptr<int>(),
                  (int)logits.size(1), (int)pre_idx.size(1),
                  at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "GVR top-k indexer decode (fp32, BS=1, B200)");
}
