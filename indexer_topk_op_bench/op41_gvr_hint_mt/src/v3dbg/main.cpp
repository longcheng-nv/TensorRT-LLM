#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void gvr_topk_launch(const float* logits, const int* pre_idx, int* out, int npad, int K, int BS,
                     cudaStream_t stream);
void gvr_topk_launch_cfg(const float* logits, const int* pre_idx, int* out, int npad, int K,
                         int BS, int tb, int cs, int maxv, int ar, int hs, cudaStream_t stream);

void run(torch::Tensor logits, torch::Tensor pre_idx, int64_t n_valid, torch::Tensor indices) {
  (void)n_valid;  // pad is the lowest float and can never enter the top-k; kernel scans npad
  gvr_topk_launch(logits.data_ptr<float>(), pre_idx.data_ptr<int>(), indices.data_ptr<int>(),
                  (int)logits.size(1), (int)pre_idx.size(1), (int)logits.size(0),
                  at::cuda::getCurrentCUDAStream().stream());
}

void run_cfg(torch::Tensor logits, torch::Tensor pre_idx, int64_t n_valid,
             torch::Tensor indices, int64_t tb, int64_t cs, int64_t maxv, int64_t ar,
             int64_t hs) {
  (void)n_valid;
  gvr_topk_launch_cfg(logits.data_ptr<float>(), pre_idx.data_ptr<int>(),
                      indices.data_ptr<int>(), (int)logits.size(1), (int)pre_idx.size(1),
                      (int)logits.size(0), (int)tb, (int)cs, (int)maxv, (int)ar, (int)hs,
                      at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "GVR top-k indexer decode, batched rows (fp32, B200)");
  m.def("run_cfg", &run_cfg, "probe: explicit (cs, maxv, ar) variant");
}
