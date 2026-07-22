// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Batched host wrapper for the compB BS>1 extension (kernel_ext.cu):
// one call = one batch. Small tiers batch via grid.y; large-n batches via
// single-wave row teams (chunked into waves when BS*team exceeds the
// co-residency cap). See R3_LEDGER "BS>1 extension design analysis".
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

void topk_launch_ext(const float* logits, long W, int n, int k, int* out,
                     int BS, cudaStream_t stream);
void topk_launch_ext_v(const float* logits, long W, int n, int k, int* out,
                       int BS, int minb, cudaStream_t stream);
void topk_launch_pq_v(const float* logits, long W, int n, int k, int* out,
                      int BS, int minb, cudaStream_t stream);
void topk_launch_tp(const float* logits, long W, int n, int k, int* out,
                    int BS, cudaStream_t stream);
void topk_launch_tp2(const float* logits, long W, int n, int k, int* out,
                     int BS, cudaStream_t stream);
unsigned int topk_tp2_fallbacks();
void topk_ext_info(int n, int k, int BS, int info[5]);
void topk_fast_stats(int minb, int out5[5]);
void topk_pq_stats(int minb, int out5[5]);

void run_batch_ext(torch::Tensor logits, int64_t n_valid,
                   torch::Tensor indices) {
    TORCH_CHECK(logits.is_contiguous() && indices.is_contiguous());
    auto stream = at::cuda::getCurrentCUDAStream();
    topk_launch_ext(logits.data_ptr<float>(), (long)logits.size(1),
                    (int)n_valid, (int)indices.size(1),
                    indices.data_ptr<int>(), (int)logits.size(0),
                    stream.stream());
}

void run_batch_ext_v(torch::Tensor logits, int64_t n_valid,
                     torch::Tensor indices, int64_t minb) {
    TORCH_CHECK(logits.is_contiguous() && indices.is_contiguous());
    auto stream = at::cuda::getCurrentCUDAStream();
    topk_launch_ext_v(logits.data_ptr<float>(), (long)logits.size(1),
                      (int)n_valid, (int)indices.size(1),
                      indices.data_ptr<int>(), (int)logits.size(0),
                      (int)minb, stream.stream());
}

std::vector<int64_t> fast_stats(int64_t minb) {
    int s[5] = {0, 0, 0, 0, 0};
    topk_fast_stats((int)minb, s);
    return {s[0], s[1], s[2], s[3], s[4]};
}

void run_batch_tp(torch::Tensor logits, int64_t n_valid,
                  torch::Tensor indices) {
    TORCH_CHECK(logits.is_contiguous() && indices.is_contiguous());
    auto stream = at::cuda::getCurrentCUDAStream();
    topk_launch_tp(logits.data_ptr<float>(), (long)logits.size(1),
                   (int)n_valid, (int)indices.size(1),
                   indices.data_ptr<int>(), (int)logits.size(0),
                   stream.stream());
}

void run_batch_tp2(torch::Tensor logits, int64_t n_valid,
                   torch::Tensor indices) {
    TORCH_CHECK(logits.is_contiguous() && indices.is_contiguous());
    auto stream = at::cuda::getCurrentCUDAStream();
    topk_launch_tp2(logits.data_ptr<float>(), (long)logits.size(1),
                    (int)n_valid, (int)indices.size(1),
                    indices.data_ptr<int>(), (int)logits.size(0),
                    stream.stream());
}

int64_t tp2_fallbacks() { return (int64_t)topk_tp2_fallbacks(); }

void run_batch_pq(torch::Tensor logits, int64_t n_valid,
                  torch::Tensor indices, int64_t minb) {
    TORCH_CHECK(logits.is_contiguous() && indices.is_contiguous());
    auto stream = at::cuda::getCurrentCUDAStream();
    topk_launch_pq_v(logits.data_ptr<float>(), (long)logits.size(1),
                     (int)n_valid, (int)indices.size(1),
                     indices.data_ptr<int>(), (int)logits.size(0),
                     (int)minb, stream.stream());
}

std::vector<int64_t> pq_stats(int64_t minb) {
    int s[5] = {0, 0, 0, 0, 0};
    topk_pq_stats((int)minb, s);
    return {s[0], s[1], s[2], s[3], s[4]};
}

std::vector<int64_t> ext_info(int64_t n, int64_t k, int64_t bs) {
    int info[5] = {0, 0, 0, 0, 0};
    topk_ext_info((int)n, (int)k, (int)bs, info);
    return {info[0], info[1], info[2], info[3], info[4]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_batch_ext", &run_batch_ext,
          "compB BS>1 extension: batched exact top-k, one call per batch");
    m.def("ext_info", &ext_info,
          "(path, team, cap, rows_per_wave, waves) for (n, k, BS)");
    m.def("run_batch_ext_v", &run_batch_ext_v,
          "extension with register-diet variant minb in {1,2,3,4}");
    m.def("fast_stats", &fast_stats,
          "(numRegs, staticSmem, localBytes, active_default, active_maxcarveout)"
          " for topk_fast<minb>");
    m.def("run_batch_tp", &run_batch_tp,
          "D1 throughput arm: barrier-free 3-kernel hist/collect/finish");
    m.def("run_batch_tp2", &run_batch_tp2,
          "D2 sampled-estimate single-pass arm");
    m.def("tp2_fallbacks", &tp2_fallbacks,
          "read-and-reset D2 fallback-row counter");
    m.def("run_batch_pq", &run_batch_pq,
          "B' persistent-queue: one launch consumes the whole batch");
    m.def("pq_stats", &pq_stats, "resource stats for topk_fast_pq<minb>");
}
