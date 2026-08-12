/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <mutex>

#include "kernel.h"

/* B2: one zero-initialised slab workspace per device, lazily allocated through
   the torch caching allocator. Concurrent STREAMS on one device that may both
   take the multi-CTA SPLIT path must pass their own workspace via run_ws().
   Hot path is a plain array load; the mutex is slow-path only. */
#define GVR_MAX_DEV 64
static std::atomic<void*> ws_ptr[GVR_MAX_DEV];
static at::Tensor ws_keep[GVR_MAX_DEV];   /* keeps the cached tensors alive */

static void* default_workspace(const at::Tensor& ref) {
    const int d = ref.get_device();
    TORCH_CHECK(d >= 0 && d < GVR_MAX_DEV, "device index out of range: ", d);
    void* p = ws_ptr[d].load(std::memory_order_acquire);
    if (p) return p;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    p = ws_ptr[d].load(std::memory_order_acquire);
    if (p) return p;
    ws_keep[d] = at::zeros({(int64_t)gvr_topk_workspace_bytes()},
                           at::TensorOptions().dtype(at::kByte).device(ref.device()));
    p = ws_keep[d].data_ptr();
    ws_ptr[d].store(p, std::memory_order_release);
    return p;
}

static void run_impl(const at::Tensor& logits, const at::Tensor& pre_idx,
                     pybind11::object& n_valid, at::Tensor& indices, void* ws) {
    /* B1/host hardening: the binding used to accept anything and reinterpret.
       All checks are predicted-taken branches — sub-100ns on the hot path. */
    TORCH_CHECK(logits.is_cuda() && pre_idx.is_cuda() && indices.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(logits.scalar_type() == at::kFloat, "logits must be float32");
    TORCH_CHECK(pre_idx.scalar_type() == at::kInt, "pre_idx must be int32");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
    TORCH_CHECK(logits.dim() == 2 && pre_idx.dim() == 2 && indices.dim() == 2,
                "logits/pre_idx/indices must be 2-D");
    TORCH_CHECK(logits.is_contiguous() && pre_idx.is_contiguous() &&
                indices.is_contiguous(), "tensors must be contiguous");

    /* Exception-free unwrap on the int fast path (a throwing cast costs
       microseconds of host time per call); tensor n_valid forces a sync,
       which is illegal under CUDA graph capture — fail loudly instead of
       crashing the capture (B1d). */
    int64_t nv = 0;
    if (pybind11::isinstance<at::Tensor>(n_valid)) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &cap);
        TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                    "tensor n_valid requires a D2H sync, illegal under CUDA "
                    "graph capture — pass n_valid as a python int");
        nv = n_valid.cast<at::Tensor>().item<int64_t>();
    } else {
        nv = n_valid.cast<int64_t>();
    }
    const int b    = (int)logits.size(0);
    const int npad = (int)logits.size(1);
    const int k    = (int)pre_idx.size(1);
    if (b == 0) return;                    /* empty batch: no-op (B1f) */
    /* float4 row loads require npad%4 and a 16B-aligned base (B1e); the
       production layout pads row stride to 64 elements. */
    TORCH_CHECK((npad & 3) == 0, "npad (logits stride) must be a multiple of "
                "4, got ", npad);
    TORCH_CHECK((reinterpret_cast<uintptr_t>(logits.data_ptr()) & 15u) == 0,
                "logits base must be 16-byte aligned (storage-offset views "
                "break the float4 row loads)");
    TORCH_CHECK(pre_idx.size(0) == b && indices.size(0) == b,
                "batch dims must match: logits ", b, " pre_idx ", pre_idx.size(0),
                " indices ", indices.size(0));
    /* k is taken from pre_idx width; a narrower `indices` would make the emit
       phase write past the output tensor. */
    TORCH_CHECK(indices.size(1) >= k, "indices width ", indices.size(1),
                " < k=", k, " (k is pre_idx.size(1))");
    TORCH_CHECK(nv >= 0, "n_valid must be non-negative, got ", nv);
    /* clamp in 64-bit BEFORE the int cast: nv = 2^31 would truncate negative */
    int n = (int)std::min<int64_t>(nv, (int64_t)npad);

    cudaError_t err = gvr_topk_launch(
        logits.data_ptr<float>(), pre_idx.data_ptr<int>(),
        indices.data_ptr<int>(), b, n, npad, k, ws,
        at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(err == cudaSuccess, "gvr_topk launch failed (b=", b, " n=", n,
                " npad=", npad, " k=", k, "): ", cudaGetErrorString(err));
}

/* Fast 4-arg form: signature-identical to the original candidate. */
void run(const at::Tensor& logits, const at::Tensor& pre_idx,
         pybind11::object n_valid, at::Tensor indices) {
    run_impl(logits, pre_idx, n_valid, indices, default_workspace(logits));
}

/* Explicit-workspace form for multi-stream callers. */
void run_ws(const at::Tensor& logits, const at::Tensor& pre_idx,
            pybind11::object n_valid, at::Tensor indices, at::Tensor workspace) {
    TORCH_CHECK(workspace.is_cuda() &&
                workspace.get_device() == logits.get_device(),
                "workspace must be a CUDA tensor on the same device");
    TORCH_CHECK((size_t)(workspace.numel() * workspace.element_size()) >=
                gvr_topk_workspace_bytes(),
                "workspace too small: need ", gvr_topk_workspace_bytes(), " bytes");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(workspace.data_ptr()) & 7u) == 0,
                "workspace must be 8-byte aligned");
    run_impl(logits, pre_idx, n_valid, indices, workspace.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "GVR exact top-k decode (CUDA, prod-hardened)");
    m.def("run_ws", &run_ws,
          "GVR exact top-k decode with explicit slab workspace (multi-stream)");
    m.def("workspace_bytes", []() { return (int64_t)gvr_topk_workspace_bytes(); },
          "slab workspace size in bytes (per concurrent stream)");
}
