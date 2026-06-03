// Torch cpp_extension binding for the standalone GVR Heuristic Top-K
// decode kernel. Exposes two Python entry points:
//
//   gvr_kernel.gvr_topk_decode(logits, preIdx, seq_lens, K,
//                              compress_ratio=4, next_n=1)
//       -> (indices, values)
//       Allocates output tensors on every call. Convenient for one-off
//       use but per-call alloc adds overhead at high BS.
//
//   gvr_kernel.gvr_topk_decode_into(logits, preIdx, seq_lens,
//                                   indices_out, values_out,
//                                   K, compress_ratio=4, next_n=1)
//       -> None  (writes into caller-provided tensors)
//       Pre-allocate `indices_out [BS, K] int32` and `values_out [BS, K]`
//       (matching logits.dtype) once; reuse across reps to remove the
//       per-call alloc + extra copy_() overhead. Recommended for perfsim
//       bench loops and for cycle-accurate timing.
//
// Argument shapes (V4 Pro contract):
//   logits     : (BS, Npad)  fp32 / bf16 / fp16, post-compress index space.
//   preIdx     : (BS, K)     int32, V4 caller offset = 0 — kernel reads
//                            preIdx[i] directly when compress_ratio != 1.
//   seq_lens   : (BS/next_n,) int32  = N * compress_ratio + (next_n - 1).
//   K          : 512 / 1024 / 2048.
//   compress_ratio : 1 (V3.2) or 4 (V4 indexer).
//   next_n     : speculative draft length (default 1).

#include "heuristicTopKDecode.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <tuple>

using tensorrt_llm_gvr_local::kernels::launchHeuristicTopKDecode;

namespace {

// Shared kernel dispatch. Caller has already validated tensor shapes and
// computed BS/Npad. indices_out and values_out must already be allocated.
void dispatch(torch::Tensor const& logits,
              torch::Tensor const& preIdx,
              torch::Tensor const& seq_lens,
              torch::Tensor& indices_out,
              torch::Tensor& values_out,
              int64_t BS, int64_t Npad,
              int64_t K, int64_t compress_ratio, int64_t next_n)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    if (logits.dtype() == torch::kFloat32)
    {
        launchHeuristicTopKDecode(
            logits.data_ptr<float>(),
            seq_lens.data_ptr<int>(),
            preIdx.data_ptr<int>(),
            indices_out.data_ptr<int>(),
            values_out.data_ptr<float>(),
            static_cast<int>(Npad), static_cast<int>(next_n),
            static_cast<int>(K), static_cast<int>(K), static_cast<int>(K),
            static_cast<int>(BS), static_cast<int>(compress_ratio), stream);
    }
    else if (logits.dtype() == torch::kBFloat16)
    {
        launchHeuristicTopKDecode(
            reinterpret_cast<__nv_bfloat16 const*>(logits.data_ptr<at::BFloat16>()),
            seq_lens.data_ptr<int>(),
            preIdx.data_ptr<int>(),
            indices_out.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(values_out.data_ptr<at::BFloat16>()),
            static_cast<int>(Npad), static_cast<int>(next_n),
            static_cast<int>(K), static_cast<int>(K), static_cast<int>(K),
            static_cast<int>(BS), static_cast<int>(compress_ratio), stream);
    }
    else if (logits.dtype() == torch::kFloat16)
    {
        launchHeuristicTopKDecode(
            reinterpret_cast<__half const*>(logits.data_ptr<at::Half>()),
            seq_lens.data_ptr<int>(),
            preIdx.data_ptr<int>(),
            indices_out.data_ptr<int>(),
            reinterpret_cast<__half*>(values_out.data_ptr<at::Half>()),
            static_cast<int>(Npad), static_cast<int>(next_n),
            static_cast<int>(K), static_cast<int>(K), static_cast<int>(K),
            static_cast<int>(BS), static_cast<int>(compress_ratio), stream);
    }
    else
    {
        TORCH_CHECK(false, "logits dtype must be float32 / bfloat16 / float16, "
                           "got ", logits.dtype());
    }
}

// Validate shared shape/contract constraints. Output-tensor checks are
// done by the caller of dispatch() if applicable.
void validate_inputs(torch::Tensor const& logits,
                     torch::Tensor const& preIdx,
                     torch::Tensor const& seq_lens,
                     int64_t K, int64_t next_n)
{
    TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [BS, Npad]");
    TORCH_CHECK(preIdx.dim() == 2, "preIdx must be 2-D [BS, K]");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1-D [BS/next_n]");
    TORCH_CHECK(logits.is_cuda() && preIdx.is_cuda() && seq_lens.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(preIdx.is_contiguous(), "preIdx must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
    TORCH_CHECK(preIdx.dtype() == torch::kInt32, "preIdx must be int32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(K == 512 || K == 1024 || K == 2048,
                "K must be 512 / 1024 / 2048 (got ", K, ")");

    int64_t BS = logits.size(0);
    TORCH_CHECK(preIdx.size(0) == BS, "preIdx BS mismatch");
    TORCH_CHECK(preIdx.size(1) == K, "preIdx K mismatch");
    TORCH_CHECK(seq_lens.size(0) == BS / next_n,
                "seq_lens length must be BS / next_n (", BS, "/", next_n,
                "), got ", seq_lens.size(0));
}


std::tuple<torch::Tensor, torch::Tensor> gvr_topk_decode(
    torch::Tensor logits,
    torch::Tensor preIdx,
    torch::Tensor seq_lens,
    int64_t K,
    int64_t compress_ratio,
    int64_t next_n)
{
    validate_inputs(logits, preIdx, seq_lens, K, next_n);
    int64_t BS = logits.size(0);
    int64_t Npad = logits.size(1);
    auto indices = torch::empty({BS, K},
        torch::dtype(torch::kInt32).device(logits.device()));
    auto values = torch::empty({BS, K},
        torch::dtype(logits.scalar_type()).device(logits.device()));
    dispatch(logits, preIdx, seq_lens, indices, values, BS, Npad,
             K, compress_ratio, next_n);
    return std::make_tuple(std::move(indices), std::move(values));
}


void gvr_topk_decode_into(
    torch::Tensor logits,
    torch::Tensor preIdx,
    torch::Tensor seq_lens,
    torch::Tensor indices_out,
    torch::Tensor values_out,
    int64_t K,
    int64_t compress_ratio,
    int64_t next_n)
{
    validate_inputs(logits, preIdx, seq_lens, K, next_n);
    int64_t BS = logits.size(0);
    int64_t Npad = logits.size(1);

    TORCH_CHECK(indices_out.is_cuda() && values_out.is_cuda(),
                "indices_out / values_out must be CUDA");
    TORCH_CHECK(indices_out.is_contiguous() && values_out.is_contiguous(),
                "indices_out / values_out must be contiguous");
    TORCH_CHECK(indices_out.dtype() == torch::kInt32,
                "indices_out must be int32");
    TORCH_CHECK(values_out.dtype() == logits.dtype(),
                "values_out dtype must match logits");
    TORCH_CHECK(indices_out.dim() == 2 && indices_out.size(0) == BS
                && indices_out.size(1) == K,
                "indices_out must be [BS=", BS, ", K=", K, "]");
    TORCH_CHECK(values_out.dim() == 2 && values_out.size(0) == BS
                && values_out.size(1) == K,
                "values_out must be [BS=", BS, ", K=", K, "]");

    dispatch(logits, preIdx, seq_lens, indices_out, values_out, BS, Npad,
             K, compress_ratio, next_n);
}

}  // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gvr_topk_decode", &gvr_topk_decode,
          "Standalone GVR Heuristic Top-K decode (V4 Pro contract). "
          "Allocates and returns (indices [BS,K] int32, values [BS,K] same dtype).",
          py::arg("logits"), py::arg("preIdx"), py::arg("seq_lens"),
          py::arg("K"), py::arg("compress_ratio") = 4, py::arg("next_n") = 1);

    m.def("gvr_topk_decode_into", &gvr_topk_decode_into,
          "GVR Heuristic Top-K decode with caller-provided output tensors. "
          "Avoids per-call torch::empty + copy_() overhead — recommended "
          "for perfsim bench loops.",
          py::arg("logits"), py::arg("preIdx"), py::arg("seq_lens"),
          py::arg("indices_out"), py::arg("values_out"),
          py::arg("K"), py::arg("compress_ratio") = 4, py::arg("next_n") = 1);
}
