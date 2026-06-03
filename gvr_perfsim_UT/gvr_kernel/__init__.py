"""Standalone GVR Heuristic Top-K kernel for DSv4 Pro indexer.

JIT-builds a torch C++/CUDA extension on first import (~30-60 s the first
time; subsequent imports reuse the cached `.so`).

Use:
    from gvr_kernel import gvr_topk_decode
    indices, values = gvr_topk_decode(
        logits,         # (BS, Npad)  fp32 / bf16 / fp16
        preIdx,         # (BS, K)     int32
        seq_lens,       # (BS / next_n,)  int32 = N * compress_ratio + next_n - 1
        K=1024,         # 512 / 1024 / 2048
        compress_ratio=4,
        next_n=1,
    )

This is a self-contained extraction of TensorRT-LLM's `heuristic_topk.cuh`
+ `heuristicTopKDecode.{cu,h}` — no `import tensorrt_llm` required and no
`libth_common.so` needed. The kernel implementation itself is byte-for-byte
identical to the in-tree source; only the three TRT-LLM common-header
references (`TRTLLM_NAMESPACE_*`, `TLLM_CHECK_WITH_INFO`,
`getEnvEnablePDL`) are stubbed in `csrc/trtllm_stubs.h`.

Target hardware: NVIDIA Blackwell (B200 sm_100 / B300 sm_103).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_THIS_DIR = Path(__file__).resolve().parent
_CSRC = _THIS_DIR / "csrc"

# Cache the JIT build under the user's torch extension dir; override with
# TORCH_EXTENSIONS_DIR if desired (e.g. for read-only-NFS workspaces).
os.environ.setdefault("TORCH_EXTENSIONS_DIR",
                      str(_THIS_DIR / "_build"))

# Force Blackwell SASS (sm_100 = B200, sm_103 = B300). Other archs may
# build but the kernel is tuned only for Blackwell — see header notes.
_arch_flags = ["-gencode=arch=compute_100,code=sm_100",
               "-gencode=arch=compute_103,code=sm_103"]
_override = os.environ.get("GVR_KERNEL_ARCH_FLAGS")
if _override:
    _arch_flags = _override.split()


_module = None


def _load_module():
    global _module
    if _module is not None:
        return _module
    name = "gvr_kernel_ext"
    sources = [str(_CSRC / "binding.cpp"), str(_CSRC / "heuristicTopKDecode.cu")]
    extra_include_paths = [str(_CSRC)]
    extra_cuda_cflags = [
        "-O3", "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        # GVR uses dynamic shared memory > 48 KB; runtime opt-in via
        # cudaFuncSetAttribute, no compile-time flag needed.
    ] + _arch_flags
    extra_cflags = ["-O3", "-std=c++17"]
    print(f"[gvr_kernel] JIT-building extension (first import only). "
          f"arch flags: {_arch_flags}", file=sys.stderr)
    _module = load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=os.environ.get("GVR_KERNEL_VERBOSE", "0") != "0",
    )
    return _module


def gvr_topk_decode(logits: torch.Tensor,
                    preIdx: torch.Tensor,
                    seq_lens: torch.Tensor,
                    K: int = 1024,
                    compress_ratio: int = 4,
                    next_n: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone GVR Heuristic Top-K decode (DSv4 contract).

    Allocates output tensors on every call — convenient for one-off use.
    For bench / perfsim loops at high BS, prefer ``gvr_topk_decode_into``
    with caller-provided output buffers to avoid per-call allocation.

    Args:
        logits:     (BS, Npad)  CUDA fp32/bf16/fp16, contiguous, post-compress
                                index space; Npad is the row stride.
        preIdx:     (BS, K)     CUDA int32, contiguous, prev-step Top-K
                                indices in compressed-index space (V4 caller
                                offset = 0).
        seq_lens:   (BS/next_n,)  CUDA int32, contiguous, total token count
                                = N * compress_ratio + (next_n - 1).
        K:          512 / 1024 / 2048. Default 1024 (V4 Pro native).
        compress_ratio: 1 (V3.2) or 4 (V4 indexer).
        next_n:     Speculative decode draft length. Default 1.

    Returns:
        (indices, values) where
          indices : (BS, K) int32
          values  : (BS, K) same dtype as logits
    """
    mod = _load_module()
    return mod.gvr_topk_decode(logits, preIdx, seq_lens,
                               int(K), int(compress_ratio), int(next_n))


def gvr_topk_decode_into(logits: torch.Tensor,
                         preIdx: torch.Tensor,
                         seq_lens: torch.Tensor,
                         indices_out: torch.Tensor,
                         values_out: torch.Tensor,
                         K: int = 1024,
                         compress_ratio: int = 4,
                         next_n: int = 1) -> None:
    """GVR Heuristic Top-K decode with caller-provided output tensors.

    Avoids the per-call ``torch::empty`` + ``copy_()`` overhead of
    :func:`gvr_topk_decode`. Recommended for perfsim bench loops and for
    cycle-accurate timing — the kernel launch contract is then identical
    to TRT-LLM's in-tree ``torch.ops.trtllm.indexer_topk_decode``.

    Caller must pre-allocate:
        indices_out: (BS, K) int32, contiguous, CUDA
        values_out : (BS, K) same dtype as logits, contiguous, CUDA

    No return value — outputs are written in place.
    """
    mod = _load_module()
    mod.gvr_topk_decode_into(logits, preIdx, seq_lens,
                             indices_out, values_out,
                             int(K), int(compress_ratio), int(next_n))


__all__ = ["gvr_topk_decode", "gvr_topk_decode_into"]
