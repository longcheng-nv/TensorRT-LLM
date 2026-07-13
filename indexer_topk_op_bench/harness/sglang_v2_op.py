# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builder + Python wrapper for the standalone LATEST SGLang v2 top-K (op #28).

Builds ops/sglang_v2/topk_v2_standalone.cu — the sglang@main (2026-07-13)
DeepSeek-V4 top-K v2 architecture: topk_impl.cuh device classes (TopKRegister /
TopKStreaming / TopKCluster<8>) + the 4 upstream kernels verbatim; only the
tvm-ffi host layer is replaced (see the .cu header).

Contract mirrors sglang_streaming_op.streaming_topk:
    topk_v2(scores[R,C] fp32, seq_lens[R] int32, K) -> out[R,K] int32
(global top-K indices via identity page table; unordered; runtime K <= 2048).

plan runs UNTIMED at wrapper level by default (production runs plan once per
step and reuses it across the ~61 indexer layers); `timed_plan=True` folds it
into the returned closure for sensitivity checks.
"""
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_OPDIR = _HERE.parent / "ops" / "sglang_v2"
_BUILD = _HERE.parent / "_build"
_BUILD.mkdir(exist_ok=True)

os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

# page_bits large enough that page_to_indices(i) == i for every column index
# i < C (max C = 1048576 = 2^20). 21 bits covers it with slack.
PAGE_BITS = 21

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        bdir = _BUILD / "sglang_v2"
        bdir.mkdir(parents=True, exist_ok=True)
        _MOD = load(
            name="sglang_topk_v2",
            sources=[str(_OPDIR / "topk_v2_standalone.cu")],
            extra_include_paths=[str(_OPDIR), str(_OPDIR / "shim")],
            extra_cuda_cflags=[
                "-O3", "-std=c++20",
                "-gencode=arch=compute_100f,code=sm_100f",
                "-DSGL_CUDA_ARCH=1000",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O3", "-std=c++20", "-DSGL_CUDA_ARCH=1000"],
            build_directory=str(bdir),
            verbose=True,
        )
    return _MOD


_PAGE_TABLE = None


def _identity_page_table(device):
    global _PAGE_TABLE
    if _PAGE_TABLE is None or _PAGE_TABLE.device != torch.device(device):
        _PAGE_TABLE = torch.zeros(1, dtype=torch.int32, device=device)
    return _PAGE_TABLE


def plan(seq_lens, metadata=None, static_cluster_threshold=0):
    """Run the upstream topk_plan kernel. metadata: [R+1, 2] int32."""
    R = seq_lens.size(0)
    if metadata is None:
        metadata = torch.zeros((R + 1, 2), dtype=torch.int32,
                               device=seq_lens.device)
    _module().topk_v2_plan(seq_lens, metadata, static_cluster_threshold)
    return metadata


def topk_v2(scores, seq_lens, K, out=None, metadata=None, page_table=None,
            max_seq_len=None):
    """One transform launch (1 or 2 kernels per upstream dispatch). Assumes
    `metadata` was produced by plan() for these seq_lens (required only when
    the persistent-cluster path can trigger; harmless otherwise)."""
    assert scores.dtype == torch.float32, "sglang v2 top-K is fp32-only"
    R = scores.size(0)
    if out is None:
        out = torch.empty((R, K), dtype=torch.int32, device=scores.device)
    if page_table is None:
        page_table = _identity_page_table(scores.device)
    if metadata is None:
        metadata = plan(seq_lens)
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item())
    _module().topk_v2_transform(scores, seq_lens, page_table, out, metadata,
                                K, PAGE_BITS, max_seq_len)
    return out


if __name__ == "__main__":
    # smoke build + correctness vs torch.topk (value-multiset compare — output
    # order is unordered and tie choice may differ from torch)
    torch.manual_seed(0)
    _module()
    print("sglang_topk_v2 built OK")
    for K in (512, 1024, 2048):
        for N in (4096, 8192, 16384, 65536, 131072):
            for BS in (1, 4, 31, 64):   # 31/64 with N>64K exercise the 2-kernel path
                x = torch.randn(BS, N, dtype=torch.float32, device="cuda")
                sl = torch.full((BS,), N, dtype=torch.int32, device="cuda")
                md = plan(sl)
                idx = topk_v2(x, sl, K, metadata=md, max_seq_len=N)
                ref = torch.topk(x, K, dim=1)
                ok = True
                for r in range(BS):
                    got = x[r][idx[r].long()].sort(descending=True).values
                    if not torch.equal(got, ref.values[r]):
                        ok = False
                        break
                tag = "2KERN" if (N > 65536 and 30 < BS <= 512) or \
                    (N > 32768 and BS <= 15 and False) else "     "
                print(f"  K={K} N={N} BS={BS} exact={ok} {tag}")
                assert ok, f"MISMATCH K={K} N={N} BS={BS}"
    print("ALL EXACT")
