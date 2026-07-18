# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""[op36 Track B] Builder + wrapper for the EXACTNESS PORT of sglang v2 top-K.

Same contract as harness/sglang_v2_op.py (plan untimed / one transform launch /
identity page table / fp32-only), plus the unconditional-exactness moat:

  - the kernels set a per-row overflow flag whenever the kMaxNumTie=2048 tie
    collect truncated while tie slots were still needed (all 4 sites, see
    topk_impl_exact.cuh). Flags are zeroed inside the UNTIMED plan kernel.
  - `escape_rerun(...)` re-runs flagged rows through the exact in-tree radix
    path (radix_cutedsl, unconditionally exact 2245/2245). Data-rare; a
    correctness escape hatch, not a perf path — the TIMED call is transform
    alone, whose only delta vs the vendored sglang_v2 arm is the flag branch.

Production-integration note: the flag check needs the row output anyway
(consumed by sparse attention on-device), so a real integration would either
fold the re-run into a dependent kernel or accept a rare host round-trip;
here the bench times the kernel span (nsys us_span) and gates exactness
offline, which measures exactly the ship-relevant ε (the in-kernel guard).
"""
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[2]                       # indexer_topk_op_bench/
_OPDIR = _BENCH / "ops" / "sglang_v2"           # vendored sgl_kernel/* headers
_BUILD = _BENCH / "_build"
_BUILD.mkdir(exist_ok=True)

os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

PAGE_BITS = 21  # identity page table for every column index < 2^21

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        bdir = _BUILD / "sgl_bx"
        bdir.mkdir(parents=True, exist_ok=True)
        _MOD = load(
            name="sglang_topk_v2_exact",
            sources=[str(_HERE / "topk_v2_exact_standalone.cu")],
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


def plan(seq_lens, metadata=None, static_cluster_threshold=0, flags=None):
    """Upstream topk_plan + [op36-TB] flag zeroing. Returns (metadata, flags)."""
    R = seq_lens.size(0)
    dev = seq_lens.device
    if metadata is None:
        metadata = torch.zeros((R + 1, 2), dtype=torch.int32, device=dev)
    if flags is None:
        flags = torch.zeros(R, dtype=torch.uint8, device=dev)
    _module().topk_v2_plan(seq_lens, metadata, static_cluster_threshold, flags)
    return metadata, flags


def topk_bx(scores, seq_lens, K, out=None, metadata=None, flags=None,
            page_table=None, max_seq_len=None):
    """One transform launch (timed call). Flags must come from plan()."""
    assert scores.dtype == torch.float32, "sgl_bx top-K is fp32-only"
    R = scores.size(0)
    if out is None:
        out = torch.empty((R, K), dtype=torch.int32, device=scores.device)
    if page_table is None:
        page_table = _identity_page_table(scores.device)
    if metadata is None or flags is None:
        metadata, flags = plan(seq_lens, metadata=metadata, flags=flags)
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item())
    _module().topk_v2_transform(scores, seq_lens, page_table, out, metadata,
                                K, PAGE_BITS, max_seq_len, flags)
    return out


def escape_rerun(scores, seq_lens, K, out, flags):
    """[op36-TB] host escape: re-run flagged rows through the exact in-tree
    radix path. Returns the number of rows re-run (0 in the common case; the
    flags.any() below is the only host sync)."""
    if not bool(flags.any().item()):
        return 0
    import sys
    sys.path.insert(0, str(_BENCH / "harness"))
    from radix_cutedsl_op import radix_cutedsl
    rows = flags.nonzero(as_tuple=False).flatten()
    sub_scores = scores[rows].contiguous()
    sub_sl = seq_lens[rows].contiguous()
    sub_out = torch.empty((rows.numel(), K), dtype=torch.int32,
                          device=scores.device)
    radix_cutedsl(sub_scores, sub_sl, K, out=sub_out)
    out[rows] = sub_out
    return int(rows.numel())


def topk_bx_exact(scores, seq_lens, K, out=None, metadata=None, flags=None,
                  page_table=None, max_seq_len=None):
    """Full exact pipeline (battery/gate use): transform + escape."""
    if flags is None or metadata is None:
        metadata, flags = plan(seq_lens, metadata=metadata, flags=flags)
    out = topk_bx(scores, seq_lens, K, out=out, metadata=metadata, flags=flags,
                  page_table=page_table, max_seq_len=max_seq_len)
    n = escape_rerun(scores, seq_lens, K, out, flags)
    return out, n
