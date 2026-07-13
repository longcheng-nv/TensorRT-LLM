# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op28 — build_call extension for the two EXTERNAL latest top-K arms:

  sglang_v2          : sglang@main (2026-07-13) DSv4 top-K v2 (vendored
                       ops/sglang_v2, kernels verbatim; plan UNTIMED at build,
                       timed call = one transform launch = 1-2 kernels).
  flashinfer_topk    : flashinfer.top_k public API (installed 0.6.11; topk.py
                       AND the B200 clusters-path kernel byte-identical to
                       main). Returns (values fp32, indices int64).
  flashinfer_topk_i32: flashinfer.topk.topk_clusters_exact minimal contract
                       (indices int32 only) — output contract matches the
                       in-tree ops.

All fp32-only. Old ops delegate to harness/sweep_nsys.build_call.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "harness"))

from sweep_nsys import build_call as _base_build_call   # noqa: E402
from sglang_v2_op import topk_v2, plan as sglv2_plan    # noqa: E402

DEV = "cuda"
EXT_OPS = ["sglang_v2", "flashinfer_topk", "flashinfer_topk_i32"]


def _fi():
    import flashinfer
    from flashinfer.topk import topk_clusters_exact
    return flashinfer, topk_clusters_exact


def build_call_ext(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    """Return (call, keep, extra) — same contract as sweep_nsys.build_call."""
    if op not in EXT_OPS:
        return _base_build_call(op, K, dtype, N, BS, cr, logits_row, preidx_row)
    if dtype != torch.float32:
        raise ValueError(f"{op} is fp32-only in this bench")

    Npad = logits_row.shape[1]
    assert Npad == N, f"external ops assume Npad==N (got Npad={Npad}, N={N})"
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_nod = torch.full((BS,), N, dtype=torch.int32, device=DEV)
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_nod, out]

    if op == "sglang_v2":
        # plan is UNTIMED (production runs it once per step, reused across all
        # indexer layers); timed call = one transform launch.
        md = sglv2_plan(seq_nod)
        torch.cuda.synchronize()
        keep += [md]
        topk_v2(logits, seq_nod, K, out=out, metadata=md, max_seq_len=N)  # warm
        return (lambda: topk_v2(logits, seq_nod, K, out=out, metadata=md,
                                max_seq_len=N)), keep, {}
    if op == "flashinfer_topk":
        fi, _ = _fi()
        fi.top_k(logits, K)   # warm (JIT load + allocator)
        return (lambda: fi.top_k(logits, K)), keep, {}
    if op == "flashinfer_topk_i32":
        _, tce = _fi()
        tce(logits, K, output_values=False, out_dtype=torch.int32)  # warm
        return (lambda: tce(logits, K, output_values=False,
                            out_dtype=torch.int32)), keep, {}
    raise ValueError(op)
