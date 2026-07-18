# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op36 arm registry — extends the op26-report rival harness with campaign
variants. Timing/exactness protocol stays VERBATIM sweep_rival (we only add
build_call branches), so op36 numbers sit on the same axis as the §8 grid.

Arms added on top of RIVAL_OPS:
  gvr_a0 : Track A0 = PR shipped head + op35 bundle-v2 (skip_h1=True +
           kNumBins_override=512 for K2048 only). Kernel class from
           op35_gvr_round2/variant/gvrpkg35 (PR head + default-off flags).
"""
import sys
from pathlib import Path

import torch
import cutlass

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]                       # indexer_topk_op_bench/
_RH = _BENCH / "op26_r0_upstream_port_report" / "rival_harness"
_OP35 = _BENCH / "op35_gvr_round2" / "variant"
for p in (_RH, str(_OP35)):
    sys.path.insert(0, str(p))

import ops_rival as OR                                        # noqa: E402
from ops_rival import ops_for_rival as _ops_base              # noqa: E402
from gvrpkg35.top_k.gvr_topk_decode import GvrTopKKernel as Gvr35  # noqa: E402
from exact import compile_kernel                              # noqa: E402

DEV = "cuda"
_CDT = {torch.float32: cutlass.Float32, torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16}

OP36_ARMS = ["gvr_a0"]


def ops_for_op36(dtype_name, K):
    return _ops_base(dtype_name, K) + OP36_ARMS


def _build_gvr35(K, dtype, N, BS, cr, logits_row, preidx_row, **kflags):
    """Same launch contract as ops_rival gvr_base/gvr_pr (padded row, expand to
    BS, seq_len = N*cr, cs = 1 if N<65536 else 4) with the gvrpkg35 class."""
    cs = 1 if N < 65536 else 4
    lg = logits_row.to(dtype).contiguous().expand(BS, -1).contiguous()
    pre = preidx_row.contiguous().expand(BS, -1).contiguous()
    sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    k = Gvr35(dtype=_CDT[dtype], top_k=K, next_n=1, num_threads=1024,
              compress_ratio=cr, use_256bit_load=True, min_blocks_per_mp=1,
              cluster_size=cs, return_output_values=False, enable_r0=True,
              **kflags)
    f = compile_kernel(k, True)
    o = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    call = lambda: f(lg, pre, sl, None, o, None)
    call()
    return call, [lg, pre, sl, o], {"cluster_size": cs, **{k_: str(v) for k_, v in kflags.items()}}, (lambda: o)


def build_call_op36(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    if op == "gvr_a0":
        flags = dict(skip_h1=True)
        if K == 2048:
            flags["kNumBins_override"] = 512
        return _build_gvr35(K, dtype, N, BS, cr, logits_row, preidx_row, **flags)
    return OR.build_call_rival(op, K, dtype, N, BS, cr, logits_row, preidx_row)
