# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op29 — build_call extension for the gvr29_hbe report arm (iter12 ship
state: col_b=False default, guard N>=65536 streaming; full 4-path dispatch =
register/streaming/cluster stock + HBE tier).

Protocol matches the sglang_v2 arm (ops_ext.py): plan UNTIMED (once per step
in production, reused across indexer layers), spill buffer pre-allocated
UNTIMED (persistent workspace in production), timed call = one transform
launch. Other ops delegate to op28's build_call_ext (sglang_v2 + base ops).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "op28_ext_topk"))
sys.path.insert(0, str(HERE.parents[1] / "harness"))

from ops_ext import build_call_ext as _op28_build_call_ext  # noqa: E402
from gvr29_op import gvr29_topk, plan as g29_plan, _spill_buf  # noqa: E402

DEV = "cuda"
GVR29 = "gvr29_hbe"


def build_call_ext29(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    """Return (call, keep, extra) — same contract as sweep_nsys.build_call."""
    if op != GVR29:
        return _op28_build_call_ext(op, K, dtype, N, BS, cr,
                                    logits_row, preidx_row)
    if dtype != torch.float32:
        raise ValueError(f"{op} is fp32-only in this bench")
    Npad = logits_row.shape[1]
    assert Npad == N, f"{op} assumes Npad==N (got Npad={Npad}, N={N})"
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    pre = preidx_row.to(torch.int32).expand(BS, -1).contiguous()
    assert pre.shape == (BS, K)
    seq = torch.full((BS,), N, dtype=torch.int32, device=DEV)
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    md = g29_plan(seq)
    spill = _spill_buf(BS, K, DEV, False)   # col_b=False ship default
    torch.cuda.synchronize()
    keep = [logits, pre, seq, out, md, spill]
    gvr29_topk(logits, seq, K, pre, out=out, metadata=md, max_seq_len=N,
               spill=spill)  # warm (JIT build + dispatch)
    return (lambda: gvr29_topk(logits, seq, K, pre, out=out, metadata=md,
                               max_seq_len=N, spill=spill)), keep, {}
