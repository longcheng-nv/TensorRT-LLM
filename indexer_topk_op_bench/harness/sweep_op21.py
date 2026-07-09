# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op#21 (gvr_ms_auto, production single-CTA/cluster dispatch) builder for the
report nsys sweep. Mirrors sweep_op8._build_op8_call exactly: same get_bundle
inputs, logits expand(BS), seq_div = N * cr, pre-allocated out — so op21 cells
are byte-identical in conditions to the other report ops.

The op comes from op21_gvr_prod/src/gvr_msc_op.py at repo state; dispatch
(single-CTA gvr_ms vs C=4 / C=8 cluster gvr_msc) is the op's own production
rule on (dtype, K, N, BS, NUM_SMS). The picked path is recorded as extra
metadata ("ms_path") by replicating the rule read-only.
"""
import sys
from pathlib import Path

import torch

_BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BENCH / "op21_gvr_prod" / "src"))

from gvr_msc_op import NUM_SMS, gvr_ms_auto  # noqa: E402

DEV = "cuda"


def _picked_path(dtype, K, N, BS):
    """Replicate gvr_ms_auto's dispatch (read-only, for metadata)."""
    dt16 = dtype in (torch.bfloat16, torch.float16)
    if dt16 and N >= 65536 and N >= 32768 * BS:
        return "msc_C8"
    if K >= 2048 and N >= 196608 and BS <= 4:
        return "msc_C8"
    # op25 Step-4 fp32 C=8 widening rule (mirrors gvr_ms_auto at HEAD)
    if (not dt16 and K < 2048 and BS <= 8
            and (N >= 131072 or (N >= 65536 and K >= 1024))):
        return "msc_C8"
    if N >= 65536 and BS * 4 <= NUM_SMS:
        return "msc_C4"
    return "ms_1cta"


def _build_op21_call(K, dtype, N, BS, cr, logits_row, preidx_row):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_div, pre, out]
    path = _picked_path(dtype, K, logits.shape[1], BS)
    gvr_ms_auto(logits, pre, seq_div, K, compress_ratio=cr, out=out)  # warm compile
    return (lambda: gvr_ms_auto(logits, pre, seq_div, K, compress_ratio=cr,
                                out=out)), keep, path
