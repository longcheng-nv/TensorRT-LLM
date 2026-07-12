# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""op26 arm builders for the nsys sweep harness.

op26_1cta — classic single-CTA GVR + op13 log/narrow P2 dispatch + corrected
            fallback (fb_fix) + op#7 exact rank-scatter P4 (dispatch-gated).
op26_mc   — PR#15198 cluster GVR + log-count P2 interpolation (fp32).

Input materialization is byte-identical to sweep.py:_build_inputs /
sweep_cluster.py:_build_cluster_call so paired A/B deltas are algorithmic.
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "op26_gvr_logfalsi_rs" / "src"))
from gvr_op26_op import (  # noqa: E402
    gvr_cutedsl_op26, gvr_multicta_op26, picked_cluster_size_op26,
)
from gvr_op26_r0_op import gvr_r0_op26  # noqa: E402
from gvr_op26_r0mc_op import gvr_r0_mc_op26, picked_cluster_size_r0mc  # noqa: E402

DEV = "cuda"


def _build_op26_1cta_call(K, dtype, N, BS, cr, logits_row, preidx_row):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_div, pre, out]
    gvr_cutedsl_op26(logits, pre, seq_div, K, compress_ratio=cr, out=out)
    return (lambda: gvr_cutedsl_op26(logits, pre, seq_div, K,
                                     compress_ratio=cr, out=out)), keep


def _build_op26_r0_call(K, dtype, N, BS, cr, logits_row, preidx_row):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_div, pre, out]
    gvr_r0_op26(logits, pre, seq_div, K, compress_ratio=cr, out=out)
    return (lambda: gvr_r0_op26(logits, pre, seq_div, K,
                                compress_ratio=cr, out=out)), keep


def _build_op26_r0mc_call(K, dtype, N, BS, cr, logits_row, preidx_row):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_div, pre, out]
    cs = picked_cluster_size_r0mc(logits, K, cr)
    gvr_r0_mc_op26(logits, pre, seq_div, K, compress_ratio=cr, out=out)
    return (lambda: gvr_r0_mc_op26(logits, pre, seq_div, K,
                                   compress_ratio=cr, out=out)), keep, cs


def _build_op26_mc_call(K, dtype, N, BS, cr, logits_row, preidx_row):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_div, pre, out]
    cs = picked_cluster_size_op26(logits, K, cr)
    gvr_multicta_op26(logits, pre, seq_div, K, compress_ratio=cr, out=out)
    return (lambda: gvr_multicta_op26(logits, pre, seq_div, K,
                                      compress_ratio=cr, out=out)), keep, cs
