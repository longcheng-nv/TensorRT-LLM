# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""HEAD-full-coverage sweep ops — PR head @e6fdbfac3d (incl. p4tt, shipped
defaults) vs base, + op26 anchor.

Same module name/contract as refresh_harness.ops_refresh so sweep_refresh.py
and the driver work unchanged. Arms:

  gvr_base    : head pkg, launch(..., enable_r0=False) — retained secant
  gvr_pr      : head pkg, launch(...) — R0 + vseed + exact_tail + p4tt
                K-gate defaults == what production would run at this head
  op26_r0auto : op-bench anchor (unchanged; cross-run drift ref)

Head package = p4f1_harness/gvrpkgprod2 (md5-matched to the branch commit
@e6fdbfac3d; its ctor defaults ARE the shipped defaults, no overrides).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
_REPORT = _HERE.parent
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_REPORT / "p4f1_harness"))

from sweep_nsys import build_call as _gvr_build_call                 # noqa: E402
from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel          # noqa: E402

DEV = "cuda"
RIVAL_OPS = ["gvr_base", "gvr_pr", "op26_r0auto"]
GVR_ANCHOR = "op26_r0auto"


def ops_for_rival(dtype_name, K):
    return list(RIVAL_OPS)


def build_call_rival(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    if op in ("gvr_base", "gvr_pr"):
        lg_row = logits_row.to(dtype).contiguous()
        pre_row = preidx_row.contiguous()
        lg = lg_row.expand(BS, -1).contiguous()
        pre = pre_row.expand(BS, -1).contiguous()
        sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
        out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
        ovr = {} if op == "gvr_pr" else {"enable_r0": False}
        cfg = GvrTopKKernel.pick_config(dtype, BS, lg.shape[1])
        call = (lambda lg=lg, pre=pre, sl=sl, out=out, ovr=ovr:
                GvrTopKKernel.launch(lg, pre, sl, out, K,
                                     compress_ratio=cr, **ovr))
        call()
        extra = {"cluster_size": cfg["cluster_size"],
                 "launch_cfg": (f"cs{cfg['cluster_size']}/T{cfg['num_threads']}"
                                f"/mb{cfg['min_blocks_per_mp']}"
                                f"/v{256 if cfg['use_256bit_load'] else 128}")}
        return call, [lg, pre, sl, out], extra, (lambda: out)

    if op == GVR_ANCHOR:
        call, keep, extra = _gvr_build_call(op, K, dtype, N, BS, cr,
                                            logits_row, preidx_row)
        return call, keep, extra, None

    raise ValueError(f"unknown headfull op {op}")
