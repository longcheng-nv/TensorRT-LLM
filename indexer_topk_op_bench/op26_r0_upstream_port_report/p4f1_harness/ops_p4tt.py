# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""p4tt tiny-tie 2-arm A/B on PR head @1128c0544f (gvrpkgprod2, K-gated).

  p4tt_off : p4_tail_fast=False == PR head (PTX byte-identical, battery caseA)
  p4tt_on  : p4_tail_fast=True  (tiny-tie collect+select in exact-tail fire)

fp32 & K in {1024, 2048} only — the ship gate turns K512 OFF (byte-identical).
Flags flow through GvrTopKKernel.launch(**kernel_overrides) (cache-key safe).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_HERE))

from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel   # noqa: E402

DEV = "cuda"
RIVAL_OPS = ["p4tt_off", "p4tt_on"]
GVR_ANCHOR = "p4tt_off"


def _ovr(op, K):
    return {"p4_tail_fast": op == "p4tt_on"}


def ops_for_rival(dtype_name, K):
    return list(RIVAL_OPS)


def build_call_rival(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    lg_row = logits_row.to(dtype).contiguous()
    pre_row = preidx_row.contiguous()
    lg = lg_row.expand(BS, -1).contiguous()
    pre = pre_row.expand(BS, -1).contiguous()
    sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ovr = _ovr(op, K)
    cfg = GvrTopKKernel.pick_config(dtype, BS, lg.shape[1])
    call = (lambda lg=lg, pre=pre, sl=sl, out=out, ovr=ovr:
            GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr, **ovr))
    call()
    extra = {"cluster_size": cfg["cluster_size"],
             "launch_cfg": (f"cs{cfg['cluster_size']}/T{cfg['num_threads']}"
                            f"/mb{cfg['min_blocks_per_mp']}"
                            f"/v{256 if cfg['use_256bit_load'] else 128}"),
             "flags": str(ovr)}
    return call, [lg, pre, sl, out], extra, (lambda: out)
