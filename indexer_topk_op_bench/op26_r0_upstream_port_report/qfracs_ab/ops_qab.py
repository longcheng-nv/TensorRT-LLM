# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""qfracs A/B — 3 GVR arms on the SHIPPED PR head (eae374554c), lever ① of the
§9c rung-recalibration study: swap the real-content-calibrated GLOBAL rung
quantiles in via ``launch(**kernel_overrides)`` (cache-key participant, no
source edits).

  gvr_ship : shipped defaults — K512/K1024 (0.85,)+vseed, K2048 (0.85,0.35)+vseed
  gvr_qr2  : real-calibrated pair per K (§5c-CCDF-b Gbest_real):
               K512 (0.9, 0.5) · K1024 (0.95, 0.6) · K2048 (0.6, 0.35)
             NOTE for K512/K1024 this ADDS one explicit count column vs ship
             (2 rungs + vseed) — the column-tax risk arm.
  gvr_qr1  : column-count-preserving variant — single explicit rung moved to the
             real-calibrated low rung: K512 (0.9,) · K1024 (0.95,) ·
             K2048 (0.6, 0.35). For K2048, qr1 == qr2 (free noise control).

All arms import the MACHINE-LOCAL gvrpkg (/tmp/gvrqab) whose gvr_topk_decode.py
is the verbatim shipped head. vseed / p4_exact_tail stay at shipped defaults in
every arm; the ONLY differing knob is r0_qfracs.
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, "/tmp/gvrqab")   # machine-local gvrpkg @ shipped head

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel   # noqa: E402

DEV = "cuda"
RIVAL_OPS = ["gvr_ship", "gvr_qr2", "gvr_qr1"]
GVR_ANCHOR = "gvr_ship"

QR2 = {512: (0.9, 0.5), 1024: (0.95, 0.6), 2048: (0.6, 0.35)}
QR1 = {512: (0.9,),     1024: (0.95,),     2048: (0.6, 0.35)}


def _ovr(op, K):
    if op == "gvr_ship":
        return {}
    return {"r0_qfracs": (QR2 if op == "gvr_qr2" else QR1)[K]}


def ops_for_rival(dtype_name, K):
    return list(RIVAL_OPS)


def build_call_rival(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    """Same contract as ops_vs4.build_call_rival (GVR arms only)."""
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
    call()   # warm: compile + populate out for the pre-timing exact gate
    extra = {"cluster_size": cfg["cluster_size"],
             "launch_cfg": (f"cs{cfg['cluster_size']}/T{cfg['num_threads']}"
                            f"/mb{cfg['min_blocks_per_mp']}"
                            f"/v{256 if cfg['use_256bit_load'] else 128}"),
             "qfracs": str(_ovr(op, K).get("r0_qfracs", "ship"))}
    return call, [lg, pre, sl, out], extra, (lambda: out)
