# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""vsfull3: new-PR-head full-envelope re-measure — 3 GVR arms.

  gvr_base : launch(..., enable_r0=False)     — secant baseline (upstream)
  gvr_pr   : OLD PR head @018251950f semantics (r0_vseed=False,
             qfracs=(0.85,0.35), p4_exact_tail=False)
  gvr_vs   : NEW PR head @eae374554c defaults (per-K vseed + exact tail)

All arms import the MACHINE-LOCAL edited gvrpkg (/tmp/gvrval1) — base/pr flags
default off there, so they compile byte-identical kernels to the NFS snapshot
(verified: round-1/2 pr/base ratios reproduce the 07-16 refresh CSVs).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, "/tmp/gvrval1")   # EDITED gvrpkg (r0_vseed flag) — machine-local

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel   # noqa: E402

DEV = "cuda"
RIVAL_OPS = ["gvr_base", "gvr_pr", "gvr_vs"]
GVR_ANCHOR = "gvr_pr"
# Round-2 verdict (b200-072, 25 cells x 4 arms): per-K hybrid.
#  K512/K1024: pmean REPLACES the q.35 rung (2 columns, zero column tax;
#    q.35's admission region is covered by pmean on all observed cells).
#  K2048: keep (0.85,0.35)+pmean (3 columns). kC/K=3 makes fat admission
#    costly; the q.35 rung keeps the miss-bracket tight (v32-64k: vs2 0.86).
def _vs_ovr(K):
    return {}   # NEW defaults: per-K vseed + p4_exact_tail baked into the ctor


_OVR = {"gvr_base": {"enable_r0": False},
        "gvr_pr": {"r0_vseed": False, "r0_qfracs": (0.85, 0.35),
                   "p4_exact_tail": False}}


def ops_for_rival(dtype_name, K):
    return list(RIVAL_OPS)


def build_call_rival(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    """Same contract as ops_refresh.build_call_rival (GVR arms only)."""
    lg_row = logits_row.to(dtype).contiguous()
    pre_row = preidx_row.contiguous()
    lg = lg_row.expand(BS, -1).contiguous()
    pre = pre_row.expand(BS, -1).contiguous()
    sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ovr = _OVR[op] if op in _OVR else _vs_ovr(K)
    cfg = GvrTopKKernel.pick_config(dtype, BS, lg.shape[1])
    call = (lambda lg=lg, pre=pre, sl=sl, out=out, ovr=ovr:
            GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr, **ovr))
    call()   # warm: compile + populate out for the pre-timing exact gate
    extra = {"cluster_size": cfg["cluster_size"],
             "launch_cfg": (f"cs{cfg['cluster_size']}/T{cfg['num_threads']}"
                            f"/mb{cfg['min_blocks_per_mp']}"
                            f"/v{256 if cfg['use_256bit_load'] else 128}")}
    return call, [lg, pre, sl, out], extra, (lambda: out)
