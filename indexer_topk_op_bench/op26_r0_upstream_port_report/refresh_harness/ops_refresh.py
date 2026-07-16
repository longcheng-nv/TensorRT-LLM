# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""PR-contract refresh sweep — 3 GVR arms only.

Replaces the report's original frozen-config kernel instantiation
(cs = N>=65536 ? 4 : 1, T=1024, mbpm=1, 256-bit even on 16-bit) with the
PR's own launch-shape contract: ``GvrTopKKernel.launch`` -> ``pick_config``
(branch HEAD 018251950f snapshot). This is what production picks, so every
pr/base number downstream is measured at the production shapes (incl. cs=8
at BS<=4 & N>=128K, T=512 below 64K, fp32-only 256-bit loads, mbpm 2/3 at
multi-wave BS).

  gvr_base    : launch(..., enable_r0=False)  — retained secant, PR shapes
  gvr_pr      : launch(...)                   — R0 default, PR shapes
  op26_r0auto : op-bench anchor via in-tree build_call (unchanged; the
                cross-run drift reference vs rival_long.csv)
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]            # indexer_topk_op_bench/
_REPORT = _HERE.parent               # op26_r0_upstream_port_report/
_SNAP = _REPORT / "gvrpkg_snapshot"  # == branch HEAD @018251950f (pick_config+launch)
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_SNAP))

from sweep_nsys import build_call as _gvr_build_call     # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel   # noqa: E402

DEV = "cuda"
RIVAL_OPS = ["gvr_base", "gvr_pr", "op26_r0auto"]
GVR_ANCHOR = "op26_r0auto"


def ops_for_rival(dtype_name, K):
    return list(RIVAL_OPS)


def build_call_rival(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    """Return (call, keep, extra, out_getter) — same contract as ops_rival."""
    if op in ("gvr_base", "gvr_pr"):
        lg_row = logits_row.to(dtype).contiguous()          # [1, Np] (padded)
        pre_row = preidx_row.contiguous()                   # [1, K]
        lg = lg_row.expand(BS, -1).contiguous()
        pre = pre_row.expand(BS, -1).contiguous()
        # seq_len = valid indexer length x cr (NOT padded Np): GVR must not
        # scan the pad tail (real-capture tail is garbage).
        sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
        out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
        ovr = {} if op == "gvr_pr" else {"enable_r0": False}
        cfg = GvrTopKKernel.pick_config(dtype, BS, lg.shape[1])
        call = (lambda lg=lg, pre=pre, sl=sl, out=out, ovr=ovr:
                GvrTopKKernel.launch(lg, pre, sl, out, K,
                                     compress_ratio=cr, **ovr))
        call()   # warm: compile + populate out for the pre-timing exact gate
        extra = {"cluster_size": cfg["cluster_size"],
                 "launch_cfg": (f"cs{cfg['cluster_size']}/T{cfg['num_threads']}"
                                f"/mb{cfg['min_blocks_per_mp']}"
                                f"/v{256 if cfg['use_256bit_load'] else 128}")}
        return call, [lg, pre, sl, out], extra, (lambda: out)

    if op == GVR_ANCHOR:
        call, keep, extra = _gvr_build_call(op, K, dtype, N, BS, cr,
                                            logits_row, preidx_row)
        return call, keep, extra, None

    raise ValueError(f"unknown refresh op {op}")
