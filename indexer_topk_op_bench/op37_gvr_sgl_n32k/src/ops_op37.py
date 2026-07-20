#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op37 arm registry — GVR-vs-sglang on real §7b, N>=32K, PURE-GVR campaign.

Arms:
  gvr_pr    : PR#16457 CURRENT head @e6fdbfac3d (p4f1_harness/gvrpkgprod2,
              md5 3396037cfc6eb7afe8924f2385ebd874) via headfull ops_refresh
              (production launch()/pick_config contract).
  sglang_v2 : rival_harness ops_rival arm (fp32-only, span-timed rival).
  gvr_a2    : op36 gvrpkg36 (eae374554c base) + dist_p4 where cs>1 — kept for
              lever attribution only (base is 3 commits older than gvr_pr).
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
_REPORT = _BENCH / "op26_r0_upstream_port_report"
for p in (_REPORT / "rival_harness", _REPORT / "headfull_harness",
          _BENCH / "op36_gvr_rival_7b" / "src"):
    sys.path.insert(0, str(p))

import ops_rival as OR                                        # noqa: E402
import ops_refresh as ORF_HEAD                                # noqa: E402  (headfull -> gvrpkgprod2)

OP37_DEFAULT = ["gvr_pr", "sglang_v2"]


PROBE_ARMS = ["gvr_cs2", "gvr_cs4", "gvr_cs8", "gvr_a2", "gvr_dp4"]


def ops_for_op37(dtype_name, K):
    ops = list(OP37_DEFAULT) + PROBE_ARMS
    if dtype_name != "fp32":
        ops = [o for o in ops if o != "sglang_v2"]
    return ops


def _build_pr_cs(cs_override, K, dtype, N, BS, cr, logits_row, preidx_row):
    """gvr_pr at current head with a forced cluster_size (L1 probe: does
    clustering pay at N=32771 / small BS where pick_config says cs=1?)."""
    import torch
    Gvr = ORF_HEAD.GvrTopKKernel
    lg = logits_row.to(dtype).contiguous().expand(BS, -1).contiguous()
    pre = preidx_row.contiguous().expand(BS, -1).contiguous()
    sl = torch.full((BS,), N * cr, dtype=torch.int32, device="cuda")
    out = torch.empty(BS, K, dtype=torch.int32, device="cuda")
    ovr = dict(cluster_size=cs_override)
    call = (lambda lg=lg, pre=pre, sl=sl, out=out, ovr=ovr:
            Gvr.launch(lg, pre, sl, out, K, compress_ratio=cr, **ovr))
    call()
    return call, [lg, pre, sl, out], {"cluster_size": cs_override,
                                      "launch_cfg": f"cs{cs_override}/forced"}, (lambda: out)


def build_call_op37(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    if op.startswith("gvr_cs"):
        return _build_pr_cs(int(op[6:]), K, dtype, N, BS, cr,
                            logits_row, preidx_row)
    if op in ("gvr_base", "gvr_pr", "op26_r0auto"):
        return ORF_HEAD.build_call_rival(op, K, dtype, N, BS, cr,
                                         logits_row, preidx_row)
    if op == "gvr_a2":
        import ops_op36
        return ops_op36._build_a2(K, dtype, N, BS, cr, logits_row, preidx_row)
    return OR.build_call_rival(op, K, dtype, N, BS, cr, logits_row, preidx_row)
