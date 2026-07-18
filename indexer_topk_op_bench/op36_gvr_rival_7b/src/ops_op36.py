# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op36 arm registry — rival harness + campaign variants, ALL GVR arms at the
PR production launch contract (GvrTopKKernel.launch -> pick_config), matching
the report §8 2026-07-16 refresh rows (the frozen-shape ops_rival build is
med 1.14 / p95 1.93 SLOWER at big BS — measured iter1, discarded).

Arms:
  gvr_pr / gvr_base : delegate to refresh_harness ops_refresh (launch contract)
  gvr_a0            : Track A0 = gvrpkg35 (PR head eae374554c + default-off
                      flags) at launch contract, skip_h1=True +
                      kNumBins_override=512 for K2048 only (op35 bundle-v2)
  sglang_v2 / radix_cutedsl / flashinfer_topk / op26_r0auto : ops_rival verbatim
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]                       # indexer_topk_op_bench/
_REPORT = _BENCH / "op26_r0_upstream_port_report"
_OP35 = _BENCH / "op35_gvr_round2" / "variant"
for p in (_REPORT / "rival_harness", _REPORT / "refresh_harness", _OP35):
    sys.path.insert(0, str(p))

import ops_rival as OR                                        # noqa: E402
import ops_refresh as ORF                                     # noqa: E402
from gvrpkg35.top_k.gvr_topk_decode import GvrTopKKernel as Gvr35  # noqa: E402

DEV = "cuda"
OP36_ARMS = ["gvr_a0"]


def ops_for_op36(dtype_name, K):
    return OR.ops_for_rival(dtype_name, K) + OP36_ARMS


def _build_a0(K, dtype, N, BS, cr, logits_row, preidx_row):
    """gvrpkg35 at the production launch contract + bundle-v2 overrides."""
    lg = logits_row.to(dtype).contiguous().expand(BS, -1).contiguous()
    pre = preidx_row.contiguous().expand(BS, -1).contiguous()
    sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ovr = dict(skip_h1=True)
    if K == 2048:
        ovr["kNumBins_override"] = 512
    call = (lambda lg=lg, pre=pre, sl=sl, out=out, ovr=ovr:
            Gvr35.launch(lg, pre, sl, out, K, compress_ratio=cr, **ovr))
    call()                                     # warm: compile + populate out
    cfg = Gvr35.pick_config(dtype, BS, lg.shape[1])
    extra = {"cluster_size": cfg["cluster_size"],
             "launch_cfg": (f"cs{cfg['cluster_size']}/T{cfg['num_threads']}"
                            f"/mb{cfg['min_blocks_per_mp']}"
                            f"/v{256 if cfg['use_256bit_load'] else 128}"),
             "flags": "skip_h1" + ("+kb512" if K == 2048 else "")}
    return call, [lg, pre, sl, out], extra, (lambda: out)


def build_call_op36(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    if op == "gvr_a0":
        return _build_a0(K, dtype, N, BS, cr, logits_row, preidx_row)
    if op in ("gvr_base", "gvr_pr"):
        return ORF.build_call_rival(op, K, dtype, N, BS, cr,
                                    logits_row, preidx_row)
    return OR.build_call_rival(op, K, dtype, N, BS, cr, logits_row, preidx_row)
