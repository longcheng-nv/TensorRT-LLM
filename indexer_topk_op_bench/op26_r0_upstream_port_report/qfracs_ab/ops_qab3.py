# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""bundle-v2 4-arm A/B on TOP of (shipped PR head + K2048 rung swap 0d6fc4f1f2).

  gvr_ship : new baseline — PR head incl. K2048 (0.6, 0.35); flags off
  gvr_h1   : + skip_h1 (drop the end-of-P2 cluster handoff; cs>1 cells only —
             cs=1 cells compile byte-identical, free noise control)
  gvr_kb   : + k_num_bins=512 at K2048 ONLY (P4 hist 2048->512; non-K2048
             cells byte-identical, free noise control)
  gvr_full : + both (the bundle-v2 ship candidate)

Kernel = /tmp/gvrqab gvrpkg with the two bench flags added (canonical patched
copy checkpointed at qfracs_ab/bundlev2/gvr_topk_decode_bundle.py). Flags flow
through GvrTopKKernel.launch(**kernel_overrides) -> ctor (cache-key safe).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, "/tmp/gvrqab")

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel   # noqa: E402

DEV = "cuda"
RIVAL_OPS = ["gvr_ship", "gvr_h1", "gvr_kb", "gvr_full"]
GVR_ANCHOR = "gvr_ship"


def _ovr(op, K):
    o = {}
    if op in ("gvr_h1", "gvr_full"):
        o["skip_h1"] = True
    if op in ("gvr_kb", "gvr_full") and K == 2048:
        o["k_num_bins"] = 512
    return o


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
