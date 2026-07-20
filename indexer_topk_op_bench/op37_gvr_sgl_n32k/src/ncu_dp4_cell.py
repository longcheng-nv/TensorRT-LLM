#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op37 — single-cell runner for ncu A/B of gvr_pr vs gvr_dp4.

Usage: ARM=pr|dp4 MODEL=v32 ISL=64k BS=8 python3 ncu_dp4_cell.py
Launches the arm 12 times (after 3 warmups) so ncu can profile replays.
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP37 = HERE.parent
OPBENCH = OP37.parent
sys.path.insert(0, str(OP37 / "variant"))
sys.path.insert(0, str(OPBENCH / "harness"))
sys.path.insert(0, str(OPBENCH / "op26_r0_upstream_port_report" / "p4f1_harness"))

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402

ARM = os.environ.get("ARM", "pr")
MODEL = os.environ.get("MODEL", "v32")
ISL = os.environ.get("ISL", "64k")
BS = int(os.environ.get("BS", "8"))
LAYER = {"v32": 34, "flash": 22, "pro": 30}[MODEL]

RD = RV32 if MODEL == "v32" else RV4
b = RD.get_bundle(MODEL, ISL, LAYER, "fp32")
N, K, cr = b["N"], b["K"], b["cr"]
lg = b["logits"].to(torch.float32).contiguous().expand(BS, -1).contiguous()
pre = b["preIdx"].contiguous().expand(BS, -1).contiguous()
sl = torch.full((BS,), N * cr, dtype=torch.int32, device="cuda")
out = torch.empty(BS, K, dtype=torch.int32, device="cuda")

if ARM == "sgl":
    sys.path.insert(0, str(OPBENCH / "op26_r0_upstream_port_report" / "rival_harness"))
    import ops_rival as OR
    call, keep, extra, getter = OR.build_call_rival(
        "sglang_v2", K, torch.float32, N, BS, cr, lg[:1], pre[:1])
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(12):
        call()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("DONE sgl", flush=True)
    sys.exit(0)
if ARM == "dp4":
    from gvrpkg37.top_k.gvr_topk_decode import GvrTopKKernel as Gvr
    ovr = dict(dist_p4=True)
else:
    from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as Gvr
    ovr = {}
cfg = Gvr.pick_config(torch.float32, BS, lg.shape[1])
if ARM == "dp4":
    assert cfg["cluster_size"] > 1, cfg
print(f"[{ARM}] {MODEL}/{ISL} N={N} BS={BS} cs={cfg['cluster_size']}", flush=True)

for _ in range(3):
    Gvr.launch(lg, pre, sl, out, K, compress_ratio=cr, **ovr)
torch.cuda.synchronize()
torch.cuda.profiler.start()
for _ in range(12):
    Gvr.launch(lg, pre, sl, out, K, compress_ratio=cr, **ovr)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("DONE", flush=True)
