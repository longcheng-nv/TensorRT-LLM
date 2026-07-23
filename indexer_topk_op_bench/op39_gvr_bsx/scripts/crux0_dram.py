# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 rung-0 crux: one launch each of candidate v3 and PR head on a big cell,
run under ncu with dram bytes metrics. Info floor = BS*N*4 bytes read.

  ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ... python3 crux0_dram.py <model> <isl> <L> <BS>
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "op26_r0_upstream_port_report" / "kf_campaign" / "gvrpkg_04a0"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import build, bundle, make_batch  # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402

model, isl, L, bs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
b = bundle(model, isl, L)
K, N, cr = b["K"], b["N"], b["cr"]
lg, pre = make_batch(b, bs)
mod = build("kernel_bs")
out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
# warm both once outside any profiling relevance (ncu profiles every launch;
# we take the LAST instance of each kernel name)
mod.run(lg, pre, N, out)
torch.cuda.synchronize()
mod.run(lg, pre, N, out)  # candidate: profile this
torch.cuda.synchronize()
sl = torch.full((bs,), N * cr, dtype=torch.int32, device="cuda")
out_p = torch.empty(bs, K, dtype=torch.int32, device="cuda")
GvrTopKKernel.launch(lg, pre, sl, out_p, K, compress_ratio=cr)
torch.cuda.synchronize()
GvrTopKKernel.launch(lg, pre, sl, out_p, K, compress_ratio=cr)  # pr: profile this
torch.cuda.synchronize()
print(f"[crux0] floor bytes = {bs*N*4/1e9:.3f} GB (BS{bs} x N{N} x 4)")
