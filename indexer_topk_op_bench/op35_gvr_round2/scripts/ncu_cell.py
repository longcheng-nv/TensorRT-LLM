# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Launch one cell's kernel a few times for NCU attribution.
Usage: ncu_cell.py <cell_id> [reps]"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from ab_op35 import iter_cells, load_cell                       # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel as BaseK  # noqa: E402

cid_want = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
for cell in iter_cells("all"):
    cid, lg_row, pre, K, cr, N = load_cell(cell)
    if cid != cid_want:
        continue
    lg = lg_row.unsqueeze(0).contiguous().cuda()
    pre = pre[:1].contiguous().cuda()
    sl = torch.full((1,), N * cr, dtype=torch.int32, device="cuda")
    out = torch.empty(1, K, dtype=torch.int32, device="cuda")
    for _ in range(reps + 1):
        BaseK.launch(lg, pre, sl, out, K, compress_ratio=cr)
    torch.cuda.synchronize()
    print("done", cid)
    break
