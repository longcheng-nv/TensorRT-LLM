# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-cell v7 launcher for ncu attribution (L3).
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=<g> ncu \
    --set full --launch-count 1 --launch-skip 3 -f -o results/ncu/<cell> \
    python3 scripts/probe_ncu_run.py <model> <isl> <layer>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP40 = HERE.parent
sys.path.insert(0, str(OP40.parent / "harness"))
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from ab40 import compile_arm, launch_cfg  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
model, isl, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
RD = RV32 if model == "v32" else RV4
b = RD.get_bundle(model, isl, layer, "fp32")
lg, pre, N, K, cr = (b["logits"].contiguous(), b["preIdx"].contiguous(),
                     b["N"], b["K"], b["cr"])
sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
cfg = launch_cfg(lg, N)
fn = compile_arm("v7", K, cr, cfg)
oi = torch.empty(1, K, dtype=torch.int32, device=DEV)
for _ in range(5):
    fn(lg, pre, sl, None, oi, None)
torch.cuda.synchronize()
print("done", flush=True)
