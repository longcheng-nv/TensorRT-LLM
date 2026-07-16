# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""iter10b: phase attribution via mode={1:A,2:A+B,3:full} event timing (relative
only; L1 axis) + dbg readout (M, bad flags)."""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../src"))
from apex_op import apex_topk, pick_config, workspace  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--reps", type=int, default=50)
a = ap.parse_args()
torch.cuda.set_device(a.gpu)
EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

CASES = [(1, 131072, 512), (1, 262144, 512), (1, 1048576, 512),
         (32, 262144, 512), (256, 262144, 512), (1024, 65536, 512)]
print(f"{'cell':>18} {'A':>8} {'A+B':>8} {'full':>8} {'M':>6} {'badflags':>8}")
for BS, N, K in CASES:
    torch.manual_seed(N + BS)
    x = torch.rand(BS, N, device="cuda") + 1.0
    cfg = pick_config(BS, N, K)
    ws = workspace(BS, K, cfg, x.device)
    dbg = torch.zeros(BS * 8, dtype=torch.int32, device="cuda")
    res = {}
    for mode in (1, 2, 3):
        apex_topk(x, K, cfg=cfg, ws=ws, mode=mode, dbg=dbg)
        torch.cuda.synchronize()
        # reset counts contamination from mode<3 probes
        ws["counts"].zero_(); ws["tickets"].zero_()
        ts = []
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(a.reps):
            EVICT.uniform_(); torch.cuda.synchronize()
            ev0.record()
            apex_topk(x, K, cfg=cfg, ws=ws, mode=mode, dbg=dbg)
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1) * 1e3)
            if mode != 3:
                ws["counts"].zero_(); ws["tickets"].zero_()
        ts.sort()
        res[mode] = ts[len(ts) // 2]
    d = dbg.view(BS, 8)
    print(f"BS{BS:<6} N{N:<9} {res[1]:8.2f} {res[2]:8.2f} {res[3]:8.2f} "
          f"{int(d[:,0].max()):>6} {int(d[:,1].max()):>8}  "
          f"tail(us) gather={d[:,4].float().mean()/1e3:.2f} "
          f"radix={d[:,5].float().mean()/1e3:.2f} "
          f"emit={d[:,6].float().mean()/1e3:.2f}")
