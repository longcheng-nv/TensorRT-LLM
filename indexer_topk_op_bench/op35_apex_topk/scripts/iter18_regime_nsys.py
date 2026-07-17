# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""iter18: pipelined vs sequential split on the target regime cells (nsys).
Cells: BS {64,128,256,512,1024} x N {131072, 262144} x K x scen (op26 synth).
NVTX c| ranges; parse with nvtx_gpu_proj (spans cover all 3 streams)."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))
from sweep_nsys import measure_cell        # noqa: E402
import bundle_data_env as SYNTH            # noqa: E402
from apex_op import apex_topk, pick_config, workspace  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--chunks", type=int, default=4)
a = ap.parse_args()
torch.cuda.set_device(a.gpu)

CELLS = [(scen, K, N, BS)
         for scen in ("best", "worst")
         for K in (512, 1024, 2048)
         for N in (131072, 262144)
         for BS in (64, 128, 256, 512, 1024)]

recs = []
prof.start()
for scen, K, N, BS in CELLS:
    b = SYNTH.get_bundle(scen, K, torch.float32, N)
    W = ((N + 63) // 64) * 64
    x = torch.full((BS, W), torch.finfo(torch.float32).min, device="cuda")
    x[:, :N] = b["logits"][0, :N].float()
    for arm in ("seq", "pipe"):
        cfg = pick_config(BS, N, K)
        cfg["pipeline"] = (arm == "pipe")
        cfg["chunks"] = a.chunks
        ws = workspace(BS, K, cfg, x.device)
        call = lambda: apex_topk(x, K, N=N, cfg=cfg, ws=ws)  # noqa: E731
        call(); torch.cuda.synchronize()
        base = f"i18|{arm}|{scen}|{K}|{N}|{BS}"
        measure_cell(call, base, a.reps, 0)
        recs.append(dict(arm=arm, scenario=scen, K=K, N=N, BS=BS,
                         range_cold=f"c|{base}"))
    print(f"{scen} K{K} N{N} BS{BS} done", flush=True)
prof.stop()
out = HERE.parent / "results/iter13/iter18_cells.json"
json.dump(recs, open(out, "w"))
print("DONE")
