# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""rung-0.1b: nsys pure-kernel spans for the probe kernels on key shapes.
Run under: env -u GITHUB_TOKEN -u HF_TOKEN PYTHONNOUSERSITE=1 nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop -o /tmp/op35_rung0 \
  python3 scripts/rung0_nsys.py --gpu 0
Then: nsys stats --report cuda_gpu_kern_sum /tmp/op35_rung0.nsys-rep
"""
import argparse
import os

import torch
import torch.cuda.profiler as prof
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()
torch.cuda.set_device(a.gpu)

ext = load(name="floor_probe", sources=[os.path.join(HERE, "../src/floor_probe.cu")],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_100,code=sm_100"],
           build_directory=os.environ.get("BUILD_DIR", "/tmp/op35_build"), verbose=False)
EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

CASES = [  # (BS, N, cpr) — matched to the frontier-comparison shapes
    (1, 131072, 148), (1, 262144, 148), (1, 1048576, 296),
    (32, 262144, 8), (256, 262144, 1), (1024, 65536, 1),
]
# NVTX name encodes the case; nsys kern_sum aggregates per kernel name, so we
# separate cases into sequential phases with distinctive python-side markers
# and rely on per-instance stats via sqlite later if needed. For rung-0 the
# kern_sum avg per (kernel, case-block) is extracted by running one case per
# profiler session? Simpler: emit all, use nvtx_kern_sum.
import torch.cuda.nvtx as nvtx

prof.start()
for BS, N, cpr in CASES:
    x = torch.rand(BS, N, device="cuda") + 1.0
    out = torch.zeros(BS, device="cuda")
    tickets = torch.zeros(BS, dtype=torch.int32, device="cuda")
    q = torch.quantile(x[0].float(), torch.tensor([0.97, 0.996], device="cuda"))
    t_lo, t_hi = q[0].item(), q[1].item()
    cap = max(4096, int(N * 0.05))
    cand_v = torch.zeros(BS, cap, device="cuda")
    cand_i = torch.zeros(BS, cap, dtype=torch.int32, device="cuda")
    counts = torch.zeros(BS * 2, dtype=torch.int32, device="cuda")
    ext.stream_reduce(x, out, tickets, cpr)
    ext.filter_append(x, t_hi, t_lo, cand_v, cand_i, counts, tickets, cpr)
    torch.cuda.synchronize()
    for _ in range(a.reps):
        EVICT.uniform_()
        torch.cuda.synchronize()
        with nvtx.range(f"R|{BS}|{N}|{cpr}"):
            ext.stream_reduce(x, out, tickets, cpr)
        torch.cuda.synchronize()
        EVICT.uniform_()
        torch.cuda.synchronize()
        with nvtx.range(f"F|{BS}|{N}|{cpr}"):
            ext.filter_append(x, t_hi, t_lo, cand_v, cand_i, counts, tickets, cpr)
        torch.cuda.synchronize()
    with nvtx.range(f"E|{BS}|{N}|{cpr}"):
        ext.empty_launch(148)
    torch.cuda.synchronize()
prof.stop()
print("done")
