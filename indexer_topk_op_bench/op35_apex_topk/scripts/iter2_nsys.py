# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""iter-1: nsys A/B of filter v1 vs v2 (SMEM staging) vs reduce/reduce2 on the
6 calibration shapes. Correctness cross-check of v2 admits vs torch reference.
Run: env -u GITHUB_TOKEN -u HF_TOKEN PYTHONNOUSERSITE=1 nsys profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true \
  -o /tmp/op35_iter1 python3 scripts/iter1_nsys.py --gpu 1
"""
import argparse
import os

import torch
import torch.cuda.profiler as prof
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=1)
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()
torch.cuda.set_device(a.gpu)

ext = load(name="floor_probe", sources=[os.path.join(HERE, "../src/floor_probe.cu")],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_100,code=sm_100"],
           build_directory=os.environ.get("BUILD_DIR", "/tmp/op35_build"), verbose=False)
EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

CASES = [(1, 131072, 148), (1, 262144, 148), (1, 1048576, 296),
         (32, 262144, 8), (256, 262144, 1), (1024, 65536, 1)]

# correctness screen for v2 (admit set == reference filter set)
for BS, N, cpr in [(1, 131072, 8), (32, 65536, 4), (4, 262144, 37)]:
    x = torch.rand(BS, N, device="cuda") + 1.0
    K = 512
    q = torch.quantile(x[0].float(), torch.tensor([1 - 2.0 * K / N, 1 - 0.8 * K / N], device="cuda"))
    t_lo, t_hi = q[0].item(), q[1].item()
    cap = max(4096, int(N * 0.08))
    cand_v = torch.full((BS, cap), -1.0, device="cuda")
    cand_i = torch.zeros(BS, cap, dtype=torch.int32, device="cuda")
    counts = torch.zeros(BS * 2, dtype=torch.int32, device="cuda")
    tickets = torch.zeros(BS, dtype=torch.int32, device="cuda")
    ext.filter_v3(x, t_hi, t_lo, cand_v, cand_i, counts, tickets, cpr)
    torch.cuda.synchronize()
    ok = True
    for r in range(BS):
        n_adm = counts.view(BS, 2)[r, 1].item()
        ref_mask = x[r] >= t_lo
        ref_n = int(ref_mask.sum())
        if n_adm != ref_n:
            ok = False; print(f"  ADMIT COUNT MISMATCH row {r}: {n_adm} vs {ref_n}"); break
        got_v = cand_v[r, :n_adm].sort().values
        ref_v = x[r][ref_mask].sort().values
        if not torch.equal(got_v, ref_v):
            ok = False; print(f"  VALUE SET MISMATCH row {r}"); break
        got_i = cand_i[r, :n_adm].long().sort().values
        ref_i = ref_mask.nonzero().flatten().sort().values
        if not torch.equal(got_i, ref_i):
            ok = False; print(f"  INDEX SET MISMATCH row {r}"); break
        n_hi = counts.view(BS, 2)[r, 0].item()
        if n_hi != int((x[r] >= t_hi).sum()):
            ok = False; print(f"  HI COUNT MISMATCH row {r}"); break
    print(f"v2 exactness BS{BS} N{N} cpr{cpr}: {'OK' if ok else 'FAIL'}")

import torch.cuda.nvtx as nvtx
prof.start()
for BS, N, cpr in CASES:
    x = torch.rand(BS, N, device="cuda") + 1.0
    out = torch.zeros(BS, device="cuda")
    tickets = torch.zeros(BS, dtype=torch.int32, device="cuda")
    K = 512
    q = torch.quantile(x[0].float(), torch.tensor([1 - 2.0 * K / N, 1 - 0.8 * K / N], device="cuda"))
    t_lo, t_hi = q[0].item(), q[1].item()
    cap = max(4096, int(N * 0.08))
    cand_v = torch.zeros(BS, cap, device="cuda")
    cand_i = torch.zeros(BS, cap, dtype=torch.int32, device="cuda")
    counts = torch.zeros(BS * 2, dtype=torch.int32, device="cuda")
    ext.stream_reduce2(x, out, tickets, cpr)
    ext.filter_v3(x, t_hi, t_lo, cand_v, cand_i, counts, tickets, cpr)
    torch.cuda.synchronize()
    for _ in range(a.reps):
        EVICT.uniform_(); torch.cuda.synchronize()
        ext.stream_reduce2(x, out, tickets, cpr)
        torch.cuda.synchronize()
        EVICT.uniform_(); torch.cuda.synchronize()
        ext.filter_v3(x, t_hi, t_lo, cand_v, cand_i, counts, tickets, cpr)
        torch.cuda.synchronize()
prof.stop()
print("profiled")
