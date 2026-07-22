# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mid-BS ladder probe: shortlisted variants across BS {2,8,32,128} plus
direct-tier cells at high BS. Defines the v2 dispatch thresholds."""
import sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe import build, bundle, make_batch, exact_rows, timeit
from probe_cfg import rep_pr
import real_data_v4cap as v4
import real_data_v32 as v32

CASES = [
    # (model, isl, L, [(tb,cs,mv,ar,hs) shortlist])  mv=0 -> streaming, mv=-1 -> production dispatch (v1)
    ("flash", "16k", 22, [(-1,), (512, 1, 0, 4, 2), (1024, 1, 0, 4, 2)]),        # npad 4160 direct tier
    ("flash", "32k", 22, [(-1,), (512, 1, 0, 4, 2), (1024, 1, 0, 4, 2)]),        # npad 8256 direct tier
    ("v32", "4k", 34, [(-1,), (512, 1, 0, 4, 2)]),                                # npad 4160 K=2048!
    ("v32", "16k", 34, [(-1,), (512, 1, 9, 8, 1), (512, 1, 0, 4, 2), (1024, 1, 0, 6, 4)]),
    ("flash", "128k", 22, [(-1,), (1024, 1, 9, 4, 2), (512, 1, 0, 4, 2), (1024, 2, 0, 4, 2)]),
    ("pro", "128k", 30, [(-1,), (1024, 1, 9, 6, 2), (512, 1, 0, 6, 4), (1024, 2, 0, 6, 4)]),
    ("flash", "256k", 22, [(-1,), (1024, 2, 9, 6, 2), (512, 1, 0, 4, 2), (1024, 2, 0, 6, 4)]),
    ("v32", "128k", 34, [(-1,), (1024, 4, 9, 6, 2), (1024, 1, 0, 4, 2), (1024, 2, 0, 4, 2)]),
    ("flash", "1024k", 22, [(-1,), (1024, 8, 8, 4, 2), (1024, 1, 0, 4, 2), (1024, 2, 0, 4, 2)]),
    ("pro", "1024k", 30, [(-1,), (1024, 8, 8, 6, 2), (1024, 1, 0, 4, 4), (1024, 2, 0, 6, 4)]),
    ("v32", "256k", 34, [(-1,), (512, 8, 10, 6, 4), (1024, 1, 0, 4, 2), (1024, 2, 0, 4, 2)]),
]
BS_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 1024]

mod = build("kernel_bs")
pr = rep_pr()
print("[probe_mid] built", flush=True)
for model, isl, L, shortlist in CASES:
    b = bundle(model, isl, L)
    K, N = b["K"], b["N"]
    for bs in BS_LIST:
        lg, pre = make_batch(b, bs)
        out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
        target = pr.get((model, isl, L, bs))
        res = []
        for v in shortlist:
            if v[0] == -1:
                call = lambda: mod.run(lg, pre, N, out)
                tag = "prod"
            else:
                tb, cs, mv, ar, hs = v
                call = lambda: mod.run_cfg(lg, pre, N, out, tb, cs, mv, ar, hs)
                tag = f"({tb},{cs},{mv},{ar},{hs})"
            call(); torch.cuda.synchronize()
            bad = exact_rows(b, out, bs)
            if bad:
                print(f"  {model}_{isl} BS{bs} {tag} INEXACT {bad}", flush=True)
                continue
            for _ in range(5):
                call()
            torch.cuda.synchronize()
            res.append((timeit(call, reps=11), tag))
        res.sort()
        line = " ".join(f"{t}={u:.1f}" for u, t in res)
        print(f"{model}_{isl}_L{L:02d} BS{bs:5d} pr={target or 0:8.2f} "
              f"WIN {res[0][1]} x{(target or 0)/res[0][0]:6.3f} | {line}", flush=True)
        del lg, pre, out
    v4._bundle_cache.clear(); v32._bundle_cache.clear()
    torch.cuda.empty_cache()
