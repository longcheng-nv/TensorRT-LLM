# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41 phase 6: KF champion (gvr_topk_r7_bcast_v2, 266bd37d) third-arm
integration gate — tie-aware exactness on all 75 envelope cells x BS
{2, 16, 256, 1024} + adversarial mini-track (constant row, near-tie row)."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP41 = HERE.parent
BENCH = OP41.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, make_batch, exact_rows  # noqa: E402
from bs38_nsys import all_cells  # noqa: E402

CHAMP = (BENCH / "op26_r0_upstream_port_report" / "kf_campaign" /
         "gvr-topk-pr16457-fresh" / "harvest_fresh" / "gvr_topk_r7_bcast_v2")


def build_champ():
    from torch.utils.cpp_extension import load
    (CHAMP / "build_pt").mkdir(exist_ok=True)
    return load(name="op41_champ_r7",
                sources=[str(CHAMP / "kernel.cu"), str(CHAMP / "main.cpp")],
                build_directory=str(CHAMP / "build_pt"),
                extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17",
                                   "-gencode",
                                   "arch=compute_100a,code=sm_100a"],
                extra_ldflags=["-lcuda"],
                extra_include_paths=[str(CHAMP)],
                verbose=False)


def main():
    mod = build_champ()
    bad = 0
    for model, isl, L in all_cells():
        b = bundle(model, isl, L)
        K, N = b["K"], b["N"]
        for bs in (2, 16, 256, 1024):
            lg, pre = make_batch(b, bs)
            out = torch.full((bs, K), -7, dtype=torch.int32, device="cuda")
            mod.run(lg, pre, N, out)
            torch.cuda.synchronize()
            e = exact_rows(b, out, bs)
            if e:
                print(f"GATE FAIL {model}_{isl}_L{L:02d} BS{bs}: {e}")
                bad += 1
        print(f"[gate] {model}_{isl}_L{L:02d} done", flush=True)
    # adversarial mini-track (K=512/1024/2048 shapes)
    for K, npad in ((512, 65600), (1024, 131136), (2048, 65600)):
        for tag, row in (
            ("const", torch.zeros(npad)),
            ("neartie", torch.cat([torch.full((npad - K,), 1.0),
                                   torch.full((K,), 1.0 + 1e-7)])
             [torch.randperm(npad)]),
        ):
            N = npad - 32  # exercise a padded tail
            lg = row.clone()
            lg[N:] = -3.4e38
            lg = lg.unsqueeze(0).expand(16, -1).contiguous().cuda()
            pre = torch.randint(0, N, (16, K), dtype=torch.int32,
                                device="cuda")
            out = torch.full((16, K), -7, dtype=torch.int32, device="cuda")
            mod.run(lg, pre, N, out)
            torch.cuda.synchronize()
            ref = torch.topk(lg[0, :N].float(), K).values.sort().values
            ok = True
            for i in range(16):
                idx = out[i].to(torch.int64)
                if (idx.min() < 0 or idx.max() >= N
                        or torch.unique(idx).numel() != K):
                    ok = False
                    break
                sel = lg[0, :N].float()[idx].sort().values
                if not torch.equal(sel, ref):
                    ok = False
                    break
            if not ok:
                print(f"GATE FAIL adversarial {tag} K{K} npad{npad}")
                bad += 1
            else:
                print(f"[gate] adversarial {tag} K{K} OK", flush=True)
    print(f"[gate] fails: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
