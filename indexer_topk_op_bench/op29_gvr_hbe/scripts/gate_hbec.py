#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op31 HBE-C gate: exactness of the tier-5 cluster hint path (GVR29_HBEC=1).

Cells: 3 scenarios x K {512,1024,2048} x N {131072..1048576} x
BS {1,4,64,256,512} x hint {real (+1 for cr=1), adversarial bottom-K,
out-of-range garbage}. BS<=30 exercises the fused small-batch HBE-C kernel,
BS>30 the persistent pool + epilogue. use_hbe=False parity arm re-checks the
stock cluster path in the same build.
Criterion: sorted value-multiset == torch.topk values + unique in-range idx.
"""
import os
import sys
from pathlib import Path

os.environ["GVR29_HBEC"] = "1"          # before first transform (static read)

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "op22_temporal_fixed_hr_bench"))
import bundle_data_rr  # noqa: E402
from gvr29_op import gvr29_topk, plan  # noqa: E402

DEV = "cuda"
NS = (131072, 262144, 524288, 1048576)
BSS = (1, 4, 64, 256, 512)
fails = errs = ok = 0
for scen in ("real", "best", "worst"):
    for K in (512, 1024, 2048):
        for N in NS:
            b = bundle_data_rr.get_bundle(scen, K, torch.float32, N,
                                          device=DEV)
            row = b["logits"][0, :N].float()
            ref_vals = torch.topk(row, K).values.sort().values
            pre_real = b["preIdx"].to(torch.int32)
            if b["cr"] == 1:
                pre_real = (pre_real + 1) % N     # production kernel-read
            pre_adv = torch.topk(-row, K).indices.to(torch.int32).view(1, K)
            pre_bad = torch.randint(-N, 2 * N, (1, K), dtype=torch.int32,
                                    device=DEV)
            for BS in BSS:
                logits = b["logits"].to(torch.float32).expand(
                    BS, -1).contiguous()
                sl = torch.full((BS,), N, dtype=torch.int32, device=DEV)
                md = plan(sl)
                arms = [("real", pre_real, True), ("adv", pre_adv, True),
                        ("bad", pre_bad, True), ("off", pre_real, False)]
                for tag_h, pre1, hbe in arms:
                    pre = pre1.expand(BS, -1).contiguous()
                    tag = (f"{scen} K={K} N={N} BS={BS} hint={tag_h} "
                           f"hbe={hbe}")
                    try:
                        out = gvr29_topk(logits, sl, K, pre, metadata=md,
                                         max_seq_len=N, use_hbe=hbe)
                        good = True
                        for r in (0, BS - 1):
                            idx = out[r].long()
                            if idx.min() < 0 or idx.max() >= N or \
                               idx.unique().numel() != K:
                                good = False
                                break
                            got = row[idx].sort().values
                            if not torch.equal(got, ref_vals):
                                good = False
                                break
                        if good:
                            ok += 1
                        else:
                            fails += 1
                            print(f"FAIL {tag}", flush=True)
                    except Exception as e:
                        errs += 1
                        print(f"ERR {tag}: {type(e).__name__}: "
                              f"{str(e)[:120]}", flush=True)
                del logits, sl, md
                torch.cuda.empty_cache()
            print(f"{scen} K={K} N={N} done (ok={ok})", flush=True)
print(f"GATE ok={ok} fails={fails} errs={errs}", flush=True)
sys.exit(1 if (fails or errs) else 0)
