#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op29 gate: gvr29 (HBE on + off) exactness on op22rr bundles (real hints).

Cells: 3 scenarios x K {512,1024,2048} x N {32768..262144} x BS {4, 64, 520}.
BS=520 (>512, no cluster) exercises HBE at every N; BS 4/64 exercise it below
the cluster floor and the baseline cluster paths above it.
Criterion: tie-aware sorted value-multiset vs torch.topk + unique indices.
Also records the HBE engagement telemetry is implicit (perf pilot separates).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "op22_temporal_fixed_hr_bench"))
import bundle_data_rr  # noqa: E402
from gvr29_op import gvr29_topk, plan  # noqa: E402

DEV = "cuda"
fails = errs = ok = 0
for scen in ("real", "best", "worst"):
    for K in (512, 1024, 2048):
        for N in (32768, 65536, 131072, 262144):
            b = bundle_data_rr.get_bundle(scen, K, torch.float32, N,
                                          device=DEV)
            row = b["logits"][0, :N].float()
            ref_vals = torch.topk(row, K).values.sort().values
            for BS in (4, 64, 520):
                logits = b["logits"].to(torch.float32).expand(
                    BS, -1).contiguous()
                pre = b["preIdx"].to(torch.int32).expand(BS, -1).contiguous()
                sl = torch.full((BS,), N, dtype=torch.int32, device=DEV)
                md = plan(sl)
                for hbe in (True, False):
                    tag = f"{scen} K={K} N={N} BS={BS} hbe={hbe}"
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
                            print("FAIL", tag, flush=True)
                    except Exception as e:
                        errs += 1
                        print("ERR ", tag, type(e).__name__, str(e)[:90],
                              flush=True)
                del logits, pre
                torch.cuda.empty_cache()
        print(f"[gate] {scen} K={K}: ok={ok}", flush=True)
print(f"\nGATE: ok={ok} fails={fails} errs={errs}")
sys.exit(1 if (fails or errs) else 0)
