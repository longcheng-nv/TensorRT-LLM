# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 W4 — exactness pre-gate on the generated bundles, BEFORE any timing.

All 5 campaign ops x 3 scenarios x 9 (dtype,K) x N {65536, 1048576} x BS
{1,16}. Criterion per row (first + last row of the replicated batch):
  - indices in [0, N) and UNIQUE (uniq == K),
  - selected-VALUE multiset == torch.topk(logits_row, K) values, compared
    SORTED (tie-order agnostic) — GVR output row order is runtime-
    nondeterministic (atomicAdd cursor; op21 iter12 LEARNINGS), so only the
    sorted set/multiset is a valid A/B criterion.
Reference is torch.topk on the SAME-dtype logits (matches harness
smoke_v2_exactness.py), so 16-bit tie handling is honest.

Usage:  python3 gate_op22.py  (exit 1 on any mismatch)
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "harness"))

from sweep import DTYPES  # noqa: E402
from sweep_nsys import build_call  # noqa: E402

import bundle_data  # noqa: E402
from sweep_op22 import ops_for  # noqa: E402

DEV = "cuda"
SCENARIOS = ["real", "best", "worst"]
N_SPOTS = [65536, 1048576]
BS_SPOTS = [1, 16]

fails, errs, n_ok = [], [], 0
for scen in SCENARIOS:
    for K in (512, 1024, 2048):
        for dt_name in ("fp32", "bf16", "fp16"):
            dtype = DTYPES[dt_name]
            for N in N_SPOTS:
                b = bundle_data.get_bundle(scen, K, dtype, N)
                logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
                ref_vals = torch.topk(logits_row[0, :N].float(), K
                                      ).values.sort().values
                for BS in BS_SPOTS:
                    for op in ops_for(dt_name, K):
                        try:
                            call, keep, extra = build_call(
                                op, K, dtype, N, BS, cr,
                                logits_row, preidx_row)
                            call()
                            torch.cuda.synchronize()
                            out = keep[3]  # [BS, K] int32 indices
                            ok = True
                            why = ""
                            for r in {0, BS - 1}:
                                idx = out[r].long()
                                if (idx < 0).any() or (idx >= N).any():
                                    ok, why = False, "idx_out_of_range"
                                    break
                                if idx.unique().numel() != K:
                                    ok, why = False, "dup_idx"
                                    break
                                got = keep[0][r][idx].float().sort().values
                                if not torch.equal(got, ref_vals):
                                    nbad = int((got != ref_vals).sum())
                                    ok, why = False, f"val_multiset({nbad})"
                                    break
                            if ok:
                                n_ok += 1
                            else:
                                fails.append((scen, op, K, dt_name, N, BS, why))
                                print(f"MISMATCH {scen} {op} K={K} {dt_name} "
                                      f"N={N} BS={BS}: {why}", flush=True)
                        except Exception as e:
                            errs.append((scen, op, K, dt_name, N, BS,
                                         f"{type(e).__name__}: {str(e)[:80]}"))
                        finally:
                            try:
                                del call, keep
                            except NameError:
                                pass
                            torch.cuda.empty_cache()
                print(f"cell {scen} K={K} {dt_name} N={N} done", flush=True)

print(f"\nGATE exact={n_ok} mismatches={len(fails)} errors={len(errs)}")
for e in errs:
    print("  ERR", e)
sys.exit(1 if (fails or errs) else 0)
