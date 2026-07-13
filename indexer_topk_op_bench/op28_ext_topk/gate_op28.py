#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op28 — exactness pre-gate on the op22rr bundles, BEFORE any timing.

All 6 arms x 3 scenarios x K {512,1024,2048} x N {65536, 262144, 1048576} x
BS {1, 16, 64} (fp32). BS=64 with N>=131072 exercises the sglang_v2
persistent-cluster 2-kernel path. Criterion per row (first + last row):
  - indices in [0, N) and UNIQUE,
  - selected-VALUE multiset == torch.topk values, compared SORTED.

Usage:  python3 gate_op28.py  (exit 1 on any mismatch)
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "harness"))
sys.path.insert(0, str(HERE.parents[0] / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(HERE))

from sweep import DTYPES  # noqa: E402
from ops_ext import build_call_ext  # noqa: E402
from sweep_op28 import ops_for  # noqa: E402
import bundle_data_rr  # noqa: E402

DEV = "cuda"
SCENARIOS = ["real", "best", "worst"]
N_SPOTS = [65536, 262144, 1048576]
BS_SPOTS = [1, 16, 64]

fails, errs, n_ok = [], [], 0
for scen in SCENARIOS:
    for K in (512, 1024, 2048):
        dt_name = "fp32"
        dtype = DTYPES[dt_name]
        for N in N_SPOTS:
            b = bundle_data_rr.get_bundle(scen, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            ref_vals = torch.topk(logits_row[0, :N].float(), K
                                  ).values.sort().values
            for BS in BS_SPOTS:
                for op in ops_for(dt_name, K):
                    tag = f"{scen} {op} K={K} N={N} BS={BS}"
                    try:
                        call, keep, extra = build_call_ext(
                            op, K, dtype, N, BS, cr, logits_row, preidx_row)
                        call()
                        torch.cuda.synchronize()
                        # find the output tensor: int32 [BS, K] in keep, or
                        # flashinfer public API returns fresh tensors -> redo
                        if op == "flashinfer_topk":
                            import flashinfer
                            logits = logits_row.to(dtype).expand(
                                BS, -1).contiguous()
                            _, idx_t = flashinfer.top_k(logits, K)
                            idx_t = idx_t.to(torch.int32)
                        elif op == "flashinfer_topk_i32":
                            from flashinfer.topk import topk_clusters_exact
                            logits = logits_row.to(dtype).expand(
                                BS, -1).contiguous()
                            idx_t, _ = topk_clusters_exact(
                                logits, K, output_values=False,
                                out_dtype=torch.int32)
                        elif op == "sglang_v2":
                            # ops_ext keep = [logits, seq_nod, out, md]
                            idx_t = keep[2]
                        else:
                            # harness _build_inputs keep = [logits, seq_div,
                            # seq_nod, out, ...] — position fixed; do NOT scan
                            # by dtype/shape (preIdx hint is also (BS,K) i32)
                            idx_t = keep[3]
                        ok = True
                        for r in (0, BS - 1) if BS > 1 else (0,):
                            idx = idx_t[r].long()
                            if idx.min() < 0 or idx.max() >= N or \
                               idx.unique().numel() != K:
                                ok = False
                                break
                            got = logits_row[0, idx].float().sort().values
                            if not torch.equal(got, ref_vals):
                                ok = False
                                break
                        n_ok += ok
                        if not ok:
                            fails.append(tag)
                            print("FAIL", tag, flush=True)
                        del call, keep
                        torch.cuda.empty_cache()
                    except Exception as e:
                        errs.append((tag, f"{type(e).__name__}: {e}"))
                        print("ERR ", tag, type(e).__name__, str(e)[:100],
                              flush=True)
        print(f"[gate] {scen} K={K}: ok so far {n_ok}", flush=True)

print(f"\nGATE: ok={n_ok} fails={len(fails)} errs={len(errs)}")
for t in fails:
    print("  FAIL", t)
for t, e in errs:
    print("  ERR ", t, e)
sys.exit(1 if (fails or errs) else 0)
