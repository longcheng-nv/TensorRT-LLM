# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 REAL-capture chapter — exactness pre-gate BEFORE any timing.

All 13 arms x 3 models x 3 dtypes x layer spot-set {first, middle, last}
x BS {1, 16} on the real last-decode-step bundles (real_data_v2).
Criterion per checked row (row 0 and row BS-1 of the replicated batch):
  vdiff == 0 (sorted selected VALUES == sorted torch.topk values on the
  SAME-dtype logits — tie-order agnostic; GVR row order is atomicAdd-
  nondeterministic) AND n_neg == 0 (no unfilled slots).

Also serves as the shared-JIT warmer: one single-GPU pass builds every
cpp_extension / cuteDSL kernel cache before the 8-GPU fleet launches
(avoids the NFS _build lock stampede).

Usage: python3 gate_op22real.py [--full-layers]   (exit 1 on any mismatch)
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "harness"))

import real_data_v2 as RD2                                     # noqa: E402
from sweep_op22_real import arms_for, build_call, _pin_env     # noqa: E402

DEV = "cuda"
BS_SPOTS = [1, 16]
FULL = "--full-layers" in sys.argv

fails, errs, n_ok = [], [], 0
for model in RD2.MODELS:
    m = RD2.MODELS[model]
    ls = m["layers"]
    layers = ls if FULL else sorted({ls[0], ls[len(ls) // 2], ls[-1]})
    for dt_name in ("fp32", "bf16", "fp16"):
        for L in layers:
            b = RD2.get_real_bundle_v2(model, L, dt_name)
            logits_row, preidx_row = b["logits"], b["preIdx"]
            K, cr, N, ref = b["K"], b["cr"], b["N"], b["ref"]
            for BS in BS_SPOTS:
                for arm, op, falsi, dist in arms_for(model, dt_name):
                    tag = f"{arm}|{model}|L{L}|{dt_name}|BS{BS}"
                    try:
                        _pin_env(falsi, dist)
                        call, keep, extra = build_call(
                            op, K, RD2.DTYPES[dt_name], N, BS, cr,
                            logits_row, preidx_row)
                        _pin_env(falsi, dist)
                        call()
                        torch.cuda.synchronize()
                        if arm == "flashinfer_topk":
                            import flashinfer
                            idx_all = flashinfer.top_k(keep[0], K)[1]
                            rows = [idx_all[0].to(torch.int32),
                                    idx_all[BS - 1].to(torch.int32)]
                        else:
                            out = keep[3]
                            assert out.dtype == torch.int32 \
                                and out.shape == (BS, K), (tag, out.shape)
                            rows = [out[0], out[BS - 1]]
                        bad = None
                        for r in rows:
                            vd, rc, nn = RD2.value_metrics(
                                r, logits_row, ref, K)
                            if vd != 0 or nn != 0:
                                bad = (vd, rc, nn)
                        if bad:
                            fails.append((tag, *bad))
                            print(f"FAIL {tag}: vdiff={bad[0]:.3e} "
                                  f"recall={bad[1]:.4f} n_neg={bad[2]}",
                                  flush=True)
                        else:
                            n_ok += 1
                        del call, keep
                        torch.cuda.empty_cache()
                    except Exception as e:
                        errs.append((tag, f"{type(e).__name__}: {e}"))
                        print(f"ERR  {tag}: {type(e).__name__}: "
                              f"{str(e)[:200]}", flush=True)
        print(f"[gate] {model} {dt_name} done "
              f"(ok={n_ok} fail={len(fails)} err={len(errs)})", flush=True)

print(f"\nGATE SUMMARY: ok={n_ok} fail={len(fails)} err={len(errs)}")
for t, *r in fails[:20]:
    print(f"  FAIL {t}: {r}")
for t, e in errs[:20]:
    print(f"  ERR  {t}: {e}")
sys.exit(1 if (fails or errs) else 0)
