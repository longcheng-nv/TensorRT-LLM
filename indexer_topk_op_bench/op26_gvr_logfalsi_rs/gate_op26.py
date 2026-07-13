# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 exactness gate — BEFORE any timing.

Suites:
  A. op22rr bundles: {real, best, worst} x {512,1024,2048} x {fp32,bf16,fp16}
     x N {8192, 65536, 262144, 1048576} x BS {1, 16, 256} for the 4 arms
     (gvr_cutedsl, op26_1cta, gvr_multicta_cutedsl, op26_mc).
     worst (hr=0.05) is the undershoot-fallback pressure axis.
  B. Adversarial undershoot: preIdx fully DISJOINT from the true top-K
     (hr=0) + preIdx == exact top-K (hr=1, the P2 undershoot-creep corner
     found in the op26 smoke) on beta-tailed rows.
  C. 16-bit tie stress: quantization-collapsed plateaus around the K-th
     value (the op#7 bf16 randn pitfall, boundary-tie rank-scatter risk).

Criterion per row (first + last row of the replicated batch): indices in
[0, N) and unique; selected-VALUE multiset == torch.topk values, compared
SORTED on same-dtype logits (tie-order agnostic; matches gate_op22.py).

Usage: python3 gate_op26.py   (exit 1 on any mismatch)
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(HERE / "src"))

from sweep import DTYPES  # noqa: E402
from sweep_nsys import build_call  # noqa: E402
import bundle_data_rr  # noqa: E402
from gvr_op26_op import gvr_cutedsl_op26, gvr_multicta_op26  # noqa: E402
from gvr_op26_r0_op import gvr_r0_op26, gvr_r0f_op26  # noqa: E402
from gvr_op26_r0mc_op import (  # noqa: E402
    gvr_r0_mc_op26, gvr_r0mcc_op26, gvr_r0mcr_op26, gvr_r0_auto_op26,
)

DEV = "cuda"
# anchors (gvr_cutedsl / gvr_multicta_cutedsl) are NOT re-gated here: both
# already carry 1752 in-sweep exactness records from the op22rr campaign;
# gating only the new arms halves the compile load. The nsys campaign's
# per-cell BS=1 exactness covers the full grid for all four arms anyway.
ARMS = ["op26_1cta", "op26_mc"]
if os.environ.get("OP26_GATE_ARMS"):
    ARMS = [a.strip() for a in os.environ["OP26_GATE_ARMS"].split(",")]
SCENARIOS = ["real", "best", "worst"]
N_SPOTS = [8192, 65536, 262144]
BS_SPOTS = [1, 16, 256]

fails, errs, n_ok = [], [], 0


def check_out(out, logits, N, K, tag):
    global n_ok
    for r in (0, out.shape[0] - 1):
        o = out[r]
        idx = o.long()
        bad = (idx < 0) | (idx >= N)
        if bool(bad.any()):
            fails.append(f"{tag} row{r}: {int(bad.sum())} out-of-range/-1")
            return
        if len(set(o.tolist())) != K:
            fails.append(f"{tag} row{r}: dup indices uniq={len(set(o.tolist()))}")
            return
        sel = logits[r].gather(0, idx).float().sort().values
        ref = torch.topk(logits[r][:N].float(), K).values.sort().values
        if not torch.equal(sel, ref):
            d = (sel - ref).abs().max().item()
            fails.append(f"{tag} row{r}: value-set mismatch maxdiff={d:.3e}")
            return
    n_ok += 1


def run_suite_a():
    print("== Suite A: op22rr bundles ==", flush=True)
    for scen in SCENARIOS:
        for K in (512, 1024, 2048):
            for dt_name in ("fp32", "bf16", "fp16"):
                dtype = DTYPES[dt_name]
                for N in N_SPOTS:
                    try:
                        b = bundle_data_rr.get_bundle(scen, K, dtype, N)
                    except Exception as e:
                        errs.append(f"bundle {scen}/{K}/{dt_name}/{N}: {e}")
                        continue
                    logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
                    for BS in BS_SPOTS:
                        for arm in ARMS:
                            tag = f"A:{scen}|{arm}|K{K}|{dt_name}|N{N}|BS{BS}"
                            try:
                                call, keep, extra = build_call(
                                    arm, K, dtype, N, BS, cr,
                                    logits_row, preidx_row)
                                call()
                                torch.cuda.synchronize()
                                out = keep[3]
                                logits = keep[0]
                                check_out(out, logits, N, K, tag)
                                del call, keep
                            except Exception as e:
                                errs.append(f"{tag}: {type(e).__name__}: "
                                            f"{str(e)[:120]}")
                        torch.cuda.empty_cache()
        print(f"  {scen}: ok so far={n_ok} fails={len(fails)} errs={len(errs)}",
              flush=True)


def _run_arms_single(logits, pre, N, K, cr, tag):
    seq = torch.full((logits.shape[0],), N * cr, dtype=torch.int32, device=DEV)
    for arm, fn in (("op26_1cta", gvr_cutedsl_op26),
                    ("op26_mc", gvr_multicta_op26),
                    ("op26_r0", gvr_r0_op26),
                    ("op26_r0f", gvr_r0f_op26),
                    ("op26_r0mc", gvr_r0_mc_op26),
                    ("op26_r0mcc", gvr_r0mcc_op26),
                    ("op26_r0mcr", gvr_r0mcr_op26),
                    ("op26_r0auto", gvr_r0_auto_op26)):
        if arm not in ARMS:
            continue
        try:
            out = fn(logits, pre, seq, K, compress_ratio=cr)
            torch.cuda.synchronize()
            check_out(out, logits, N, K, f"{tag}|{arm}")
        except Exception as e:
            errs.append(f"{tag}|{arm}: {type(e).__name__}: {str(e)[:120]}")


def run_suite_b():
    print("== Suite B: adversarial undershoot (hr=0 / hr=1) ==", flush=True)
    torch.manual_seed(1234)
    for K, cr in ((512, 4), (1024, 4), (2048, 1)):
        for N in (16384, 131072):
            for dt_name in ("fp32", "bf16", "fp16"):
                dtype = DTYPES[dt_name]
                # beta-tailed row (heavier tail than randn)
                base = torch.distributions.Beta(2.0, 5.0).sample((N,))
                row = (base * 8.0).to(dtype).cuda().view(1, N).contiguous()
                topk_idx = torch.topk(row[0].float(), 2 * K).indices
                # hr=1: preIdx == exact top-K (P2 undershoot-creep corner)
                pre_hit = topk_idx[:K].int().view(1, K).contiguous()
                _run_arms_single(row, pre_hit, N, K, cr,
                                 f"B:hr1|K{K}|{dt_name}|N{N}")
                # hr=0: preIdx fully disjoint from the true top-K
                mask = torch.ones(N, dtype=torch.bool)
                mask[topk_idx.cpu()] = False
                rest = torch.arange(N)[mask]
                pre_miss = rest[torch.randperm(rest.numel())[:K]].int().cuda()
                pre_miss = pre_miss.view(1, K).contiguous()
                _run_arms_single(row, pre_miss, N, K, cr,
                                 f"B:hr0|K{K}|{dt_name}|N{N}")
    print(f"  ok so far={n_ok} fails={len(fails)} errs={len(errs)}", flush=True)


def run_suite_c():
    print("== Suite C: 16-bit tie plateaus ==", flush=True)
    torch.manual_seed(99)
    for K, cr in ((512, 4), (1024, 4), (2048, 1)):
        for N in (16384, 131072):
            for dt_name in ("bf16", "fp16"):
                dtype = DTYPES[dt_name]
                # plateau row: tie block crossing the K-th value in the
                # 16-bit grid. Plateau size capped at kC(16-bit)=5120: a
                # wider plateau (e.g. 5*K at K2048 -> count jumps
                # 1024 -> 10240 with no thr in [kK, kC]) is OUTSIDE the GVR
                # candidate-buffer design envelope — diag_tie_anchor.py
                # shows BOTH production anchors truncate identically there
                # (inherited §5 red-card family, not an op26 regression).
                # At the cap the plateau sits exactly on the kC boundary,
                # the strongest in-envelope tie stress.
                row = torch.rand(N) * 0.5
                plateau = torch.randperm(N)[: min(5 * K, 5120)]
                row[plateau] = 0.75
                winners = plateau[: K // 2]
                row[winners] = 0.9  # clear winners above the tie plateau
                row = row.to(dtype).cuda().view(1, N).contiguous()
                pre = torch.topk(row[0].float(), K).indices.int().view(1, K)
                pre = pre.contiguous()
                _run_arms_single(row, pre, N, K, cr,
                                 f"C:tie|K{K}|{dt_name}|N{N}")
    print(f"  ok so far={n_ok} fails={len(fails)} errs={len(errs)}", flush=True)


if __name__ == "__main__":
    suites = os.environ.get("OP26_GATE_SUITES", "A,B,C").upper().split(",")
    if "A" in suites:
        run_suite_a()
    if "B" in suites:
        run_suite_b()
    if "C" in suites:
        run_suite_c()
    print(f"\nGATE RESULT: ok={n_ok} fails={len(fails)} errs={len(errs)}")
    for f_ in fails[:40]:
        print("  FAIL", f_)
    for e_ in errs[:40]:
        print("  ERR ", e_)
    sys.exit(1 if (fails or errs) else 0)
