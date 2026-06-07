#!/usr/bin/env python3
"""Iteration 6 (GA10): production NVF4 fp4 path on silicon (local B200).

The production DSV4 MoE weights are NVF4 (fp4 e2m1 + e4m3 per-16 block scale), not the
per-tensor fp8 of iter3. fp4 weights are 0.5 B/elem vs bf16 2 B → 4× less weight traffic,
which is where the 40% target lives in memory-bound decode. This drives the REAL nvfp4
`torch._scaled_mm` (torchao nvfp4_quantize + to_blocked swizzle) across the DSV4 FC1/FC2
BS=1..512 shapes and measures: twin_fidelity, fp4-vs-bf16 speedup, SOL, roofline regime.

Production-faithful: WEIGHT is pre-quantized once (static), only the ACTIVATION is
quantized per call. TF32 off. Every number measured here. Run on a FREE B200.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, nvfp4_quantize
from torchao.prototype.mx_formats.utils import to_blocked

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda"
B200_BF16_PEAK_TFLOPS = 2250.0
B200_FP4_PEAK_TFLOPS = 9000.0  # B200 fp4 tensor-core peak (approx, for SOL)
BLK = 16

DSV4_TOPK = 6
BS_LIST = [1, 8, 32, 64, 128, 256, 512]
GEMMS = [("FC1", 4096, 4096), ("FC2", 2048, 4096)]


def _bench(fn, iters=40, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def _nvfp4_operands(X):
    """Return (data_fp4 viewed as float4_e2m1fn_x2, swizzled e4m3 block scale)."""
    a, b = nvfp4_quantize(X, BLK)
    scale, data = (
        (a, b)
        if (a.dtype == torch.float8_e4m3fn and a.dim() == 2 and a.shape[1] * BLK == X.shape[1])
        else (b, a)
    )
    if data.dtype != torch.float4_e2m1fn_x2:
        data = data.view(torch.float4_e2m1fn_x2)
    return data, to_blocked(scale)


def run():
    rows = []
    g = torch.Generator(device=DEV).manual_seed(42)
    for bs in BS_LIST:
        M = bs * DSV4_TOPK
        for gemm, K, N in GEMMS:
            A = torch.randn(M, K, device=DEV, dtype=torch.float32, generator=g) * 0.5
            W = torch.randn(N, K, device=DEV, dtype=torch.float32, generator=g) * 0.5  # [N,K]

            A_bf16, W_bf16 = A.to(torch.bfloat16), W.to(torch.bfloat16)

            def mm_bf16():
                return torch.mm(A_bf16, W_bf16.t())

            D_bf16 = mm_bf16().float()

            # WEIGHT pre-quantized once (production: static fp4 weights)
            Wd, Ws = _nvfp4_operands(W)
            Ad, As = _nvfp4_operands(A)  # activation pre-quantized for GEMM-ONLY timing

            # GEMM-only fp4 kernel (the fair kernel-vs-kernel comparison: in production the
            # activation quant is FUSED into the upstream op, not a separate eager call)
            def gemm_fp4():
                return torch._scaled_mm(
                    Ad, Wd.t(), scale_a=As, scale_b=Ws, out_dtype=torch.bfloat16
                )

            # the unfused eager activation-quant, timed separately to expose its cost
            def quant_act():
                return _nvfp4_operands(A)

            try:
                D_fp4 = gemm_fp4().float()
            except Exception as ex:
                rows.append(dict(bs=bs, gemm=gemm, M=M, error=str(ex)[:140]))
                print(f"  [skip] bs{bs} {gemm}: {str(ex)[:90]}")
                continue

            # measured accuracy vs bf16 ref
            meas_rel = float(torch.norm(D_bf16 - D_fp4) / (torch.norm(D_bf16) + 1e-30))

            # TWIN fidelity: dequantize the SAME nvfp4 operands, fp64-accumulate on GPU
            A_deq = NVFP4Tensor.to_nvfp4(A, block_size=BLK).dequantize().double()
            W_deq = NVFP4Tensor.to_nvfp4(W, block_size=BLK).dequantize().double()
            D_twin = torch.mm(A_deq, W_deq.t())
            twin_fid = float(
                torch.norm(D_twin - D_fp4.double()) / (torch.norm(D_fp4.double()) + 1e-30)
            )

            # latency / SOL  (weights AND activation pre-quantized → GEMM-only kernel time)
            t_bf16 = _bench(mm_bf16)
            t_fp4 = _bench(gemm_fp4)
            t_quant = _bench(quant_act)  # unfused eager activation-quant cost
            flop = 2.0 * M * K * N
            tflops_bf16 = flop / (t_bf16 * 1e-3) / 1e12
            tflops_fp4 = flop / (t_fp4 * 1e-3) / 1e12
            speedup_gemm = t_bf16 / t_fp4  # kernel-vs-kernel (fair)
            speedup_eager = t_bf16 / (t_fp4 + t_quant)  # incl. unfused quant
            regime = "launch" if M < 16 else ("memory" if M < 128 else "compute")
            rows.append(
                dict(
                    bs=bs,
                    gemm=gemm,
                    M=M,
                    K=K,
                    N=N,
                    regime=regime,
                    twin_fidelity=twin_fid,
                    measured_rel=meas_rel,
                    t_bf16_us=t_bf16 * 1e3,
                    t_fp4_gemm_us=t_fp4 * 1e3,
                    t_quant_us=t_quant * 1e3,
                    speedup_gemm=speedup_gemm,
                    speedup_with_eager_quant=speedup_eager,
                    sol_bf16_pct=100 * tflops_bf16 / B200_BF16_PEAK_TFLOPS,
                    sol_fp4_pct=100 * tflops_fp4 / B200_FP4_PEAK_TFLOPS,
                )
            )
            print(
                f"  bs{bs:<4} {gemm} M={M:<5} twin_fid={twin_fid:.2e} meas_rel={meas_rel:.3f} "
                f"GEMM_speedup={speedup_gemm:.2f}x (eager+quant {speedup_eager:.2f}x) ({regime})"
            )

    outdir = Path(__file__).resolve().parent / "results"
    outdir.mkdir(exist_ok=True)
    ok = [r for r in rows if "error" not in r]
    if ok:
        with open(outdir / "iter6_silicon_nvf4.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ok[0].keys()))
            w.writeheader()
            w.writerows(ok)
    findings = {
        "device": torch.cuda.get_device_name(0),
        "twin_fidelity_max": max((r["twin_fidelity"] for r in ok), default=None),
        "measured_rel_mean": float(np.mean([r["measured_rel"] for r in ok])) if ok else None,
        "fp4_GEMM_speedup_by_shape": {
            f"{r['gemm']}_bs{r['bs']}": round(r["speedup_gemm"], 3) for r in ok
        },
        "fp4_GEMM_speedup_max": max((r["speedup_gemm"] for r in ok), default=None),
        "fp4_GEMM_speedup_min": min((r["speedup_gemm"] for r in ok), default=None),
        "speedup_with_eager_quant_max": max(
            (r["speedup_with_eager_quant"] for r in ok), default=None
        ),
        "note": "speedup_gemm = kernel-vs-kernel (production fuses act-quant); "
        "speedup_with_eager_quant includes the unfused eager nvfp4_quantize overhead",
    }
    (outdir / "iter6_findings.json").write_text(json.dumps(findings, indent=2, default=float))
    print("\nFindings:\n" + json.dumps(findings, indent=2, default=float))


if __name__ == "__main__":
    print("=" * 92)
    print(
        f"Iteration 6 (GA10) — production NVF4 fp4 _scaled_mm on {torch.cuda.get_device_name(0)} (TF32 off)"
    )
    print("=" * 92)
    run()
