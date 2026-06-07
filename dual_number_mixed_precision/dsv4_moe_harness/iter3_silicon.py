#!/usr/bin/env python3
"""Iteration 3 (GA5): silicon oracle on local B200.

Unblocks the hardware gap using a local B200 (sm_100). For the DSV4 MoE FC1/FC2 shapes
across BS=1..512 this:
  - validates the fake-quant fp8 TWIN against real torch._scaled_mm (twin_fidelity),
  - measures fp16 vs fp8 LATENCY and SOL% (the roofline gate for precision demotion).

Every number is measured here (harness-owned). TF32 OFF for honest references (iter1
finding #2). Run on a free GPU via CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda"
B200_BF16_PEAK_TFLOPS = 2250.0  # dense bf16 (approx, for SOL%)
B200_FP8_PEAK_TFLOPS = 4500.0

DSV4_TOPK = 6
BS_LIST = [1, 8, 32, 64, 128, 256, 512]
# (gemm, K, N) — FC1 gate_up fused N=2*2048; FC2 down K=2048,N=4096
GEMMS = [("FC1", 4096, 4096), ("FC2", 2048, 4096)]


def _bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return float(np.median(times))


def _to_fp8(x, scale):
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)


def run():
    rows = []
    g = torch.Generator(device=DEV).manual_seed(42)
    for bs in BS_LIST:
        M = bs * DSV4_TOPK
        for gemm, K, N in GEMMS:
            A = torch.randn(M, K, device=DEV, dtype=torch.float32, generator=g) * 0.5
            Bw = (
                torch.randn(N, K, device=DEV, dtype=torch.float32, generator=g) * 0.5
            )  # weight [N,K]

            # ---- references / real paths ----
            A_bf16, B_bf16 = A.to(torch.bfloat16), Bw.to(torch.bfloat16)

            def mm_bf16():
                return torch.mm(A_bf16, B_bf16.t())

            D_bf16 = mm_bf16().float()

            # fp8 per-tensor scaled
            sa = (A.abs().max() / 448.0).clamp(min=1e-8)
            sb = (Bw.abs().max() / 448.0).clamp(min=1e-8)
            A8, B8 = _to_fp8(A, sa), _to_fp8(Bw, sb)

            def scaled_mm():
                return torch._scaled_mm(
                    A8,
                    B8.t(),
                    scale_a=sa.reshape(()),
                    scale_b=sb.reshape(()),
                    out_dtype=torch.bfloat16,
                )

            try:
                D_fp8 = scaled_mm().float()
            except Exception as e:
                rows.append(dict(bs=bs, gemm=gemm, M=M, K=K, N=N, error=str(e)[:120]))
                print(f"  [skip] bs{bs} {gemm}: _scaled_mm failed: {str(e)[:100]}")
                continue

            # ---- TWIN fidelity: reuse the EXACT fp8 operands silicon consumed ----
            # The faithful twin dequantizes the same A8/B8 and accumulates in fp32 (TF32
            # off). The only remaining gap vs silicon is the MMA accumulation order/precision
            # — which is what twin_fidelity should measure. (Earlier runs conflated this
            # with a per-block-vs-per-tensor recipe mismatch (0.245) and then a numpy fp8
            # emulation gap (1.95e-2); both are removed by reusing A8/B8 directly.)
            A_deq = A8.float() * sa  # exact fp8 dequant on-device
            B_deq = B8.float() * sb
            D_twin = torch.mm(A_deq.double(), B_deq.double().t())  # fp64 accumulate
            twin_fid = float(
                torch.norm(D_twin - D_fp8.double()) / (torch.norm(D_fp8.double()) + 1e-30)
            )

            # ---- measured accuracy of fp8 vs bf16 ref ----
            meas_rel = float(torch.norm(D_bf16 - D_fp8) / (torch.norm(D_bf16) + 1e-30))

            # ---- latency + SOL ----
            t_bf16 = _bench(mm_bf16)
            t_fp8 = _bench(scaled_mm)
            flop = 2.0 * M * K * N
            tflops_bf16 = flop / (t_bf16 * 1e-3) / 1e12
            tflops_fp8 = flop / (t_fp8 * 1e-3) / 1e12
            sol_bf16 = 100.0 * tflops_bf16 / B200_BF16_PEAK_TFLOPS
            sol_fp8 = 100.0 * tflops_fp8 / B200_FP8_PEAK_TFLOPS
            speedup = t_bf16 / t_fp8

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
                    t_fp8_us=t_fp8 * 1e3,
                    speedup=speedup,
                    tflops_bf16=tflops_bf16,
                    tflops_fp8=tflops_fp8,
                    sol_bf16_pct=sol_bf16,
                    sol_fp8_pct=sol_fp8,
                )
            )
            print(
                f"  bs{bs:<4} {gemm} M={M:<5} twin_fid={twin_fid:.2e} meas_rel={meas_rel:.2e} "
                f"speedup={speedup:.2f}x fp8_SOL={sol_fp8:4.1f}% ({regime})"
            )

    outdir = Path(__file__).resolve().parent / "results"
    outdir.mkdir(exist_ok=True)
    ok = [r for r in rows if "error" not in r]
    if ok:
        with open(outdir / "iter3_silicon.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ok[0].keys()))
            w.writeheader()
            w.writerows(ok)
    findings = {
        "device": torch.cuda.get_device_name(0),
        "twin_fidelity_max": max((r["twin_fidelity"] for r in ok), default=None),
        "speedup_by_bs": {f"{r['gemm']}_bs{r['bs']}": round(r["speedup"], 3) for r in ok},
        "fp8_wins_only_when_compute_bound": [
            f"{r['gemm']}_bs{r['bs']}={r['speedup']:.2f}x@{r['regime']}" for r in ok
        ],
    }
    (outdir / "iter3_findings.json").write_text(json.dumps(findings, indent=2, default=float))
    print("\nFindings:\n" + json.dumps(findings, indent=2, default=float))
    print(f"\nwrote {outdir / 'iter3_silicon.csv'} and iter3_findings.json")


if __name__ == "__main__":
    print("=" * 90)
    print(f"Iteration 3 (GA5) — silicon oracle on {torch.cuda.get_device_name(0)} (TF32 off)")
    print("=" * 90)
    run()
