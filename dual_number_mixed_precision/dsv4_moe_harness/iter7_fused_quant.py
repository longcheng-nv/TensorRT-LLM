#!/usr/bin/env python3
"""Iteration 7 (GA11): does fusing the activation-quant recover decode fp4?

iter6 showed the UNFUSED eager nvfp4_quantize collapses decode fp4 to 0.05-0.15× even
though the fp4 GEMM kernel alone is ~0.9× at decode. This tests whether fusing the
quant+_scaled_mm (CUDA-graph capture via torch.compile reduce-overhead, which removes the
per-op launch overhead that dominates small-M decode) recovers it toward the GEMM-only
ceiling. Falsifiable threshold: fused decode (BS≤8) fp4 ≥ 0.7× of bf16 (eager was ~0.05×).

Run on a FREE B200. TF32 off. Harness owns every number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
from torchao.prototype.mx_formats.utils import to_blocked

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
BLK = 16
DSV4_TOPK = 6
SHAPES = [
    ("FC1_bs1", 1 * DSV4_TOPK, 4096, 4096),
    ("FC1_bs8", 8 * DSV4_TOPK, 4096, 4096),
    ("FC1_bs512", 512 * DSV4_TOPK, 4096, 4096),
]


def _bench(fn, iters=50, warmup=20):
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


def _operands(X):
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
    for name, M, K, N in SHAPES:
        A = torch.randn(M, K, device=DEV, dtype=torch.float32, generator=g) * 0.5
        W = torch.randn(N, K, device=DEV, dtype=torch.float32, generator=g) * 0.5
        A_bf16, W_bf16 = A.to(torch.bfloat16), W.to(torch.bfloat16)
        Wd, Ws = _operands(W)  # weight pre-quantized (static, as in production)

        def mm_bf16():
            return torch.mm(A_bf16, W_bf16.t())

        def fp4_eager():  # activation quant + gemm, eager (iter6's path)
            Ad, As = _operands(A)
            return torch._scaled_mm(Ad, Wd.t(), scale_a=As, scale_b=Ws, out_dtype=torch.bfloat16)

        # fused: compile the quant+gemm with cudagraphs (reduce-overhead)
        fp4_compiled = torch.compile(fp4_eager, mode="reduce-overhead", fullgraph=False)
        try:
            _ = fp4_compiled()  # trigger compile
            torch.cuda.synchronize()
            compiled_ok = True
        except Exception as ex:
            compiled_ok = False
            comp_err = str(ex)[:120]

        t_bf16 = _bench(mm_bf16)
        t_eager = _bench(fp4_eager)
        if compiled_ok:
            t_comp = _bench(fp4_compiled)
            sp_comp = t_bf16 / t_comp
        else:
            t_comp, sp_comp = None, None
        rows.append(
            dict(
                shape=name,
                M=M,
                t_bf16_us=t_bf16 * 1e3,
                t_eager_us=t_eager * 1e3,
                t_compiled_us=(t_comp * 1e3 if t_comp else None),
                speedup_eager=t_bf16 / t_eager,
                speedup_compiled=sp_comp,
                compiled_ok=compiled_ok,
            )
        )
        msg = (
            f"  {name:<11} M={M:<5} bf16={t_bf16 * 1e3:7.1f}us eager={t_eager * 1e3:8.1f}us "
            f"({t_bf16 / t_eager:.3f}x)"
        )
        if compiled_ok:
            msg += f"  compiled={t_comp * 1e3:7.1f}us ({sp_comp:.3f}x)"
        else:
            msg += f"  compiled=FAILED ({comp_err})"
        print(msg)

    outdir = Path(__file__).resolve().parent / "results"
    outdir.mkdir(exist_ok=True)
    decode = [r for r in rows if r["M"] <= 48 and r["speedup_compiled"]]
    findings = {
        "device": torch.cuda.get_device_name(0),
        "decode_eager_speedup": {
            r["shape"]: round(r["speedup_eager"], 3) for r in rows if r["M"] <= 48
        },
        "decode_compiled_speedup": {
            r["shape"]: (round(r["speedup_compiled"], 3) if r["speedup_compiled"] else None)
            for r in rows
            if r["M"] <= 48
        },
        "large_bs512_compiled_speedup": next(
            (
                round(r["speedup_compiled"], 3)
                for r in rows
                if r["M"] > 1000 and r["speedup_compiled"]
            ),
            None,
        ),
        "threshold_decode_ge_0.7x_met": all(
            r["speedup_compiled"] and r["speedup_compiled"] >= 0.7 for r in decode
        )
        if decode
        else False,
    }
    (outdir / "iter7_findings.json").write_text(json.dumps(findings, indent=2, default=float))
    print("\nFindings:\n" + json.dumps(findings, indent=2, default=float))


if __name__ == "__main__":
    print("=" * 88)
    print(
        f"Iteration 7 (GA11) — fused (cudagraph) activation-quant + fp4 GEMM on {torch.cuda.get_device_name(0)}"
    )
    print("=" * 88)
    run()
