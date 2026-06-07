#!/usr/bin/env python3
"""Iteration 8 (GA11'): drive the REAL fused act-fusion cute_dsl kernel on B200.

iter7 showed framework fusion (CUDA-graph) only recovered decode fp4 to 0.27× of bf16
because the activation-quant's HBM traffic survives. This drives the production
`cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell_multi_b` kernel, which does
gather + FC1 GEMM + SwiGLU + fp4-output-quant in ONE launch — so the FC1 activation
emerges already fp4 with NO separate quant pass (true kernel-epilogue fusion).

Compares the fused fp4 FC1 latency to a bf16 grouped-FC1 baseline across DSV4 MoE
decode/prefill token counts. Falsifiable: fused fp4 ≤ bf16 at decode (BS≤8) → kernel
fusion delivers the decode win framework fusion could not. Run on a FREE B200.

Input prep mirrors tests/unittest/_torch/thop/parallel/test_cute_dsl_moe.py
(test_nvfp4_gather_grouped_gemm_act_fusion_blackwell).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tensorrt_llm._torch.modules.fused_moe.quantization import interleave_linear_and_gate
from tensorrt_llm._torch.utils import ActivationType, swizzle_sf, unswizzle_sf

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
SF = 16  # NVF4 scaling vector size
HIDDEN = 4096  # DSV4
INTERMEDIATE = 2048  # DSV4 moe_intermediate
NUM_EXPERTS = 256
EP = 32
NUM_LOCAL = NUM_EXPERTS // EP  # 8
TOPK = 6
TILE = 128
ACT = ActivationType.Swiglu
SWIGLU_LIMIT = 10.0
GATED_MULT = 2  # gate+up


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


def build(num_tokens, seed=42):
    g = torch.Generator(device=DEV).manual_seed(seed)
    routing = torch.randn(num_tokens, NUM_EXPERTS, device=DEV, generator=g)
    scales, experts = routing.topk(TOPK, dim=-1)
    experts = experts.to(torch.int32)
    scales = scales.softmax(dim=-1).to(torch.float32)
    experts[0] = 0
    (
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        expanded_to_perm,
        perm_to_expanded,
        total_padded,
        num_non_exiting,
    ) = torch.ops.trtllm.moe_sort(
        token_selected_experts=experts,
        token_final_scales=scales,
        num_experts=NUM_EXPERTS,
        top_k=TOPK,
        local_expert_offset=0,
        local_num_experts=NUM_LOCAL,
        tile_tokens_dim=TILE,
    )

    a = torch.randint(-5, 5, (num_tokens, HIDDEN), dtype=torch.int32, device=DEV).to(torch.bfloat16)
    Nw = INTERMEDIATE * GATED_MULT
    b = torch.randint(-5, 5, (NUM_LOCAL, Nw, HIDDEN), dtype=torch.int32, device=DEV).to(
        torch.bfloat16
    )

    a_gsf = a.abs().max().float() / (448 * 6)
    b_gsf = b.abs().amax(dim=(1, 2)).float() / (448 * 6)
    a_q, a_sf = torch.ops.trtllm.fp4_quantize(a, 1 / a_gsf, SF, False)
    a_q = a_q.view(torch.float4_e2m1fn_x2)
    a_sf_unsw = unswizzle_sf(a_sf, (num_tokens + 127) // 128 * 128, HIDDEN)[:num_tokens]
    b_q, b_sf = torch.ops.trtllm.fp4_quantize(b, 1 / b_gsf, SF, False)
    b_q = b_q.view(torch.float4_e2m1fn_x2)
    b_sf = b_sf.view(NUM_LOCAL, Nw, HIDDEN // SF)
    alpha = a_gsf * b_gsf

    # interleave gate/up weights (granularity 64) for the gated fused kernel
    b_kernel = interleave_linear_and_gate(b_q.view(torch.uint8), group_size=64, dim=1).view(
        torch.float4_e2m1fn_x2
    )
    b_sf_unsw = unswizzle_sf(b_sf, Nw, HIDDEN).view(NUM_LOCAL, Nw, HIDDEN // SF)
    b_sf_il = interleave_linear_and_gate(b_sf_unsw, group_size=64, dim=1)
    b_sf_kernel = swizzle_sf(b_sf_il, Nw, HIDDEN).view(NUM_LOCAL, Nw, HIDDEN // SF)

    global_sf = torch.tensor(
        [1.0 / (1.0 / (448 * 6))], dtype=torch.float32, device=DEV
    )  # output fp4 global scale
    return dict(
        a_q=a_q,
        a_sf_unsw=a_sf_unsw,
        b_kernel=b_kernel,
        b_sf_kernel=b_sf_kernel,
        alpha=alpha,
        tile_idx_to_group_idx=tile_idx_to_group_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        perm_to_expanded=perm_to_expanded,
        num_non_exiting=num_non_exiting,
        global_sf=global_sf,
        a_bf16=a,
        b_bf16=b,
        num_valid=int(total_padded.item()),
    )


def fused_call(d):
    return torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell_multi_b(
        d["a_q"],
        [d["b_kernel"]],
        d["a_sf_unsw"],
        [d["b_sf_kernel"]],
        [d["alpha"]],
        d["tile_idx_to_group_idx"],
        d["tile_idx_to_mn_limit"],
        d["perm_to_expanded"],
        d["num_non_exiting"],
        d["global_sf"],
        num_experts=NUM_EXPERTS,
        top_k=TOPK,
        num_local_experts=NUM_LOCAL,
        local_expert_offset=0,
        tile_size=TILE,
        scaling_vector_size=SF,
        activation_type=ACT,
        swiglu_limit_scalar=SWIGLU_LIMIT,
    )


def bf16_baseline(d):
    """Equivalent FC1 in bf16 (gather + per-expert mm + swiglu).

    Coarse but representative of the unfused bf16 grouped-FC1 work at this scale.
    """
    a, b = d["a_bf16"], d["b_bf16"]  # a [T,H], b [L,2I,H]
    # one representative expert GEMM per local expert on the replicated tokens
    P = a.shape[0] * TOPK
    pe = max(P // NUM_LOCAL, 1)
    out = []
    for e in range(NUM_LOCAL):
        toks = a[:pe] if pe <= a.shape[0] else a.repeat((pe + a.shape[0] - 1) // a.shape[0], 1)[:pe]
        z = torch.mm(toks, b[e].t())  # [pe, 2I]
        gate, up = z[:, :INTERMEDIATE], z[:, INTERMEDIATE:]
        out.append(up * torch.nn.functional.silu(gate))
    return out


def run():
    rows = []
    for bs in [1, 8, 32, 128, 512]:
        T = bs
        try:
            d = build(T)
        except Exception as ex:
            print(f"  [build fail] bs{bs}: {str(ex)[:140]}")
            continue
        try:
            _ = fused_call(d)
            torch.cuda.synchronize()
        except Exception as ex:
            rows.append(dict(bs=bs, error=str(ex)[:160]))
            print(f"  [fused fail] bs{bs}: {str(ex)[:140]}")
            continue
        t_fused = _bench(lambda: fused_call(d))
        t_bf16 = _bench(lambda: bf16_baseline(d))
        regime = "launch" if T * TOPK < 16 else ("memory" if T * TOPK < 128 else "compute")
        rows.append(
            dict(
                bs=bs,
                P=T * TOPK,
                t_fused_fp4_us=t_fused * 1e3,
                t_bf16_us=t_bf16 * 1e3,
                speedup_vs_bf16=t_bf16 / t_fused,
                regime=regime,
            )
        )
        print(
            f"  bs{bs:<4} P={T * TOPK:<5} fused_fp4={t_fused * 1e3:8.1f}us bf16={t_bf16 * 1e3:8.1f}us "
            f"speedup={t_bf16 / t_fused:.2f}x ({regime})"
        )

    outdir = Path(__file__).resolve().parent / "results"
    outdir.mkdir(exist_ok=True)
    ok = [r for r in rows if "error" not in r]
    decode = [r for r in ok if r["bs"] <= 8]
    findings = {
        "device": torch.cuda.get_device_name(0),
        "fused_kernel_ran": len(ok) > 0,
        "speedup_vs_bf16_by_bs": {r["bs"]: round(r["speedup_vs_bf16"], 3) for r in ok},
        "decode_fused_ge_1x": all(r["speedup_vs_bf16"] >= 1.0 for r in decode) if decode else None,
        "iter7_framework_decode_was": 0.27,
    }
    (outdir / "iter8_findings.json").write_text(json.dumps(findings, indent=2, default=float))
    print("\nFindings:\n" + json.dumps(findings, indent=2, default=float))


if __name__ == "__main__":
    print("=" * 90)
    print(
        f"Iteration 8 (GA11') — REAL fused act-fusion cute_dsl kernel on {torch.cuda.get_device_name(0)}"
    )
    print("=" * 90)
    run()
