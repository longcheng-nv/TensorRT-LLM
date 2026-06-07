---
name: silicon-precision-oracle
description: >
  Drive REAL low-precision GPU kernels (fp8/fp4 _scaled_mm, production fused cute_dsl MoE kernels)
  as the silicon ground truth for the mptracer harness — measuring twin-fidelity, accuracy, and
  latency/SOL. Use when validating a twin against silicon or measuring whether a precision demotion
  actually speeds up on the target GPU. Contains the version-fragile fp8/fp4 invocation recipes.
metadata:
  tool: mptracer
  requires: NVIDIA GPU (B200/sm_100 for fp4/MXFP); torch + torchao; falls back to host twin if absent
  stack_pinned: torch 2.11, torchao 0.15  # recipes are version-fragile — re-verify on upgrade
---

# Silicon precision oracle

The harness twin must be validated against real silicon, and latency must be measured, not modeled.
**TF32 OFF** for every reference (`torch.backends.cuda.matmul.allow_tf32 = False`) — a silent
fp32-in-TF32 reference invalidates fidelity.

## Twin ↔ silicon fidelity (do this right or it's meaningless)

`twin_fidelity = ||Y_twin - Y_silicon|| / ||Y_silicon||`. To get the true accumulation floor (~1e-3
for fp8/fp4 MMA), the twin must (1) use the **same scaling recipe** as silicon (per-tensor vs
per-block), and (2) **reuse the exact quantized operands** silicon consumed — dequantize *those*,
accumulate in fp64. A software fp8/fp4 emulation differs from hardware rounding element-wise
(gives ~1e-2, not the real floor).

## fp8 per-tensor `_scaled_mm`

```python
sa = a.abs().max()/448; A8 = (a/sa).clamp(-448,448).to(torch.float8_e4m3fn)
D = torch._scaled_mm(A8, B8.t(), scale_a=sa.reshape(()), scale_b=sb.reshape(()), out_dtype=torch.bfloat16)
```

## NVFP4 fp4 `_scaled_mm` (torchao) — recipe

`nvfp4_quantize(X, 16)` returns **(block_scale_e4m3 [M,K/16], packed_fp4_uint8 [M,K/2])** —
scale first, data second. View the packed data as `torch.float4_e2m1fn_x2`, swizzle scales with
`torchao...to_blocked`, transpose the column operand:
`torch._scaled_mm(a_fp4, b_fp4.t(), scale_a=to_blocked(sa), scale_b=to_blocked(sb), out_dtype=bf16)`.

## Latency: measure GEMM-only AND the quant separately

The activation-quant pass (multi-pass amax/scale/cast/pack over the full tensor in HBM), not the
GEMM, is the decode bottleneck. Report **two** numbers: GEMM-only speedup (pre-quantize both
operands, time `_scaled_mm` alone) and the eager activation-quant cost. **A single quant+GEMM number
misattributes the cost.**

## The decode-fusion gate (decisive, measured)

For low-precision MoE FC1 at decode (small batch): eager separate quant → ~0.05× of bf16;
framework fusion (CUDA-graph / `torch.compile`) → ~0.27× (HBM traffic of the quant survives); the
**kernel-epilogue-fused** production kernel (`torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell_multi_b`,
which emits fp4 activations directly) → **1.8–1.9×**. ⇒ a precision demotion wins at decode **iff**
activation quantization is fused into the producing kernel. Gate decode proposals on this, not on
the GEMM-kernel speedup alone. (Input-prep recipe: mirror
`tests/.../test_cute_dsl_moe.py::test_nvfp4_gather_grouped_gemm_act_fusion_blackwell` — moe_sort maps,
fp4_quantize + view, interleave_linear_and_gate for gated weights/scales, single-B-as-list.)

## No GPU?

Fall back to the host twin + fake-quant (accuracy/attribution still valid); tag latency/SOL as
`silicon-pending` rather than reporting a modeled number.
