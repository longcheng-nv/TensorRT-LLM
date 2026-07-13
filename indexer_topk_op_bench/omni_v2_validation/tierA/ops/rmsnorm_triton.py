"""Tier-A candidate impl: Triton RMSNorm (dense class), omni-kernel v2 module contract.

Contract: kernel_fn(*args), reference_fn(*args), get_inputs(),
optional get_adversarial_inputs().  Shapes = DSv4-ish (hidden 7168, bf16);
tokens sized so the working set (~235 MB) exceeds B200 L2 -> cold/warm contrast
is meaningful for bench_cold.py validation.
"""
import torch
import triton
import triton.language as tl

TOKENS, HIDDEN, EPS = 16384, 7168, 1e-6
_SEED = (TOKENS * 31 + HIDDEN) & 0x7FFFFFFF  # seed policy: f(shape), not constant


@triton.jit
def _rmsnorm(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + row * stride + offs, y.to(Y.dtype.element_ty), mask=mask)


def kernel_fn(x, w):
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(x.shape[1])
    _rmsnorm[(x.shape[0],)](x, w, y, x.stride(0), x.shape[1], EPS,
                            BLOCK=BLOCK, num_warps=8)
    return y


def reference_fn(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(x.dtype)


def get_inputs():
    g = torch.Generator(device="cuda").manual_seed(_SEED)
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.float32,
                    generator=g).to(torch.bfloat16)
    w = torch.randn(HIDDEN, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    return [x, w]


def get_adversarial_inputs():
    cases = []
    # near-zero rows (rstd blowup territory)
    x = torch.full((8, HIDDEN), 1e-20, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
    cases.append([x, w])
    # large-magnitude rows (fp32 accumulate vs bf16 overflow)
    x = torch.full((8, HIDDEN), 224.0, device="cuda", dtype=torch.bfloat16)
    cases.append([x, w.clone()])
    # single-token row
    cases.append([torch.randn(1, HIDDEN, device="cuda").to(torch.bfloat16), w.clone()])
    return cases
