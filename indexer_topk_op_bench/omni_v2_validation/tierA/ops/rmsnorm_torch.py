"""Tier-A incumbent baseline: eager torch RMSNorm (multi-kernel composed path).
Same contract + same shapes/seed as rmsnorm_triton.py so A/B is paired."""
import torch

TOKENS, HIDDEN, EPS = 16384, 7168, 1e-6
_SEED = (TOKENS * 31 + HIDDEN) & 0x7FFFFFFF


def kernel_fn(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).to(x.dtype)


reference_fn = kernel_fn


def get_inputs():
    g = torch.Generator(device="cuda").manual_seed(_SEED)
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.float32,
                    generator=g).to(torch.bfloat16)
    w = torch.randn(HIDDEN, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    return [x, w]
