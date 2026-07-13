"""Tier-A NEGATIVE control: subtly wrong RMSNorm (+5% scale error).
verify_exact --mode dense --dtype bf16 MUST FAIL on this module.
A gate that cannot fail is not a gate."""
import torch

from rmsnorm_triton import get_inputs, get_adversarial_inputs, reference_fn, EPS  # noqa: F401


def kernel_fn(x, w):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    return (y * 1.05).to(x.dtype)  # 5% error > bf16 tol 1e-2
