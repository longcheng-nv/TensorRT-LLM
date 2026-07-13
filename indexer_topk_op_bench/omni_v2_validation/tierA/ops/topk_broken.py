"""Tier-A NEGATIVE control (selection): returns K indices but duplicates one.
verify_exact --mode select MUST FAIL (duplicate-index and/or multiset check)."""
import torch

from topk_tie import get_inputs, reference_fn, K  # noqa: F401


def kernel_fn(x):
    idx = torch.topk(x, K, sorted=False).indices.to(torch.int64)
    idx[-1] = idx[0]  # duplicate index — classic atomic-collect escape bug
    return idx
