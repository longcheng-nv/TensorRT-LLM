"""Tier-A selection-class positive case: top-K over a tie-rich row.

kernel_fn (torch.topk, sorted=False) and reference_fn (argsort) legitimately
pick DIFFERENT indices at the tie boundary -> index equality would FAIL, but
the tie-aware value-multiset criterion of verify_exact --mode select must PASS.
This validates exactly the criterion the v2 skill mandates for selection kernels.
"""
import torch

N, K = 262144, 2048
_SEED = (N * 31 + K) & 0x7FFFFFFF


def kernel_fn(x):
    return torch.topk(x, K, sorted=False).indices.to(torch.int64)


def reference_fn(x):
    return torch.argsort(x, descending=True, stable=True)[:K]


def get_inputs():
    g = torch.Generator(device="cuda").manual_seed(_SEED)
    # heavy ties: only 64 distinct levels over 262144 elements
    x = torch.randint(0, 64, (N,), device="cuda", generator=g).float()
    return [x]


def get_adversarial_inputs():
    cases = []
    # near-tie cluster: K-boundary values 1 ULP apart
    x = torch.zeros(N, device="cuda")
    x[:K * 2] = 1.0
    x[K // 2:K * 2] = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).cuda()
    cases.append([x])
    # all-equal row (maximal tie storm)
    cases.append([torch.ones(N, device="cuda")])
    return cases
