# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Standalone validation of the pick_config + launch addition (PR #16457):

A. cs=8 exactness gate — R0 (default) AND secant (enable_r0=False), both via
   GvrTopKKernel.launch(cluster_size=8 override): dtype x K x N x hint x BS,
   value-multiset vs torch.topk + fp32 R0==secant index-set.
B. launch() autoconfig — no overrides: assert pick_config picks the expected
   cluster regime (1/2/4/8) and output is a valid top-K in every regime.

Runs against the scratch gvrpkg (snapshot + MODIFIED kernel file staged at
/tmp/gvrval1/pickcfg). cr=1 convention: pre_idx stores true_index-1.
"""
import sys

sys.path.insert(0, "/tmp/gvrval1/pickcfg")
import torch  # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402

DEV = "cuda"
FAIL = 0


def make_pre(logits, K, hint, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_rows, N = logits.shape
    if hint == "hr1":
        pre = torch.topk(logits.float(), K, dim=1).indices.int()
    elif hint == "mid":
        noised = logits.float() + 0.8 * logits.float().std() * torch.randn(
            n_rows, N, generator=g, device=DEV)
        pre = torch.topk(noised, K, dim=1).indices.int()
    else:  # hr0
        pre = torch.randint(0, N, (n_rows, K), generator=g, device=DEV).int()
    return (pre - 1).clamp(min=0).contiguous()


def valid(out, logits, K):
    for r in (0, logits.shape[0] - 1):
        idx = out[r]
        if torch.unique(idx[idx >= 0]).numel() != K or int(idx.min()) < 0 \
                or int(idx.max()) >= logits.shape[1]:
            return False
        v = logits[r].float()
        kv = v.gather(0, idx.long()).sort().values
        rv = torch.topk(v, K).values.sort().values
        if not torch.equal(kv, rv):
            return False
    return True


def check(tag, ok):
    global FAIL
    if not ok:
        FAIL += 1
    print(f"{'OK ' if ok else 'FAIL'} {tag}", flush=True)


# ---- A. cs=8 exactness -----------------------------------------------------
print("== A. cs=8 exactness (launch override cluster_size=8, R0 + secant) ==",
      flush=True)
for dt in (torch.float32, torch.bfloat16, torch.float16):
    for K in (512, 1024, 2048):
        for N in (131072,) + ((262144,) if K == 1024 else ()):
            torch.manual_seed(0)
            for BS in (1, 4):
                logits = (torch.randn(BS, N, device=DEV) * 2.0).to(dt).contiguous()
                sl = torch.full((BS,), N, dtype=torch.int32, device=DEV)
                for hint in ("hr1", "mid", "hr0"):
                    pre = make_pre(logits, K, hint, seed=1)
                    o_r0 = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                    o_se = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                    GvrTopKKernel.launch(logits, pre, sl, o_r0, K, cluster_size=8)
                    GvrTopKKernel.launch(logits, pre, sl, o_se, K, cluster_size=8,
                                         enable_r0=False)
                    torch.cuda.synchronize()
                    ok = valid(o_r0, logits, K) and valid(o_se, logits, K)
                    if dt == torch.float32 and ok:
                        ok = torch.equal(o_r0.sort(dim=-1).values,
                                         o_se.sort(dim=-1).values)
                    check(f"cs8 {str(dt)[6:]:9s} K{K:<5d} N{N:<7d} BS{BS} {hint}", ok)

# ---- B. launch autoconfig regimes ------------------------------------------
print("== B. launch() autoconfig (pick_config regimes) ==", flush=True)
CASES = [  # (dtype, K, N, BS, expected_cs)
    (torch.float32, 2048, 32768, 1, 1),
    (torch.bfloat16, 512, 65536, 16, 4),
    (torch.float32, 1024, 131072, 2, 8),
    (torch.float32, 512, 65536, 64, 2),
    (torch.bfloat16, 1024, 65536, 256, 1),
    (torch.bfloat16, 512, 16384, 1024, 1),
]
for dt, K, N, BS, exp_cs in CASES:
    cfg = GvrTopKKernel.pick_config(dt, BS, N)
    torch.manual_seed(0)
    logits = (torch.randn(BS, N, device=DEV) * 2.0).to(dt).contiguous()
    sl = torch.full((BS,), N, dtype=torch.int32, device=DEV)
    pre = make_pre(logits, K, "mid", seed=1)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    GvrTopKKernel.launch(logits, pre, sl, out, K)
    torch.cuda.synchronize()
    ok = cfg["cluster_size"] == exp_cs and valid(out, logits, K)
    check(f"auto {str(dt)[6:]:9s} K{K:<5d} N{N:<7d} BS{BS:<5d} "
          f"cs={cfg['cluster_size']} (exp {exp_cs}) T{cfg['num_threads']} "
          f"mb{cfg['min_blocks_per_mp']}", ok)

print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}", flush=True)
sys.exit(1 if FAIL else 0)
