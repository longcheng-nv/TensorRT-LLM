# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""[op36-A2] distP4 exactness battery.

Builds the gvrpkg36 kernel via the PRODUCTION launch contract
(GvrTopKKernel.launch -> pick_config; mirrors ops_op36._build_a0 shape but
with NO A0 flags — dist_p4=True only). For every cell it runs BOTH the
dist_p4=True kernel and the unmodified baseline (dist_p4=False, same package,
same forced launch shape) on the same inputs and checks each against
torch.topk value-multiset exactness. cr=1 (v32 configuration), fp32 only.
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from gvrpkg36.top_k.gvr_topk_decode import GvrTopKKernel as Gvr36  # noqa: E402

DEV = "cuda"
CR = 1  # v32 configuration: N == seqlen


def run_kernel(logits, pre, K, dist, cs_force=None):
    BS, N = logits.shape
    sl = torch.full((BS,), N * CR, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ovr = {}
    if dist:
        ovr["dist_p4"] = True
    if cs_force is not None:
        ovr["cluster_size"] = cs_force
    Gvr36.launch(logits, pre, sl, out, K, compress_ratio=CR, **ovr)
    torch.cuda.synchronize()
    return out


def check_exact(logits, out, K):
    """Value-multiset exactness vs torch.topk + index validity/uniqueness."""
    BS, N = logits.shape
    ref = torch.sort(torch.topk(logits.float(), K, dim=1).values,
                     dim=1, descending=True).values
    for b in range(BS):
        idx = out[b].long()
        if (idx < 0).any() or (idx >= N).any():
            return False, f"row{b}: invalid index (pad/-1 or OOB)"
        if idx.unique().numel() != K:
            return False, f"row{b}: duplicate indices"
        got = torch.sort(logits[b, idx].float(), descending=True).values
        if not torch.equal(got, ref[b]):
            nbad = (got != ref[b]).sum().item()
            return False, f"row{b}: value multiset mismatch ({nbad} slots)"
    return True, ""


def make_inputs(K, N, BS, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    logits = torch.randn(BS, N, dtype=torch.float32, device=DEV, generator=g)
    # preIdx: hit ~ 0.5 (loose calibration per spec)
    noisy = logits + 0.8 * logits.std() * torch.randn(
        BS, N, dtype=torch.float32, device=DEV, generator=g)
    pre = torch.topk(noisy, K, dim=1).indices.int().contiguous()
    return logits, pre


def main():
    torch.cuda.init()
    npass = 0
    ntot = 0
    results = []

    grid_K = (512, 1024, 2048)
    grid_N = (65536, 131072, 262144)
    grid_BS = (2, 8, 64)
    seed = 20260718

    for K in grid_K:
        for N in grid_N:
            for BS in grid_BS:
                seed += 1
                logits, pre = make_inputs(K, N, BS, seed)
                cfg = Gvr36.pick_config(torch.float32, BS, N)
                cs = cfg["cluster_size"]
                cs_force = None
                if cs == 1:
                    cs_force = 4  # force the cluster path so dist_p4 is live
                    cs = 4
                out_base = run_kernel(logits, pre, K, dist=False, cs_force=cs_force)
                ok_b, why_b = check_exact(logits, out_base, K)
                out_dist = run_kernel(logits, pre, K, dist=True, cs_force=cs_force)
                ok_d, why_d = check_exact(logits, out_dist, K)
                ntot += 1
                npass += int(ok_b and ok_d)
                tag = (f"K={K} N={N} BS={BS} cs={cs}"
                       f"{'(forced)' if cs_force else ''} T={cfg['num_threads']}")
                line = (f"[{'PASS' if ok_b and ok_d else 'FAIL'}] {tag} | "
                        f"base={'OK' if ok_b else 'FAIL:' + why_b} "
                        f"dist={'OK' if ok_d else 'FAIL:' + why_d}")
                print(line, flush=True)
                results.append(line)

    # ---- adversarial (i): ambiguous exact-tail (massive boundary tie) ----
    K, N, BS = 2048, 131072, 2
    logits = torch.full((BS, N), -1e30, dtype=torch.float32, device=DEV)
    logits[:, : K + 500] = 1.0
    g = torch.Generator(device=DEV).manual_seed(7)
    noisy = logits + 0.8 * logits.std() * torch.randn(
        BS, N, dtype=torch.float32, device=DEV, generator=g)
    pre = torch.topk(noisy, K, dim=1).indices.int().contiguous()
    cfg = Gvr36.pick_config(torch.float32, BS, N)
    cs = cfg["cluster_size"]
    cs_force = 4 if cs == 1 else None
    out_dist = run_kernel(logits, pre, K, dist=True, cs_force=cs_force)
    ok_d, why_d = check_exact(logits, out_dist, K)
    ntot += 1
    npass += int(ok_d)
    print(f"[{'PASS' if ok_d else 'FAIL'}] ADV-i ambiguous-tie K={K} N={N} "
          f"BS={BS} cs={cs if cs_force is None else cs_force} | "
          f"dist={'OK' if ok_d else 'FAIL:' + why_d}", flush=True)

    # ---- adversarial (ii): all-distinct near-boundary (shuffled linspace) ----
    K, N, BS = 2048, 131072, 2
    g = torch.Generator(device=DEV).manual_seed(11)
    row = torch.linspace(0.0, 1.0, N, dtype=torch.float32, device=DEV)
    logits = torch.stack(
        [row[torch.randperm(N, generator=g, device=DEV)] for _ in range(BS)])
    logits = logits.contiguous()
    noisy = logits + 0.8 * logits.std() * torch.randn(
        BS, N, dtype=torch.float32, device=DEV, generator=g)
    pre = torch.topk(noisy, K, dim=1).indices.int().contiguous()
    out_dist = run_kernel(logits, pre, K, dist=True, cs_force=cs_force)
    ok_d, why_d = check_exact(logits, out_dist, K)
    ntot += 1
    npass += int(ok_d)
    print(f"[{'PASS' if ok_d else 'FAIL'}] ADV-ii distinct-linspace K={K} "
          f"N={N} BS={BS} cs={cs if cs_force is None else cs_force} | "
          f"dist={'OK' if ok_d else 'FAIL:' + why_d}", flush=True)

    print(f"BATTERY_A2: {npass}/{ntot} PASS", flush=True)
    return 0 if npass == ntot else 1


if __name__ == "__main__":
    sys.exit(main())
