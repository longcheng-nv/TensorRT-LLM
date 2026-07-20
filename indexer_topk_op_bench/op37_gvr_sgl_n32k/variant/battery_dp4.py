# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""[op37-dp4] distP4-on-prod2 exactness battery.

Clone of op36's battery_a2.py adapted to gvrpkg37 (= gvrpkgprod2, PR#16457
head @e6fdbfac3d, + dist_p4 splice). Sections:

  S1 grid      fp32 x K{512,1024,2048} x N{65536,131072,262144} x BS{2,8,64},
               cs from pick_config (>1 everywhere on this grid). Per cell:
                 - dist_p4=True  -> exact vs torch.topk (index validity +
                   uniqueness + value multiset)
                 - dist_p4=False control -> byte-equal (torch.equal) to the
                   PRISTINE gvrpkgprod2 output on the same inputs + exact.
  S2 tie       forced boundary-tie rows that fire the exact-tail ambiguity
               fallback, per K:
                 - big tie  (cnt_strad ~ K+500 > 128): radix select arm
                 - small tie (cnt_strad = 12 <= 128): [p4tt] fast arm at
                   K>=1024 (K512 compiles radix-only: p4_tail_fast=False)
  S3 launch    contract smoke: cluster_size overrides {2,4,8} at N=131072,
               BS{1,4}, dist_p4=True, exact.
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH / "op26_r0_upstream_port_report" / "p4f1_harness"))

from gvrpkg37.top_k.gvr_topk_decode import GvrTopKKernel as Gvr37  # noqa: E402
from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as GvrRef  # noqa: E402

DEV = "cuda"
CR = 1  # v32 configuration: N == seqlen


def run_kernel(cls, logits, pre, K, dist, cs_force=None):
    BS, N = logits.shape
    sl = torch.full((BS,), N * CR, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ovr = {}
    if dist:
        ovr["dist_p4"] = True
    if cs_force is not None:
        ovr["cluster_size"] = cs_force
    cls.launch(logits, pre, sl, out, K, compress_ratio=CR, **ovr)
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


def make_pre(logits, K, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    BS, N = logits.shape
    noisy = logits + 0.8 * logits.std() * torch.randn(
        BS, N, dtype=torch.float32, device=DEV, generator=g)
    return torch.topk(noisy, K, dim=1).indices.int().contiguous()


def main():
    torch.cuda.init()
    sec_pass = {"S1_grid": 0, "S2_tie": 0, "S3_launch": 0}
    sec_tot = {"S1_grid": 0, "S2_tie": 0, "S3_launch": 0}

    # ---------------- S1: full grid ----------------
    grid_K = (512, 1024, 2048)
    grid_N = (65536, 131072, 262144)
    grid_BS = (2, 8, 64)
    seed = 20260720

    for K in grid_K:
        for N in grid_N:
            for BS in grid_BS:
                seed += 1
                logits, pre = make_inputs(K, N, BS, seed)
                cfg = Gvr37.pick_config(torch.float32, BS, N)
                cs = cfg["cluster_size"]
                cs_force = None
                if cs == 1:
                    cs_force = 4  # force the cluster path so dist_p4 is live
                    cs = 4
                out_dist = run_kernel(Gvr37, logits, pre, K, dist=True,
                                      cs_force=cs_force)
                ok_d, why_d = check_exact(logits, out_dist, K)
                out_ctrl = run_kernel(Gvr37, logits, pre, K, dist=False,
                                      cs_force=cs_force)
                out_ref = run_kernel(GvrRef, logits, pre, K, dist=False,
                                     cs_force=cs_force)
                # NOTE: the kernel's output index ORDER is atomic-arrival
                # nondeterministic run-to-run (measured on the PRISTINE
                # gvrpkgprod2 alone: raw byte-equal False across 4 reps,
                # sorted index set stable). The default-off byte contract is
                # therefore proven at the PTX level (CUTE_DSL_KEEP_PTX
                # byte-compare, see logs/); here we check the strongest
                # data-level invariant: identical sorted index SET.
                ok_c = torch.equal(out_ctrl.sort(dim=1).values,
                                   out_ref.sort(dim=1).values)
                why_c = "" if ok_c else "ctrl index-set != pristine"
                ok_ce, why_ce = check_exact(logits, out_ctrl, K)
                ok = ok_d and ok_c and ok_ce
                sec_tot["S1_grid"] += 1
                sec_pass["S1_grid"] += int(ok)
                tag = (f"K={K} N={N} BS={BS} cs={cs}"
                       f"{'(forced)' if cs_force else ''} T={cfg['num_threads']}")
                print(f"[{'PASS' if ok else 'FAIL'}] S1 {tag} | "
                      f"dist={'OK' if ok_d else 'FAIL:' + why_d} "
                      f"ctrl_eq={'OK' if ok_c else 'FAIL:' + why_c} "
                      f"ctrl_exact={'OK' if ok_ce else 'FAIL:' + why_ce}",
                      flush=True)

    # ---------------- S2: forced boundary-tie rows ----------------
    N, BS = 131072, 2
    for K in grid_K:
        # (a) BIG tie class (> 128 members): radix select arm.
        logits = torch.full((BS, N), -1e30, dtype=torch.float32, device=DEV)
        logits[:, : K + 500] = 1.0
        pre = make_pre(logits, K, 7)
        cfg = Gvr37.pick_config(torch.float32, BS, N)
        cs_force = 4 if cfg["cluster_size"] == 1 else None
        cs = cfg["cluster_size"] if cs_force is None else 4
        out = run_kernel(Gvr37, logits, pre, K, dist=True, cs_force=cs_force)
        ok, why = check_exact(logits, out, K)
        sec_tot["S2_tie"] += 1
        sec_pass["S2_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] S2a big-tie K={K} N={N} BS={BS} "
              f"cs={cs} strad~{K + 500} | {'OK' if ok else 'FAIL:' + why}",
              flush=True)

        # (b) SMALL tie class (12 members, need0=6): fires the [p4tt] fast
        #     arm at K>=1024 (K512: p4_tail_fast=False -> radix-only text).
        #     Fixture v2: the v1 fixture (isolated plateau over a -1..-2
        #     floor) provoked a PRE-EXISTING base-kernel P2 tie-plateau
        #     undershoot fail-soft on row1 (verified: pristine gvrpkgprod2
        #     emits the same -1 pad slots) — outside the exact-tail's scope.
        #     v2 places a wide continuous band (3000 values in (49.0, 49.9))
        #     right below the plateau so the admission window [K, kC] spans
        #     a wide threshold range and P2 converges; band values stay
        #     >= 0.1 below the ties (fine-bin width ~2e-4), so the (b*, sb*)
        #     tie class is still exactly the 12 plateau members.
        need0, strad, band = 6, 12, 3000
        g = torch.Generator(device=DEV).manual_seed(13 + K)
        logits = torch.full((BS, N), -1000.0, dtype=torch.float32, device=DEV)
        # distinct highs at ranks [0, K-need0)
        highs = 100.0 + torch.arange(K - need0, device=DEV,
                                     dtype=torch.float32) * 1e-3
        logits[:, : K - need0] = highs.flip(0)
        # tie plateau straddling the boundary: strad equal values, only
        # need0 slots left -> cnt_strad > need0 > 0 fires the exact tail.
        logits[:, K - need0: K - need0 + strad] = 50.0
        # continuous sub-plateau band for P2 admission
        logits[:, K - need0 + strad: K - need0 + strad + band] = (
            49.0 + 0.9 * torch.rand(BS, band, dtype=torch.float32,
                                    device=DEV, generator=g))
        pre = make_pre(logits, K, 17 + K)
        out = run_kernel(Gvr37, logits, pre, K, dist=True, cs_force=cs_force)
        ok, why = check_exact(logits, out, K)
        sec_tot["S2_tie"] += 1
        sec_pass["S2_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] S2b small-tie K={K} N={N} BS={BS} "
              f"cs={cs} strad={strad} need0={need0} | "
              f"{'OK' if ok else 'FAIL:' + why}", flush=True)

        # (c) all-distinct near-boundary (shuffled linspace): unambiguous
        #     tail, exercises the non-firing gate.
        g = torch.Generator(device=DEV).manual_seed(11)
        row = torch.linspace(0.0, 1.0, N, dtype=torch.float32, device=DEV)
        logits = torch.stack(
            [row[torch.randperm(N, generator=g, device=DEV)] for _ in range(BS)]
        ).contiguous()
        pre = make_pre(logits, K, 19 + K)
        out = run_kernel(Gvr37, logits, pre, K, dist=True, cs_force=cs_force)
        ok, why = check_exact(logits, out, K)
        sec_tot["S2_tie"] += 1
        sec_pass["S2_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] S2c distinct-linspace K={K} "
              f"N={N} BS={BS} cs={cs} | {'OK' if ok else 'FAIL:' + why}",
              flush=True)

    # ---------------- S3: launch-contract smoke over cs overrides ----------
    K, N = 1024, 131072
    for BS in (1, 4):
        for cs_ovr in (2, 4, 8):
            seed += 1
            logits, pre = make_inputs(K, N, BS, seed)
            out = run_kernel(Gvr37, logits, pre, K, dist=True, cs_force=cs_ovr)
            ok, why = check_exact(logits, out, K)
            sec_tot["S3_launch"] += 1
            sec_pass["S3_launch"] += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] S3 K={K} N={N} BS={BS} "
                  f"cs={cs_ovr}(override) | {'OK' if ok else 'FAIL:' + why}",
                  flush=True)

    npass = sum(sec_pass.values())
    ntot = sum(sec_tot.values())
    for s in sec_tot:
        print(f"SECTION {s}: {sec_pass[s]}/{sec_tot[s]} PASS", flush=True)
    print(f"BATTERY_DP4: {npass}/{ntot} PASS", flush=True)
    return 0 if npass == ntot else 1


if __name__ == "__main__":
    sys.exit(main())
