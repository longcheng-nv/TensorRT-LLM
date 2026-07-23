# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal repro + flag bisect for the plateau tie-class inexactness found by
gate40 on the vendored PR#16457 @e612 baseline (2026-07-23).

Failing: plateau_K512_N8192, plateau_K1024_N8192, plateau_K2048_N65536.
Bisect axes: p4_tail_fast, p4_warp_redundant, p2_warp_redundant, kc_diet,
enable_p4_rank_scatter_exact-only, enable_r0.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP40 = HERE.parent
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(HERE))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

from gvrpkg40b.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402

DEV = "cuda"
CASES = [(512, 4, 8192), (1024, 4, 8192), (2048, 1, 65536)]
FLAGSETS = [
    ("default", {}),
    ("no_p4tt", dict(p4_tail_fast=False)),
    ("no_p4wr", dict(p4_warp_redundant=False)),
    ("no_p2wr", dict(p2_warp_redundant=False)),
    ("no_kcdiet", dict(kc_diet=False)),
    ("no_r0", dict(enable_r0=False)),
    ("no_p4rse", dict(enable_p4_rank_scatter_exact=False)),
]


def make_case(K, N, seed_extra=0):
    g = torch.Generator(device=DEV).manual_seed(hash((K, N)) % (2**31) + seed_extra)
    lp = torch.randn(1, N, generator=g, device=DEV)
    m = torch.rand(1, N, generator=g, device=DEV) < 0.6
    lp[m] = 1.2345
    noisy = lp[0] + 0.5 * torch.randn(N, generator=g, device=DEV)
    pre = torch.topk(noisy, K).indices.to(torch.int32).reshape(1, K).contiguous()
    return lp, pre


def run(K, cr, N, flags, logits, pre):
    cfg = GvrTopKKernel.pick_config(torch.float32, 1, N)
    cfg["cluster_size"] = min(cfg["cluster_size"], 16)
    if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
        cfg["use_256bit_load"] = False
    kobj = GvrTopKKernel(dtype=cutlass.Float32, top_k=K, next_n=1,
                         compress_ratio=cr, return_output_values=False,
                         **cfg, **flags)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if cfg["use_256bit_load"] else 16
    in_f = crt.make_fake_compact_tensor(cutlass.Float32, (nr, nc),
                                        stride_order=(1, 0), assumed_align=ia)
    pi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb, K),
                                        stride_order=(1, 0), assumed_align=16)
    sl_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nr, K),
                                        stride_order=(1, 0), assumed_align=16)
    fs = crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    fn = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, stream=fs,
                      options="--enable-tvm-ffi")
    sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
    oi = torch.full((1, K), -7, dtype=torch.int32, device=DEV)
    fn(logits, pre, sl, None, oi, None)
    torch.cuda.synchronize()
    lg = logits[0].float()
    idx = oi.flatten().to(torch.int64)
    ref = torch.topk(lg, K).values
    ok_range = bool(idx.numel() == K and idx.min() >= 0 and idx.max() < N)
    uniq = int(torch.unique(idx).numel())
    if ok_range:
        got = lg[idx].sort(descending=True).values
        nmis = int((got != ref).sum())
        first = int((got != ref).float().argmax()) if nmis else -1
        return dict(cs=cfg["cluster_size"], uniq=uniq, nmis=nmis, first=first,
                    got_at=float(got[first]) if nmis else None,
                    ref_at=float(ref[first]) if nmis else None,
                    kth=float(ref[K - 1]))
    return dict(cs=cfg["cluster_size"], uniq=uniq, nmis=-1, first=-1,
                got_at=None, ref_at=None, kth=float(ref[K - 1]))


def main():
    for K, cr, N in CASES:
        logits, pre = make_case(K, N)
        tie = int((logits[0] == 1.2345).sum())
        print(f"\n== K={K} N={N} tie_class={tie} "
              f"(kth==1.2345: {float(torch.topk(logits[0], K).values[K-1]) == 1.2345})")
        for name, flags in FLAGSETS:
            r = run(K, cr, N, flags, logits, pre)
            tag = "OK " if r["nmis"] == 0 else "FAIL"
            print(f"  {name:10s} cs={r['cs']} {tag} nmis={r['nmis']:5d} "
                  f"first@{r['first']} got={r['got_at']} ref={r['ref_at']} "
                  f"uniq={r['uniq']}/{K}", flush=True)
        # determinism: 3 repeats of default
        rs = [run(K, cr, N, {}, logits, pre)["nmis"] for _ in range(3)]
        print(f"  default nmis over 3 runs: {rs}")


if __name__ == "__main__":
    main()
