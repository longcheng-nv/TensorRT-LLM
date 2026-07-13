# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builder + wrapper for the op29 GVR-HBE experiment op (fork of sglang v2 +
hint-boundary-exact streaming path).

    gvr29_topk(scores[R,C] fp32, seq_lens[R] i32, K, pre_idx[R,K] i32,
               use_hbe=True) -> out[R,K] i32
"""
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src" / "gvr29"
_BUILD = _HERE.parents[1] / "_build"
_BUILD.mkdir(exist_ok=True)

os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
PAGE_BITS = 21

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        bdir = _BUILD / "gvr29"
        bdir.mkdir(parents=True, exist_ok=True)
        _MOD = load(
            name="gvr29_topk_hbe",
            sources=[str(_SRC / "gvr29_standalone.cu")],
            extra_include_paths=[str(_SRC), str(_SRC / "shim")],
            extra_cuda_cflags=[
                "-O3", "-std=c++20",
                "-gencode=arch=compute_100f,code=sm_100f",
                "-DSGL_CUDA_ARCH=1000",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O3", "-std=c++20", "-DSGL_CUDA_ARCH=1000"],
            build_directory=str(bdir),
            verbose=True,
        )
    return _MOD


_PAGE_TABLE = None


def _identity_page_table(device):
    global _PAGE_TABLE
    if _PAGE_TABLE is None or _PAGE_TABLE.device != torch.device(device):
        _PAGE_TABLE = torch.zeros(1, dtype=torch.int32, device=device)
    return _PAGE_TABLE


def plan(seq_lens, metadata=None, static_cluster_threshold=0):
    R = seq_lens.size(0)
    if metadata is None:
        metadata = torch.zeros((R + 1, 2), dtype=torch.int32,
                               device=seq_lens.device)
    _module().gvr29_plan(seq_lens, metadata, static_cluster_threshold)
    return metadata


_SPILL = {}


def _spill_buf(R, K, device):
    """Per-row global spill region: (spillA+spillB)=56*K TieValue (8 B)."""
    need = R * 56 * K * 8
    key = device
    buf = _SPILL.get(key)
    if buf is None or buf.numel() < need:
        _SPILL[key] = buf = torch.empty(need, dtype=torch.uint8, device=device)
    return buf


def gvr29_topk(scores, seq_lens, K, pre_idx, out=None, metadata=None,
               page_table=None, max_seq_len=None, use_hbe=True, spill=None):
    assert scores.dtype == torch.float32
    R = scores.size(0)
    if out is None:
        out = torch.empty((R, K), dtype=torch.int32, device=scores.device)
    if page_table is None:
        page_table = _identity_page_table(scores.device)
    if metadata is None:
        metadata = plan(seq_lens)
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item())
    if spill is None:
        spill = _spill_buf(R, K, scores.device)
    _module().gvr29_transform(scores, seq_lens, page_table, out, metadata,
                              K, PAGE_BITS, max_seq_len, pre_idx, use_hbe,
                              spill)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    _module()
    print("gvr29 built OK")
    # smoke: exactness with GOOD hints (true topk), BAD hints (random), and
    # ADVERSARIAL hints (bottom-K) — hint must never affect correctness.
    for K in (512, 1024, 2048):
        # BS=4 @32768 exercises HBE below the cluster floor; BS=520 (>512
        # disables the cluster path) exercises HBE at large N.
        for N, BS in ((32768, 4), (65536, 520), (262144, 520)):
            x = torch.randn(BS, N, dtype=torch.float32, device="cuda")
            sl = torch.full((BS,), N, dtype=torch.int32, device="cuda")
            ref = torch.topk(x, K, dim=1)
            good = ref.indices.to(torch.int32)
            bad = torch.randint(0, N, (BS, K), dtype=torch.int32,
                                device="cuda")
            adv = torch.topk(-x, K, dim=1).indices.to(torch.int32)
            for tag, pre in (("good", good), ("bad", bad), ("adv", adv)):
                idx = gvr29_topk(x, sl, K, pre)
                ok = True
                for r in (0, BS // 2, BS - 1):
                    got = x[r][idx[r].long()].sort(descending=True).values
                    if not torch.equal(got, ref.values[r]):
                        ok = False
                        break
                print(f"  K={K} N={N} BS={BS} hint={tag} exact={ok}",
                      flush=True)
                assert ok
            del x, ref, good, bad, adv
            torch.cuda.empty_cache()
    print("SMOKE EXACT")
