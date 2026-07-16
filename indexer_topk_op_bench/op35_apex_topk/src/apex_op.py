# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op35 APEX-FR top-K v0 — python wrapper: pick_config policy + workspace cache.

Usage:
    from apex_op import apex_topk
    idx = apex_topk(x, K)            # x [BS, stride] fp32 cuda, N defaults stride
    idx = apex_topk(x, K, N=1027)    # logical N < physical row stride
"""
import math
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None


def ext():
    global _ext
    if _ext is None:
        bd = os.environ.get("BUILD_DIR", "/tmp/op35_build")
        os.makedirs(bd, exist_ok=True)
        _ext = load(name="apex_topk_v0", sources=[os.path.join(HERE, "apex_topk.cu")],
                    extra_cuda_cflags=["-O3", "--use_fast_math",
                                       "-gencode=arch=compute_100,code=sm_100"],
                    build_directory=bd, verbose=False)
    return _ext


Z = 6.0
SEED = 0x0035A9E5


def pick_config(BS, N, K):
    if BS == 1:
        cpr = 148
    elif BS <= 8:
        cpr = max(1, 148 // BS)
    else:
        cpr = 1
    nt = 1024 if BS >= 32 else 512
    if N > 524288:
        s = 8192
    elif N > 262144:
        s = 4096
    else:
        s = 2048
    s = min(s, 1 << int(math.log2(max(2, N))))  # s <= N (pow2)
    assert s % nt == 0, (s, nt)  # warp-uniform sample loops in-kernel
    q = K / N
    r0 = s * q
    sig = math.sqrt(max(1.0, s * q * (1 - q)))
    i_hi = max(0, int(math.floor(r0 - Z * sig)) - 1)
    i_lo = min(s - 1, int(math.ceil(r0 + Z * sig)) - 1)
    segcap = 256 if cpr > 1 else 512
    nseg = (nt // 32) * cpr
    assert nseg <= 2432, nseg  # kernel warr capacity
    return dict(cpr=cpr, nt=nt, s=s, i_hi=i_hi, i_lo=i_lo, segcap=segcap,
                nseg=nseg)


_ws = {}


def workspace(BS, K, cfg, device):
    key = (BS, K, cfg["nseg"], cfg["segcap"], str(device))
    if key not in _ws:
        cap = cfg["nseg"] * cfg["segcap"]
        _ws[key] = dict(
            cand=torch.empty(BS, cap * 2, dtype=torch.int32, device=device),
            counts=torch.zeros(BS * (2 + cfg["nseg"]), dtype=torch.int32, device=device),
            tickets=torch.zeros(BS, dtype=torch.int32, device=device),
            out=torch.empty(BS, K, dtype=torch.int32, device=device))
    return _ws[key]


_EMPTY = None


def apex_topk(x, K, N=None, cfg=None, ws=None, mode=3, dbg=None):
    global _EMPTY
    assert x.dtype == torch.float32 and x.dim() == 2 and x.is_contiguous()
    BS = x.size(0)
    if N is None:
        N = x.size(1)
    assert N <= x.size(1) and K <= N
    if cfg is None:
        cfg = pick_config(BS, N, K)
    if ws is None:
        ws = workspace(BS, K, cfg, x.device)
    if dbg is None:
        if _EMPTY is None:
            _EMPTY = torch.empty(0, dtype=torch.int32, device=x.device)
        dbg = _EMPTY
    ext().apex_topk(x, ws["out"], ws["cand"], ws["counts"],
                    ws["tickets"], N, K, cfg["cpr"], cfg["nt"], cfg["s"],
                    cfg["i_hi"], cfg["i_lo"], SEED, cfg["segcap"], mode, dbg)
    return ws["out"]
