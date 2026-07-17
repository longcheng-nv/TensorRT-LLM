# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op35 APEX-FR top-K v3 — python wrapper: pick_config policy + workspace cache.

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

PAIR_CAP = 12288  # must match apex_topk.cu
GCAP = 32768      # global candidate cap (must match apex_topk.cu)


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


N_SMALL = 0  # row-in-smem single-CTA mode FALSIFIED (iter14) — disabled


def pick_config(BS, N, K):
    if N <= N_SMALL:
        return dict(small=True, split=False)
    # mixed dispatch (iter17): fused single launch wins <=65536 (launch-bound),
    # split 3-kernel wins above (filter occupancy at NT1024 dominates)
    split = BS >= 32 and N > 65536
    if split:
        nt = 1024
        cpr = max(1, 256 // BS)
    else:
        nt = 512
        cpr = max(1, 148 // BS)
    s = min(8192, max(1024, 1 << math.ceil(math.log2(max(1, 8 * N // K)))))
    s = min(s, 1 << int(math.log2(max(2, N))))  # s <= N (pow2)
    assert s % 1024 == 0 or s >= 1024, (s,)     # uniform sample loops (s % NT)
    q = K / N
    r0 = s * q
    # scalar strata are spatially independent -> IID margins (iter15)
    sig = math.sqrt(max(1.0, s * q * (1 - q)))
    i_lo = min(s - 1, int(math.ceil(r0 + Z * sig)) - 1)
    tail_cap = 8192 if K <= 1024 else PAIR_CAP
    return dict(small=False, cpr=cpr, nt=nt, s=s, i_lo=i_lo, split=split,
                tail_cap=tail_cap)


_ws = {}


def workspace(BS, K, cfg, device):
    key = (BS, K, str(device))
    if key not in _ws:
        _ws[key] = dict(
            cand=torch.empty(BS, GCAP * 2, dtype=torch.int32, device=device),
            counts=torch.zeros(BS * 3, dtype=torch.int32, device=device),
            tickets=torch.zeros(BS, dtype=torch.int32, device=device),
            thr=torch.empty(BS, dtype=torch.float32, device=device),
            out=torch.empty(BS, K, dtype=torch.int32, device=device))
    return _ws[key]


_EMPTY = None
_streams = None


def apex_topk_pipelined(x, K, N, cfg, ws, dbg, chunks=4):
    # C++-side chunked 3-stream pipeline (thr/filter/tail overlap across chunks)
    ext().apex_pipe(x, ws["out"], ws["cand"], ws["counts"], ws["tickets"],
                    ws["thr"], N, K, cfg["cpr"], cfg["nt"], cfg["s"],
                    cfg["i_lo"], SEED, cfg["tail_cap"], chunks)
    return ws["out"]


def apex_topk(x, K, N=None, cfg=None, ws=None, mode=3, dbg=None):
    global _EMPTY
    assert x.dtype in (torch.float32, torch.bfloat16, torch.float16)
    assert x.dim() == 2 and x.is_contiguous()
    if cfg is not None and cfg.get("small"):
        assert x.dtype == torch.float32  # small kernel is fp32-only (disabled)
    BS = x.size(0)
    if N is None:
        N = x.size(1)
    assert N <= x.size(1) and K <= N
    if cfg is None:
        cfg = pick_config(BS, N, K)
    if ws is None:
        ws = workspace(BS, K, cfg, x.device)
    if cfg.get("small"):
        ext().apex_small(x, ws["out"], N, K)
        return ws["out"]
    if dbg is None:
        if _EMPTY is None:
            _EMPTY = torch.empty(0, dtype=torch.int32, device=x.device)
        dbg = _EMPTY
    if (cfg["split"] and mode == 3 and cfg.get("pipeline", False)
            and x.size(0) >= 64):  # chunked pipe FALSIFIED iter18; opt-in only
        return apex_topk_pipelined(x, K, N, cfg, ws, dbg,
                                   chunks=cfg.get("chunks", 4))
    ext().apex_topk(x, ws["out"], ws["cand"], ws["counts"], ws["tickets"],
                    ws["thr"], N, K, cfg["cpr"], cfg["nt"], cfg["s"],
                    cfg["i_lo"], SEED, cfg["tail_cap"], mode, cfg["split"], dbg)
    return ws["out"]
