# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Parameterized host fp32 replay of the GVR Phase-2 secant + candidate logic.

Faithful to ``harness/count_gvr_iters.py`` (validated 108/108 against the real
cuteDSL kernel), but exposes the P2 secant knobs as parameters so we can sweep
them cheaply on the host (no kernel recompile) to find a configuration that
LOWERS the Phase-4 candidate count while keeping Phase-2 ``count_ge`` passes low
and staying EXACT (no fallback) across every (dtype, K, N, beta-cfg, seed).

Tunable knobs (``SecantCfg``):
  * init_mode  — how the Phase-1 initial threshold is chosen from the preIdx
                 value stats {pmin, pmax, pmean (+ optional quantile)}:
                   "mean"      : thr0 = pmean                (kernel baseline)
                   "lerp"      : thr0 = pmean + a*(pmax-pmean), a=init_alpha
                   "pquantile" : thr0 = q-th-from-top preIdx value, q=init_q
  * kFTarget   — secant aim count (None => GvrParams default)
  * kCC        — candidate acceptance upper bound / P4 cap (None => default)
  * f_lo,f_hi  — secant interpolation fraction clamp (default 0.05, 0.95)
  * f_iter0_cap— first-iter fraction cap (default 0.5)

Returns, per row, everything needed to score a config:
  p2_evals, cand_count, converged(done==1), exact(value-equiv to torch.topk).

Speed: counts via ``searchsorted`` on the pre-sorted row -> O(log N) per eval.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "ops"))
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrParams  # noqa: E402

F32 = np.float32
FLT_MAX = F32(3.4028235e38)
NEG_FLT_MAX = F32(-3.4028235e38)
MAX_REFINE_ITERS = 15

_DTYPE_NAME = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}


@dataclass(frozen=True)
class SecantCfg:
    init_mode: str = "mean"          # "mean" | "lerp" | "pquantile"
    init_alpha: float = 0.0          # for "lerp"
    init_q: float = 0.5              # for "pquantile": fraction-from-top in (0,1]
    kFTarget: Optional[int] = None   # None -> GvrParams default
    kCC: Optional[int] = None        # None -> GvrParams default
    f_lo: float = 0.05
    f_hi: float = 0.95
    f_iter0_cap: float = 0.5
    interp_mode: str = "linear"      # "linear" | "logcount" | "illinois" | "logillinois"

    def tag(self) -> str:
        parts = [self.init_mode]
        if self.init_mode == "lerp":
            parts.append(f"a{self.init_alpha:g}")
        elif self.init_mode == "pquantile":
            parts.append(f"q{self.init_q:g}")
        if self.kFTarget is not None:
            parts.append(f"ft{self.kFTarget}")
        if self.kCC is not None:
            parts.append(f"kc{self.kCC}")
        if (self.f_lo, self.f_hi, self.f_iter0_cap) != (0.05, 0.95, 0.5):
            parts.append(f"f{self.f_lo:g}-{self.f_hi:g}-{self.f_iter0_cap:g}")
        if self.interp_mode != "linear":
            parts.append(self.interp_mode)
        return "/".join(parts)


@dataclass
class RowStat:
    p2_evals: int
    p2_iters: int
    cand_count: int      # entering P4 (== count_ge(thr_final), capped at kCC)
    converged: bool      # done == 1
    exact: bool          # value-equivalent to torch.topk
    thr_final: float


def _prep_row(logits_row, pre_idx_row, N, K, cr, dtype):
    """Pre-sort row once; precompute preIdx value stats + true K-th value."""
    x = logits_row[:N].to(dtype).float().contiguous()
    xs, _ = torch.sort(x)                       # ascending, for searchsorted
    v_K = float(torch.kthvalue(x, N - K + 1).values.item())  # K-th largest value
    pre_idx_offset = 0 if cr == 4 else 1
    idx = pre_idx_row.to(torch.int64) + pre_idx_offset
    valid = (idx >= 0) & (idx < N)
    pcnt = int(valid.sum().item())
    if pcnt > 0:
        pv = x[idx[valid]]
        pvs, _ = torch.sort(pv, descending=True)  # preIdx values, desc
        pmin = F32(pv.min().item()); pmax = F32(pv.max().item())
        pmean = F32(F32(pv.sum().item()) / F32(pcnt))
    else:
        pvs = None; pmin, pmax = FLT_MAX, NEG_FLT_MAX
        pmean = F32((pmin + pmax) * F32(0.5))
    return dict(xs=xs, N=N, v_K=v_K, pmin=pmin, pmax=pmax, pmean=pmean,
                pvs=pvs, pcnt=pcnt)


def _count_ge(xs_sorted, thr):
    """#elements >= thr via searchsorted on ascending-sorted row."""
    pos = torch.searchsorted(xs_sorted, torch.tensor(float(thr), device=xs_sorted.device),
                             right=False)
    return int(xs_sorted.numel() - pos.item())


def _init_thr(prep, cfg: SecantCfg):
    pmin, pmax, pmean = prep["pmin"], prep["pmax"], prep["pmean"]
    if cfg.init_mode == "mean":
        return pmean
    if cfg.init_mode == "lerp":
        return F32(pmean + F32(cfg.init_alpha) * (pmax - pmean))
    if cfg.init_mode == "pquantile":
        pvs = prep["pvs"]
        if pvs is None or pvs.numel() == 0:
            return pmean
        r = int(min(max(cfg.init_q, 1e-6), 1.0) * (pvs.numel() - 1))
        return F32(pvs[r].item())   # q-th-from-top preIdx value
    raise ValueError(cfg.init_mode)


def replay_row(logits_row, pre_idx_row, N, K, cr, dtype, cfg: SecantCfg) -> RowStat:
    gp = GvrParams.get(_DTYPE_NAME[dtype], K, cr)
    kK = K
    kCC = cfg.kCC if cfg.kCC is not None else gp.kC
    kFTarget = cfg.kFTarget if cfg.kFTarget is not None else gp.kFTarget

    prep = _prep_row(logits_row, pre_idx_row, N, K, cr, dtype)
    xs, v_K = prep["xs"], prep["v_K"]
    pmin, pmax = prep["pmin"], prep["pmax"]

    thr = _init_thr(prep, cfg)
    val_lo, val_hi = pmin, pmax
    cnt_lo = kK + (kK >> 2)
    cnt_hi = 1

    if pmax <= NEG_FLT_MAX or pmin >= pmax:
        # degenerate -> identity; treat as exact iff N<=K
        cand = min(kK, N)
        return RowStat(0, 0, cand, True, N <= K, float(thr))

    log_family = cfg.interp_mode in ("logcount", "logillinois")
    ill_family = cfg.interp_mode in ("illinois", "logillinois")

    # Effective bracket counts used ONLY for interpolation. Illinois scales the
    # stale endpoint's count toward kFTarget when the same side is replaced
    # twice in a row; the real counts still drive the done/window logic.
    clo_eff = F32(cnt_lo)
    chi_eff = F32(cnt_hi)
    last_side = 0  # +1 = lo replaced last, -1 = hi replaced last

    def _scale_stale(c_eff):
        if log_family:
            return F32(np.sqrt(F32(kFTarget) * max(c_eff, F32(1.0))))
        return F32(F32(kFTarget) + (c_eff - F32(kFTarget)) * F32(0.5))

    done = 0
    p2_evals = 1
    c = _count_ge(xs, thr)
    if kK <= c <= kCC:
        done = 1
    elif c > kCC:
        val_lo, cnt_lo = thr, c
        clo_eff, last_side = F32(c), 1
    else:
        val_hi, cnt_hi = thr, c
        chi_eff, last_side = F32(c), -1

    it = 0
    while it < MAX_REFINE_ITERS and done == 0:
        vlo, vhi = val_lo, val_hi
        clo, chi = clo_eff, chi_eff
        rng = F32(vhi - vlo)
        if clo > chi and rng > F32(1e-10):
            if log_family:
                chi_c = max(F32(chi), F32(1.0))
                denom = F32(np.log2(F32(clo) / chi_c))
                if denom > F32(0.0):
                    f = F32(F32(np.log2(F32(clo) / F32(kFTarget))) / denom)
                else:
                    f = F32(F32(clo - kFTarget) / F32(clo - chi))
            else:
                f = F32(F32(clo - kFTarget) / F32(clo - chi))
            f = F32(max(F32(cfg.f_lo), f))
            f = F32(min(f, F32(cfg.f_hi)))
            if it == 0:
                f = F32(min(f, F32(cfg.f_iter0_cap)))
            nv = F32(vlo + rng * f)
        else:
            nv = F32((vlo + vhi) * F32(0.5))
        if nv <= vlo:
            nv = F32(vlo + rng * F32(0.05))
        if nv >= vhi:
            nv = F32(vhi - rng * F32(0.05))
        if nv == vlo or nv == vhi:
            nv = F32((vlo + vhi) * F32(0.5))
            if nv == vlo or nv == vhi:
                thr = vlo
                done = 2
            else:
                thr = nv
        else:
            thr = nv

        if done == 0:
            c = _count_ge(xs, thr)
            p2_evals += 1
            if kK <= c <= kCC:
                done = 1
            elif c > kCC:
                val_lo, cnt_lo = thr, c
                if ill_family and last_side == 1:
                    chi_eff = _scale_stale(chi_eff)
                clo_eff, last_side = F32(c), 1
            else:
                val_hi, cnt_hi = thr, c
                if ill_family and last_side == -1:
                    clo_eff = _scale_stale(clo_eff)
                chi_eff, last_side = F32(c), -1
        it += 1

    if done == 0:
        thr = val_lo if cnt_lo <= kCC * 2 else val_hi
        done = 2

    # candidate count entering P4
    c_final = _count_ge(xs, thr)
    cand = min(c_final, kCC)
    converged = (done == 1)
    # value-exactness: superset must hold (thr <= v_K, i.e. c_final >= K with ties)
    # AND no cap truncation (c_final <= kCC). P4 then snaps exactly.
    exact = (c_final >= kK) and (c_final <= kCC)
    return RowStat(p2_evals, it, cand, converged, exact, float(thr))
