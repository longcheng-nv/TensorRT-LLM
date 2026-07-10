# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Variant simulation on op22rr bundle rows: quantify algorithmic P2 fixes.

Variants (all keep acceptance [kK,kCC] and the vendored clamps):
  anchor : linear false-position, stock window, kFT stock       (baseline)
  op26   : log false-position, op26 window/kFT                  (current, regressed)
  V1     : log false-position, aim at GEOMETRIC CENTER sqrt(kK*kCC)
  V2     : log SECANT through last two MEASURED points, center aim,
           bracket-clamped (superlinear on log-linear tails; immune to
           unmeasured seeds and endpoint stagnation)
  V3     : V2 + op26 narrow window (keeps the P4 candidate savings)

Run: python3 diag_p2_variants.py   (companion to diag_p2_replay.py)
"""
import collections
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str((Path(__file__).parent / "../op22_temporal_fixed_hr_bench").resolve()))
sys.path.insert(0, str((Path(__file__).parent / "../harness").resolve()))
import bundle_data_rr  # noqa: E402

MAXI = 15
STOCK = {("fp32", 512): (512, 5120), ("fp32", 1024): (1024, 5120),
         ("fp32", 2048): (3072, 6144),
         ("bf16", 512): (512, 5120), ("bf16", 1024): (1024, 5120),
         ("bf16", 2048): (4096, 5120),
         ("fp16", 512): (512, 5120), ("fp16", 1024): (1024, 5120),
         ("fp16", 2048): (4096, 5120)}
DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def p1(row, pre, N, K):
    idx = pre[(pre >= 0) & (pre < N)].long()
    pv = row[idx]
    return float(pv.mean()), float(pv.min()), float(pv.max()), K + (K >> 2), 1


def run(row_f32, pre, N, K, kFT, kCC, mode):
    v = row_f32[:N]
    passes = [0]

    def cnt(t):
        passes[0] += 1
        return int((v >= t).sum().item())

    thr0, vlo, vhi, clo, chi = p1(row_f32, pre, N, K)
    meas = []
    c0 = cnt(thr0)
    meas.append((thr0, c0))
    if K <= c0 <= kCC:
        return passes[0], c0
    if c0 > kCC:
        vlo, clo = thr0, c0
    else:
        vhi, chi = thr0, c0
    for it in range(MAXI):
        rng = vhi - vlo
        nv = None
        f = 0.5
        if mode == "lin":
            if clo > chi and rng > 1e-10:
                f = (clo - kFT) / (clo - chi)
        elif mode in ("log", "logc"):
            if clo > chi and rng > 1e-10:
                chif = max(float(chi), 1.0)
                den = math.log2(clo / chif)
                f = math.log2(clo / kFT) / den if den > 0 else (clo - kFT) / (clo - chi)
        elif mode == "logsec":
            f = None
            if len(meas) >= 2:
                (v1, c1), (v2, c2) = meas[-2], meas[-1]
                l1, l2 = math.log2(max(c1, 1)), math.log2(max(c2, 1))
                if abs(l2 - l1) > 1e-6 and abs(v2 - v1) > 1e-10:
                    t = (math.log2(kFT) - l2) / (l1 - l2)
                    nv = v2 + t * (v1 - v2)
            if nv is None:
                chif = max(float(chi), 1.0)
                den = math.log2(clo / chif)
                f = math.log2(clo / kFT) / den if den > 0 else 0.5
        if nv is None:
            if clo > chi and rng > 1e-10:
                f = min(max(f, 0.05), 0.95)
                if it == 0:
                    f = min(f, 0.5)
                nv = vlo + rng * f
            else:
                nv = (vlo + vhi) * 0.5
        if nv <= vlo:
            nv = vlo + (vhi - vlo) * 0.05
        if nv >= vhi:
            nv = vhi - (vhi - vlo) * 0.05
        if nv == vlo or nv == vhi:
            nv = (vlo + vhi) * 0.5
            if nv == vlo or nv == vhi:
                return passes[0], clo  # give-up (not hit on these rows)
        c = cnt(nv)
        meas.append((nv, c))
        if K <= c <= kCC:
            return passes[0], c
        if c > kCC:
            vlo, clo = nv, c
        else:
            vhi, chi = nv, c
    return passes[0], -1


def main():
    cells = [(1024, "fp32", 131072), (1024, "fp32", 32768), (1024, "fp32", 8192),
             (1024, "bf16", 131072), (2048, "bf16", 32768), (2048, "fp16", 65536),
             (2048, "fp32", 32768), (2048, "fp32", 131072), (512, "fp32", 131072)]
    print(f"{'cell':<24}{'scen':<6}{'anchor':>9}{'op26':>9}{'V1':>9}{'V2':>9}{'V3':>9}   (passes/cand)")
    sums = collections.defaultdict(list)
    for K, dt, N in cells:
        kFTs, kCs = STOCK[(dt, K)]
        if dt == "fp32":
            okC, okFT = {512: (1536, 1280), 1024: (2048, 1024),
                         2048: (4096, 2048)}[K]
            use_narrow = True
        else:
            okC, okFT = kCs, kFTs
            use_narrow = False
        ctr_s = int(math.sqrt(K * kCs))
        ctr_n = int(math.sqrt(K * okC))
        for scen in ("real", "best", "worst"):
            b = bundle_data_rr.get_bundle(scen, K, DT[dt], N, device="cuda")
            row = b["logits"][0].float()
            pre = b["preIdx"][0]
            a = run(row, pre, N, K, kFTs, kCs, "lin")
            o = run(row, pre, N, K, okFT, okC, "log")
            v1 = run(row, pre, N, K, ctr_n if use_narrow else ctr_s, okC, "logc")
            v2 = run(row, pre, N, K, ctr_s, kCs, "logsec")
            v3 = run(row, pre, N, K, ctr_n, okC, "logsec")
            print(f"K{K:<5}{dt:<5}N{N:<8}{scen:<6}"
                  f"{a[0]:>4}/{a[1]:<5}{o[0]:>4}/{o[1]:<5}{v1[0]:>4}/{v1[1]:<5}"
                  f"{v2[0]:>4}/{v2[1]:<5}{v3[0]:>4}/{v3[1]:<5}")
            for name, r in (("anchor", a), ("op26", o), ("V1", v1),
                            ("V2", v2), ("V3", v3)):
                sums[name].append(r[0])
    print("\nmean passes:",
          {k: round(sum(v) / len(v), 2) for k, v in sums.items()})


if __name__ == "__main__":
    main()
