# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op27 iter2 — full-grid host replay: K2048 tail ladder vs stock.

Guards against the op25 historical rejection ("deep cols regress the v32
band geometry: real 0.75 -> 0.375 fast-rate in replay") applying to the
op27 tail variant (0.75, 0.45, 0.048 — stock top column kept). Replays
simulate_r0 + shipped logfalsi fallback on every op22rr K2048 bundle
(best/worst/real x all N), reporting mode mix + fallback passes per arm.

Usage: python3 screen_k2048_tail.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPB = HERE.parent
sys.path.insert(0, str(OPB / "op21_gvr_prod" / "scripts"))
sys.path.insert(0, str(OPB / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(OPB / "harness"))
sys.path.insert(0, str(OPB / "ops"))

import torch  # noqa: E402

import bundle_data_rr  # noqa: E402
from proto_hls import Row, cols_static, simulate_r0, fb_logfalsi  # noqa: E402

NS = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
ARMS = {"stock": (0.75, 0.5, 0.25), "tail": (0.75, 0.45, 0.048)}


def main():
    print(f"{'scen':>6} {'N':>8} | " + " | ".join(
        f"{a}: mode fb_p ok" for a in ARMS))
    agg = {a: {"fast": 0, "fb_passes": 0.0, "cells": 0} for a in ARMS}
    for scen in ("best", "worst", "real"):
        for N in NS:
            try:
                b = bundle_data_rr.get_bundle(scen, 2048, torch.float32, N)
            except Exception as e:
                print(f"skip {scen} N{N}: {type(e).__name__}")
                continue
            line = f"{scen:>6} {N:>8} | "
            for arm, qf in ARMS.items():
                row = Row(scen, 2048, N, b)
                cols, fr = cols_static(row, qfracs=qf)
                r0 = simulate_r0(row, cols, fr)
                r0["_cols"] = cols
                if r0["mode"] == "fast":
                    p, ok = 0.0, True
                    agg[arm]["fast"] += 1
                else:
                    p, ok = fb_logfalsi(row, r0, 0.2)
                agg[arm]["fb_passes"] += p
                agg[arm]["cells"] += 1
                line += f"{r0['mode'][:8]:>8} {p:>4.1f} {'Y' if ok else 'N'} | "
            print(line)
    print("\n== aggregate ==")
    for arm, d in agg.items():
        print(f"  {arm:5}: fast {d['fast']}/{d['cells']}  "
              f"mean fb passes {d['fb_passes']/max(d['cells'],1):.2f}")


if __name__ == "__main__":
    main()
