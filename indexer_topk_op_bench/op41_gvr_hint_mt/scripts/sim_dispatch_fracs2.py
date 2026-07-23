# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41: per-(group, BS) simulation with the REAL dispatch (AR, HS) of
launch_bs, comparing stock quantile fractions vs alternatives. Metric =
layers one-passed (any rung count in [K, kC]) out of all GVR-active layers.
The straggler tiers are AR4/HS2-4 — this finds a zero-cost frac retune."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle  # noqa: E402
from sim_stage3 import sim_rungs  # noqa: E402

LAYERS = {"flash": list(range(2, 43, 2)), "pro": list(range(2, 61, 2)),
          "v32": list(range(3, 61))}
GROUPS = [("v32", "16k"), ("v32", "64k"), ("v32", "128k"), ("v32", "256k"),
          ("pro", "256k"), ("pro", "512k"), ("pro", "1024k"),
          ("flash", "512k")]
BS_LIST = [16, 64, 256, 1024]

FRACS = {
    "AR4": [25, 65], "AR6": [15, 40, 70, 92], "AR8": [10, 25, 45, 65, 82, 94],
}
ALTS = {
    "AR4": {"a4_2580": [25, 80], "a4_2587": [25, 87], "a4_3085": [30, 85],
            "a4_3075": [30, 75], "a4_3570": [35, 70], "a4_5588": [55, 88],
            "a4_5090": [50, 90], "a4_6090": [60, 90], "a4_5085": [50, 85],
            "a4_3070": [30, 70]},
    "AR6": {"a6_shift": [25, 50, 75, 92], "a6_hi": [30, 55, 80, 94]},
    "AR8": {},
}


def dispatch(npad, K, bs):
    """(AR, HS) replicating launch_bs for our groups."""
    if npad <= 49152:
        if bs <= 8:
            return "AR8", 1
        if bs <= 128:
            return "AR6", 2
        return ("AR4", 2) if K <= 512 else ("AR6", 4)
    if npad <= 98304:
        if bs <= 8:
            return "AR8", 1
        if bs <= 16 and K <= 512:
            return "AR8", 1
        if 16 < bs <= 32 and K == 1024:
            return "AR8", 1
        if bs <= 64:
            return "AR6", 4
        if bs <= 128:
            return ("AR4", 2) if K <= 1024 else ("AR6", 4)
        return "AR4", 2
    if npad <= 163840:
        if bs <= 4:
            return "AR6", 1
        if bs <= 16:
            if bs > 8 and npad <= 131136:
                return ("AR6", 1) if K <= 512 else ("AR6", 2)
            return "AR6", 4
        if bs <= 32:
            return ("AR6", 2) if npad <= 131136 else ("AR8", 1)
        return "AR4", 2
    if npad <= 262144:
        if bs <= 4:
            return "AR8", 1
        if bs <= 16:
            return ("AR4", 2) if K <= 512 else ("AR6", 2)
        if bs <= 32:
            return "AR8", 1
        if bs <= 64:
            return ("AR4", 2) if K <= 512 else ("AR6", 4)
        if 256 < bs <= 512 and K <= 512:
            return "AR4", 2
        return ("AR4", 2) if K <= 512 else ("AR4", 4)
    return "AR8", 1


def main():
    print("group,BS,AR,HS,stock_ok,alts...")
    agg = {}
    for model, isl in GROUPS:
        rows, npad_ref = [], None
        for L in LAYERS[model]:
            try:
                b = bundle(model, isl, L)
            except Exception:
                continue
            npad = b["logits"].shape[1]
            if npad_ref is None:
                npad_ref = npad
            if npad != npad_ref:
                continue
            K = b["K"]
            lg = b["logits"][0].float().cuda()
            pre = b["preIdx"][0].to(torch.int64).cuda()
            pre = pre[(pre >= 0) & (pre < lg.numel())]
            rows.append((lg, lg[pre], K))
        kC = 8192 if rows[0][2] >= 2048 else 6144
        for bs in BS_LIST:
            ar, hs = dispatch(npad_ref, rows[0][2], bs)
            variants = [("stock", FRACS[ar])] + \
                [(nm, fr) for nm, fr in ALTS[ar].items()]
            oks = []
            for nm, fr in variants:
                ok = 0
                for lg, g, K in rows:
                    rungs = sim_rungs(g, K, fr, hs, False)
                    if any(K <= int((lg >= t).sum().item()) <= kC
                           for t in rungs):
                        ok += 1
                oks.append((nm, ok))
                key = (ar, hs, nm)
                a, b_ = agg.get(key, (0, 0))
                agg[key] = (a + ok, b_ + len(rows))
            print(f"{model}_{isl},{bs},{ar},{hs}," +
                  ",".join(f"{nm}:{ok}/{len(rows)}" for nm, ok in oks),
                  flush=True)
    print("\n== aggregate per (AR,HS,variant) ==", file=sys.stderr)
    for k in sorted(agg):
        a, b_ = agg[k]
        print(f"  {k}: {a}/{b_}", file=sys.stderr)


if __name__ == "__main__":
    main()
