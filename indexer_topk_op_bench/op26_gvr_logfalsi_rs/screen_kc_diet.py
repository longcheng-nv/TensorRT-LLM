# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 backlog-3 host screen: kC-diet K512@1536 static admission.

Narrowing the classic window [K, kC] from kC=5120 to 1536 saves ~28KB
SMEM (keys+vals slots) -> occupancy, but a 2.3x narrower window raises
the R0 miss rate and pushes pressure onto the fb_fix fallback. Per
RESUME_POST_ITER7.md section 4: screen admission on the host FIRST;
only silicon-A/B if the acceptance holds.

Ladders screened = M2D (shipped default) + uh4 (backlog-1 candidate).
Grid = all op22rr bundles at K=512 (3 scenarios x 3 dtypes x all N) +
Suite-B adversarial poles (hr0/hr1), window [512, kC] for
kC in {5120, 3072, 1536}. Kernel-faithful rung extraction reused from
screen_r0_qfracs.hist_quantile_rungs.

Usage: python3 screen_kc_diet.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "op22_temporal_fixed_hr_bench"))
from screen_r0_qfracs import hist_quantile_rungs  # noqa: E402
import bundle_data_rr                             # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 512
KCS = (5120, 3072, 1536)
LADDERS = {"m2d": (0.85, 0.35), "uh4": (0.90, 0.65, 0.40, 0.15)}
SCENARIOS = ("real", "best", "worst")
DTS = ("fp32", "bf16", "fp16")
NS = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
TDT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    # stats[(ladder, kC)] and stats[(ladder, kC, scenario)]
    stats = defaultdict(lambda: {"n": 0, "acc": 0, "bracket": 0,
                                 "below": 0, "above": 0, "cand": []})
    rows = []
    for sc in SCENARIOS:
        for dt in DTS:
            for N in NS:
                try:
                    b = bundle_data_rr.get_bundle(sc, K, TDT[dt], N,
                                                  device=DEV)
                except (FileNotFoundError, KeyError, AssertionError):
                    continue
                row_f32 = b["logits"][0].float()
                cr = b["cr"]
                off = 1 if cr == 1 else 0
                idx = b["preIdx"][0].long() + off
                idx = idx[(idx >= 0) & (idx < N)]
                if idx.numel() < 8:
                    continue
                pools = {"": row_f32[idx],
                         "hr1": torch.topk(row_f32[:N], K).values,
                         "hr0": torch.topk(row_f32[:N], K,
                                           largest=False).values}
                for hrname, pv in pools.items():
                    pmean = float(pv.mean())
                    scen = sc if not hrname else hrname
                    for lname, qf in LADDERS.items():
                        rungs = hist_quantile_rungs(pv, qf, K, pmean)
                        counts = [int((row_f32[:N] >= t).sum())
                                  for t in rungs]
                        for kc in KCS:
                            adm = [c for c in counts if K <= c <= kc]
                            srt = sorted(counts)
                            key_all = (lname, kc)
                            key_sc = (lname, kc, scen)
                            for kk in (key_all, key_sc):
                                s = stats[kk]
                                s["n"] += 1
                                if adm:
                                    s["acc"] += 1
                                    s["cand"].append(min(adm))
                                elif all(c > kc for c in counts):
                                    s["above"] += 1
                                elif all(c < K for c in counts):
                                    s["below"] += 1
                                else:
                                    s["bracket"] += 1
                            if not hrname:
                                rows.append((sc, dt, N, lname,
                                             "|".join(map(str, counts))))
    scens = SCENARIOS + ("hr0", "hr1")
    print(f"K={K} cells x ladders x kC windows; DEV={DEV}\n")
    print("static admission rate (any rung count in [512, kC]):")
    hdr = f"{'ladder':6s} {'kC':>5s} {'ALL':>7s}" + \
        "".join(f" {sc:>7s}" for sc in scens)
    print(hdr)
    for ln in LADDERS:
        for kc in KCS:
            s = stats[(ln, kc)]
            line = f"{ln:6s} {kc:5d} {s['acc']/max(s['n'],1):7.3f}"
            for sc in scens:
                t = stats[(ln, kc, sc)]
                line += f" {t['acc']/max(t['n'],1):7.3f}"
            print(line)
    print("\nmiss decomposition (grid+poles) + accepted cand size:")
    for ln in LADDERS:
        for kc in KCS:
            s = stats[(ln, kc)]
            cand = sorted(s["cand"])
            med = cand[len(cand)//2] if cand else -1
            p90 = cand[int(len(cand)*0.9)] if cand else -1
            print(f"{ln:6s} kC={kc:5d} miss={s['n']-s['acc']:4d}/{s['n']:4d} "
                  f"(bracket={s['bracket']:4d} below={s['below']:3d} "
                  f"above={s['above']:4d})  cand med={med} p90={p90}")


if __name__ == "__main__":
    main()
