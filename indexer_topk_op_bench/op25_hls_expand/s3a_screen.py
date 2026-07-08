# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 Step 5 (S3a) — admission cost of a small-N kC diet, host replay.

Small-N HLS is floor-bound; kC (band-accept + collect budget) sets the
P3 over-collect P4 feeds on (op13 precedent: kCC 2-3xK nets ~10-15% at
K512 small N on the secant kernel). This screens what a kC diet does to
the S1a ship ladder's admission (band_gt_kC + falsi aim shrink) on the
small-N bundles; the perf side is silicon's call (ab_kc.py).

Arms: ship ladder (wide4b for K512/1024, stock for K2048), kC in
{100%, 50%, 30%} of spec. Slot cap unchanged (t=512 floor keeps
slot_cap=8*scale regardless of kC).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[0]
sys.path.insert(0, str(BENCH / "ops"))
sys.path.insert(0, str(BENCH / "op21_gvr_prod" / "scripts"))

import proto_hls as P  # noqa: E402

WIDE4B = (0.92, 0.60, 0.25, 0.048)
STOCK = (0.75, 0.5, 0.25)
KC_FRACS = (1.0, 0.5, 0.3)
NS = (4096, 8192, 16384, 32768)
CAP = 8192


def eval_row(row):
    qf = WIDE4B if row.K in (512, 1024) else STOCK
    kc0 = row.kC
    out = {}
    for kf in KC_FRACS:
        row.kC = max(int(kc0 * kf), row.K + 64)
        cols, fr = P.cols_static(row, qfracs=qf)
        r0 = P.simulate_r0(row, cols, fr, cap=CAP)
        r0["_cols"] = cols
        fbp = 0
        if r0["mode"] != "fast":
            fbp, _ = P.fb_logfalsi(row, r0, alpha=0.2)
        out[f"kc{int(kf * 100)}"] = {"mode": r0["mode"], "fbp": fbp,
                                     "kC": row.kC}
    row.kC = kc0
    return out


def main():
    recs = []
    b22 = BENCH / "op22_temporal_fixed_hr_bench"
    for scen, root in (("best", b22 / "bundles_rr" / "best"),
                       ("worst", b22 / "bundles_rr" / "worst"),
                       ("real", b22 / "bundles" / "real")):
        for md in sorted(root.glob("*_fp32_N*")):
            model = md.name.split("_")[0]
            N = int(md.name.split("_N")[1])
            if N not in NS:
                continue
            leaf = sorted(md.glob("*_bs1"))
            if not leaf:
                continue
            K = {"v4flash": 512, "v4pro": 1024, "v32": 2048}[model]
            d = leaf[0]
            logits = torch.load(d / "logits.pt", map_location=P.DEV)
            preIdx = torch.load(d / "preIdx.pt", map_location=P.DEV)
            meta = json.loads((d / "meta.json").read_text())
            row = P.Row(scen, K, N, {"logits": logits,
                                     "preIdx": preIdx.to(torch.int32),
                                     "cr": meta["compress_ratio"]})
            recs.append({"scen": scen, "K": K, "N": N,
                         "h": round(row.h_true, 3), "arms": eval_row(row)})
    agg = defaultdict(lambda: {"n": 0, "fast": 0, "fbp": 0.0})
    for r in recs:
        for a, v in r["arms"].items():
            key = (r["scen"], r["K"], a)
            agg[key]["n"] += 1
            agg[key]["fast"] += v["mode"] == "fast"
            agg[key]["fbp"] += v["fbp"]
    print(f"{'scen':6s} {'K':>5s}" + "".join(
        f" {a:>18s}" for a in ("kc100", "kc50", "kc30")))
    for scen in ("best", "worst", "real"):
        for K in (512, 1024, 2048):
            line = f"{scen:6s} {K:5d}"
            for a in ("kc100", "kc50", "kc30"):
                g = agg[(scen, K, a)]
                if g["n"]:
                    line += (f"  fast={g['fast'] / g['n']:.2f} "
                             f"fbp={g['fbp'] / g['n']:.2f}")
                else:
                    line += f" {'--':>18s}"
            print(line)
    (HERE / "results" / "s3a_screen.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")


if __name__ == "__main__":
    main()
