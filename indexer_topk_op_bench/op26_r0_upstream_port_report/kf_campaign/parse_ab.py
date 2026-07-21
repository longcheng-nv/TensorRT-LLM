# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse an nsys_ab rep -> per-cell cand-vs-pr speedups (cold + warm)."""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True)
    ap.add_argument("--tag", default="t0")
    args = ap.parse_args()
    kern = parse_rep(args.rep)   # {"c|arm|uuid": us_per_call, "w|...": ...}
    cells = {}
    for rng, us in kern.items():
        mode, arm, uuid = rng.split("|", 2)
        cells.setdefault(uuid, {})[f"{mode}|{arm}"] = us
    exact = {}
    ep = HERE / f"exact_{args.tag}.json"
    if ep.exists():
        exact = json.load(open(ep))
    out, ratios_c, ratios_w = [], [], []
    for uuid, d in sorted(cells.items()):
        rc = d.get("c|gvr_pr", 0) / d["c|kf_cand"] if d.get("c|kf_cand") else None
        rw = d.get("w|gvr_pr", 0) / d["w|kf_cand"] if d.get("w|kf_cand") else None
        ok = exact.get(f"{uuid}|kf_cand", [None])[0]
        if rc: ratios_c.append(rc)
        if rw: ratios_w.append(rw)
        out.append(dict(uuid=uuid, pr_cold=d.get("c|gvr_pr"),
                        cand_cold=d.get("c|kf_cand"), speedup_cold=rc,
                        pr_warm=d.get("w|gvr_pr"), cand_warm=d.get("w|kf_cand"),
                        speedup_warm=rw, cand_exact=ok))
        print(f"{uuid:22s} cold pr={d.get('c|gvr_pr',0):7.2f} "
              f"cand={d.get('c|kf_cand',0):7.2f} x{rc or 0:5.3f}  "
              f"warm x{rw or 0:5.3f}  exact={ok}")
    if ratios_c:
        gc = statistics.geometric_mean(ratios_c)
        gw = statistics.geometric_mean(ratios_w)
        nreg = sum(1 for x in ratios_c if x < 1.0)
        print(f"\nCOLD geomean {gc:.4f} (reg {nreg}/{len(ratios_c)})  "
              f"WARM geomean {gw:.4f}")
    (HERE / f"ab_{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
