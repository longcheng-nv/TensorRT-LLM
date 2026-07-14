# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op34 — current BS=1 fp32 cold-L2 gap map from the already-parsed op22 v4cap
sweep (results_b200_op22v4cap/v4cap_sweep/results.jsonl, 44982 recs).

For every (model, ISL, layer) it pulls us_cold for the arms of interest and
reports, aggregated per (model,ISL) as geomean over layers:
  - sgl      : sglang_v2 latency (the target)
  - r0       : op26_r0auto latency (our starting incumbent-to-beat's-rival)
  - base     : gvr_cutedsl latency (reference)
  - best_it  : min over in-tree GVR arms {base, mc, hls, r0}
  - r0/sgl   : how much slower r0 is than sglang (>1 => sgl faster)
  - need     : improvement factor op34 must get over r0 to beat sgl by 30%
               = 1.30 * (r0_us / sgl_us)
Also prints the BS=1 fp32 grand geomean (the headline the user's 30% is on).
"""
import json
import math
from collections import defaultdict
from pathlib import Path

RES = (Path(__file__).resolve().parents[2] /
       "results_b200_op22v4cap" / "v4cap_sweep" / "results.jsonl")
ARMS = ["sglang_v2", "op26_r0auto", "gvr_cutedsl", "gvr_multicta_cutedsl",
        "op27_hls", "radix_single_cuda", "radix_multi_cuda", "flashinfer_topk"]
INTREE_GVR = ["gvr_cutedsl", "gvr_multicta_cutedsl", "op27_hls", "op26_r0auto"]


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    # (model,isl,layer) -> {arm: us_cold}
    cell = defaultdict(dict)
    order = []
    for line in RES.read_text().splitlines():
        r = json.loads(line)
        if r["BS"] != 1 or r["dtype"] != "fp32":
            continue
        if r.get("us_cold") is None or "error" in r:
            continue
        k = (r["model"], r["isl"], r["layer"])
        cell[k][r["op"]] = r["us_cold"]
        io = (r["model"], r["isl"])
        if io not in order:
            order.append(io)

    ISLN = {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536,
            "128k": 131072, "256k": 262144, "512k": 524288, "1024k": 1048576}
    order.sort(key=lambda mi: (mi[0], ISLN[mi[1]]))

    print(f"{'cell':13s} {'N':>7s} {'sgl':>7s} {'r0':>7s} {'base':>7s} "
          f"{'bestIT':>7s} {'r0/sgl':>7s} {'need×':>6s} {'winlayers':>9s}")
    all_r0, all_sgl, all_base = [], [], []
    for (model, isl) in order:
        rows = [cell[(model, isl, L)] for L in
                sorted({L for (m, i, L) in cell if m == model and i == isl})]
        sgl = [c.get("sglang_v2") for c in rows]
        r0 = [c.get("op26_r0auto") for c in rows]
        base = [c.get("gvr_cutedsl") for c in rows]
        bestit = [min([c[a] for a in INTREE_GVR if a in c] or [None])
                  for c in rows]
        # per-layer: does ANY in-tree GVR already beat sgl?
        winl = sum(1 for c in rows if c.get("sglang_v2") and
                   min([c[a] for a in INTREE_GVR if a in c] or [9e9])
                   < c["sglang_v2"])
        gsgl, gr0, gbase = gm(sgl), gm(r0), gm(base)
        gbest = gm([x for x in bestit if x])
        N = [r.get("N") for r in
             [json.loads(l) for l in RES.read_text().splitlines()[:1]]]
        # N from any rec of this cell
        n = next((v for (m, i, L), d in cell.items()
                  if m == model and i == isl for v in [None]), None)
        print(f"{model+'/'+isl:13s} {'':>7s} {gsgl:7.2f} {gr0:7.2f} "
              f"{gbase:7.2f} {gbest:7.2f} {gr0/gsgl:7.3f} "
              f"{1.30*gr0/gsgl:6.2f} {winl:>4d}/{len(rows):<4d}")
        all_r0 += [x for x in r0 if x]
        all_sgl += [x for x in sgl if x]
        all_base += [x for x in base if x]

    print("-" * 78)
    G_r0, G_sgl, G_base = gm(all_r0), gm(all_sgl), gm(all_base)
    print(f"GRAND BS=1 fp32 geomean:  sgl={G_sgl:.2f}  r0={G_r0:.2f}  "
          f"base={G_base:.2f}   r0/sgl={G_r0/G_sgl:.3f}  "
          f"base/sgl={G_base/G_sgl:.3f}")
    print(f"op34 target: new GVR <= sgl/1.30 = {G_sgl/1.30:.2f} us  "
          f"=> need {1.30*G_r0/G_sgl:.2f}x over r0  "
          f"({1.30*G_base/G_sgl:.2f}x over base)")


if __name__ == "__main__":
    main()
