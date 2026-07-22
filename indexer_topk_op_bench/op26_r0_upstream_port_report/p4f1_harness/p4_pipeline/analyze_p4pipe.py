# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze p4pipe_full.csv -> p4pipe_analysis.json + console findings.

Per-cell only (no cross-cell averaging except explicit medians/quantiles,
mirroring §9e discipline). Feeds §9f prose + charts.
"""
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUB = ["p4_peer_wait", "p4_dsmem_gather", "p4_minmax", "p4_coarse_hist",
       "p4_coarse_search", "p4_fine", "p4_scatter", "p4_tail"]


def rung(d):
    if d["cs"] == 1:
        return "cs1-small" if d["N"] <= 8448 else "cs1-mid"
    return "cs4" if d["cs"] == 4 else "cs8"


def med(v):
    return st.median(v) if v else None


def q(v, p):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))]


def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for j, i in enumerate(s):
            r[i] = j
        return r
    rx, ry = rk(x), rk(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main():
    rows = []
    for r in csv.DictReader(open(HERE / "p4pipe_full.csv")):
        d = dict(u=r["uuid"], m=r["model"], i=r["isl"], l=int(r["layer"]),
                 K=int(r["K"]), N=int(r["N"]), h=float(r["hit"]),
                 cs=int(r["cs"]), T=int(r["T"]),
                 us=float(r["us_prod_nsys"]), ov=float(r["overhead"]),
                 exact=r["exact"] == "True", mono=r["mono"] == "True",
                 drift=(float(r["p4frac_drift_vs_9e"])
                        if r.get("p4frac_drift_vs_9e") else None),
                 p4frac=float(r["frac_p4_select"]),
                 p4us=float(r["us_p4_select"]))
        for s in SUB:
            d[f"us_{s}"] = float(r[f"us_{s}"])
            d[f"sh_{s}"] = (float(r[f"p4share_{s}"])
                            if r[f"p4share_{s}"] else 0.0)
            d[f"kf_{s}"] = float(r[f"frac_{s}"])
        d["rg"] = rung(d)
        rows.append(d)

    A = {"n": len(rows),
         "exact": sum(d["exact"] for d in rows),
         "mono": sum(d["mono"] for d in rows),
         "ovh_med": med([d["ov"] for d in rows]),
         "ovh_p95": q([abs(d["ov"]) for d in rows], 0.95),
         "drift_med": med([d["drift"] for d in rows if d["drift"] is not None]),
         "drift_maxabs": max((abs(d["drift"]) for d in rows
                              if d["drift"] is not None), default=None)}

    # dominant sub-stage per cell
    dom = Counter()
    for d in rows:
        s = max(SUB, key=lambda x: d[f"us_{x}"])
        d["dom"] = s
        dom[s] += 1
    A["dominant_substage_counts"] = dict(dom.most_common())

    # per-rung medians (us + share-of-P4 + share-of-kernel)
    per_rung = {}
    for rg in ["cs1-small", "cs1-mid", "cs4", "cs8"]:
        g = [d for d in rows if d["rg"] == rg]
        per_rung[rg] = {
            "n": len(g),
            "p4us_med": med([d["p4us"] for d in g]),
            "us_med": med([d["us"] for d in g]),
            **{s: {"us": med([d[f"us_{s}"] for d in g]),
                   "shP4": med([d[f"sh_{s}"] for d in g]),
                   "shK": med([d[f"kf_{s}"] for d in g])} for s in SUB},
        }
    A["per_rung"] = per_rung

    # per (model, isl): share-of-P4 medians
    tbl = {}
    g2 = defaultdict(list)
    for d in rows:
        g2[(d["m"], d["i"])].append(d)
    for (m, i), g in sorted(g2.items()):
        tbl[f"{m}/{i}"] = {
            "n": len(g), "cs": g[0]["cs"],
            "p4us": med([d["p4us"] for d in g]),
            **{s: med([d[f"sh_{s}"] for d in g]) for s in SUB},
        }
    A["per_model_isl"] = tbl

    # K effect on coarse_hist (kNumBins == K: 512/1024/2048 zero+build cost)
    A["coarse_hist_by_K"] = {
        K: {"us_med": med([d["us_p4_coarse_hist"] for d in rows
                           if d["K"] == K]),
            "shP4_med": med([d["sh_p4_coarse_hist"] for d in rows
                             if d["K"] == K])}
        for K in (512, 1024, 2048)}

    # cluster tax: peer_wait + dsmem_gather at cs>=4
    for rg in ("cs4", "cs8"):
        g = [d for d in rows if d["rg"] == rg]
        A[f"cluster_tax_{rg}"] = {
            "us_med": med([d["us_p4_peer_wait"] + d["us_p4_dsmem_gather"]
                           for d in g]),
            "shP4_med": med([d["sh_p4_peer_wait"] + d["sh_p4_dsmem_gather"]
                             for d in g]),
            "shK_med": med([d["kf_p4_peer_wait"] + d["kf_p4_dsmem_gather"]
                            for d in g])}

    # correlations within rungs
    corr = {}
    for rg in ["cs1-small", "cs1-mid", "cs4", "cs8"]:
        g = [d for d in rows if d["rg"] == rg]
        if len(g) < 8:
            continue
        corr[rg] = {
            "hit_vs_tail_us": spearman([d["h"] for d in g],
                                       [d["us_p4_tail"] for d in g]),
            "hit_vs_scatter_us": spearman([d["h"] for d in g],
                                          [d["us_p4_scatter"] for d in g]),
            "N_vs_fine_us": spearman([d["N"] for d in g],
                                     [d["us_p4_fine"] for d in g]),
            "N_vs_peerwait_us": spearman([d["N"] for d in g],
                                         [d["us_p4_peer_wait"] for d in g]),
        }
    A["spearman"] = corr

    # degenerate cells (collapse: coarse..scatter ~ 0)
    dg = [d["u"] for d in rows
          if d[f"us_p4_coarse_hist"] + d["us_p4_fine"] + d["us_p4_scatter"]
          < 0.02 * d["p4us"]]
    A["degenerate_cells"] = dg

    # tail-heavy cells (exact-tail/p4tt fired hard)
    th = sorted(rows, key=lambda d: -d["sh_p4_tail"])[:20]
    A["tail_heavy_top20"] = [
        dict(u=d["u"], K=d["K"], h=d["h"], shP4=round(d["sh_p4_tail"], 3),
             us=round(d["us_p4_tail"], 2)) for d in th]

    # PR-loss class (from real_3arm csv merged at aggregation? use meta)
    meta = {f"{m['model']}_{m['isl']}_L{int(m['layer']):02d}": m
            for m in csv.DictReader(
                open(HERE.parents[1] / "real_3arm_layers_full.csv"))}
    loss = [d for d in rows if meta[d["u"]]["pr_vs_base"]
            and float(meta[d["u"]]["pr_vs_base"]) < 1.0]
    A["pr_loss_profile"] = {
        "n": len(loss),
        **{s: med([d[f"sh_{s}"] for d in loss]) for s in SUB},
    }
    A["grid_profile"] = {s: med([d[f"sh_{s}"] for d in rows]) for s in SUB}

    json.dump(A, open(HERE / "p4pipe_analysis.json", "w"), indent=1)
    print(json.dumps({k: A[k] for k in
                      ["n", "exact", "mono", "ovh_med", "ovh_p95",
                       "drift_med", "drift_maxabs",
                       "dominant_substage_counts"]}, indent=1))
    print("\nper-rung sub-stage medians (us | share-of-P4):")
    for rg, v in per_rung.items():
        line = "  ".join(f"{s.replace('p4_',''):>13s}="
                         f"{v[s]['us']:.2f}us/{100*v[s]['shP4']:.0f}%"
                         for s in SUB)
        print(f" {rg:9s} n={v['n']:3d} P4={v['p4us_med']:.2f}us | {line}")
    print(f"-> {HERE/'p4pipe_analysis.json'}")


if __name__ == "__main__":
    main()
