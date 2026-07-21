# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Statistical analysis of the 865-cell phase breakdown (no averaging of the
final results: distributions, quantiles, per-cell classification).

Reads  phase_full_<tag>.csv (aggregate_phases.py) + ../real_3arm_layers_full.csv
Writes PHASE_FULL_ANALYSIS.md + phase_analysis.json (consumed by the REPORT
injector).

Statistics reported per (model, ISL) group and per class — median / p25 / p75 /
min / max over layers, never a single mean. Classification axes:
  R  dispatch rung: cs1-small (N<8.5k), cs1-mid, cs4, cs8-T512, cs8-T1024
  P  PR-vs-base behaviour (§7b frozen ratios, classification only):
     strong-win >=1.15 / win 1.05-1.15 / parity 0.95-1.05 / loss <0.95
  H  hint tier: hi >=0.60 / mid 0.35-0.60 / lo <0.35
  S  within-(model,ISL) PR-time quartile: fastest 25% vs slowest 25% layers
"""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]

PHASES = ["p1_gather_stats", "smem_stage", "p1b_rungs", "p2_count_admission",
          "p3_collect", "p4_select", "epilogue"]
PLABEL = {
    "p1_gather_stats": "P1 gather/stats",
    "smem_stage": "smem-stage",
    "p1b_rungs": "P1b rungs",
    "p2_count_admission": "P2 count+admission",
    "p3_collect": "P3 collect",
    "p4_select": "P4 select(+tail)",
    "epilogue": "epilogue",
}
ISL_ORDER = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]


def q(vals, p):
    s = sorted(vals)
    if not s:
        return None
    i = (len(s) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def fivenum(vals):
    return dict(min=min(vals), p25=q(vals, .25), med=q(vals, .5),
                p75=q(vals, .75), max=max(vals), n=len(vals))


def rung_of(r):
    if r["cs"] == 1:
        return "cs1-small" if r["N"] <= 8448 else "cs1-mid"
    if r["cs"] == 4:
        return "cs4"
    return f"cs8-T{r['T']}"


def pclass_of(v):
    if v is None:
        return "n/a"
    if v >= 1.15:
        return "strong-win"
    if v >= 1.05:
        return "win"
    if v >= 0.95:
        return "parity"
    return "loss"


def htier_of(h):
    return "hi" if h >= 0.60 else ("mid" if h >= 0.35 else "lo")


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            rr = (i + j) / 2.0
            for k2 in range(i, j + 1):
                r[s[k2]] = rr
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** .5
    dy = sum((b - my) ** 2 for b in ry) ** .5
    return num / (dx * dy) if dx and dy else 0.0


def main(tag="full"):
    rows = []
    for r in csv.DictReader(open(HERE / f"phase_full_{tag}.csv")):
        d = dict(uuid=r["uuid"], model=r["model"], isl=r["isl"],
                 layer=int(r["layer"]), K=int(r["K"]), N=int(r["N"]),
                 hit=float(r["hit"]), cs=int(r["cs"]), T=int(r["T"]),
                 us=float(r["us_prod_nsys"]), ovh=float(r["overhead"]),
                 exact=r["exact"] == "True", mono=r["mono"] == "True")
        for ph in PHASES:
            d[f"f_{ph}"] = float(r[f"frac_{ph}"])
            d[f"u_{ph}"] = float(r[f"us_{ph}"])
        rows.append(d)
    meta = {f"{m['model']}_{m['isl']}_L{int(m['layer']):02d}": m
            for m in csv.DictReader(open(REPORT / "real_3arm_layers_full.csv"))}
    for d in rows:
        m = meta[d["uuid"]]
        d["pr_vs_base"] = float(m["pr_vs_base"]) if m["pr_vs_base"] else None
        d["rung"] = rung_of(d)
        d["pclass"] = pclass_of(d["pr_vs_base"])
        d["htier"] = htier_of(d["hit"])
    # within-(model,isl) speed quartile
    for (mo, isl), grp in group_by(rows, lambda d: (d["model"], d["isl"])).items():
        srt = sorted(grp, key=lambda d: d["us"])
        nq = max(1, len(srt) // 4)
        for d in srt[:nq]:
            d["squart"] = "fastest25"
        for d in srt[-nq:]:
            d["squart"] = "slowest25"
        for d in srt[nq:-nq]:
            d.setdefault("squart", "mid50")

    A = {}
    # 1. validation
    A["validation"] = dict(
        n=len(rows),
        inexact=sum(1 for d in rows if not d["exact"]),
        nonmono=sum(1 for d in rows if not d["mono"]),
        ovh=fivenum([d["ovh"] for d in rows]),
        ovh_gt7=sum(1 for d in rows if abs(d["ovh"]) > 0.07),
    )
    # 2. per model x ISL five-number stats (frac + us) per phase
    per_mi = {}
    for (mo, isl), grp in sorted(group_by(rows, lambda d: (d["model"], d["isl"])).items()):
        e = dict(n=len(grp), cs=grp[0]["cs"], T=grp[0]["T"], N=grp[0]["N"],
                 us=fivenum([d["us"] for d in grp]),
                 hit=fivenum([d["hit"] for d in grp]))
        for ph in PHASES:
            e[f"f_{ph}"] = fivenum([d[f"f_{ph}"] for d in grp])
            e[f"u_{ph}"] = fivenum([d[f"u_{ph}"] for d in grp])
        per_mi[f"{mo}|{isl}"] = e
    A["per_model_isl"] = per_mi
    # 3. per rung
    A["per_rung"] = {
        rg: {**dict(n=len(grp)),
             **{f"f_{ph}": fivenum([d[f"f_{ph}"] for d in grp]) for ph in PHASES}}
        for rg, grp in sorted(group_by(rows, lambda d: d["rung"]).items())}
    # 4. classes: pclass / htier / squart — phase fraction + us distributions
    for key in ("pclass", "htier", "squart"):
        A[f"by_{key}"] = {}
        for cls, grp in sorted(group_by(rows, lambda d: d[key]).items()):
            e = dict(n=len(grp),
                     us=fivenum([d["us"] for d in grp]),
                     hit=fivenum([d["hit"] for d in grp]),
                     pr_vs_base=fivenum([d["pr_vs_base"] for d in grp
                                         if d["pr_vs_base"] is not None]))
            for ph in PHASES:
                e[f"f_{ph}"] = fivenum([d[f"f_{ph}"] for d in grp])
                e[f"u_{ph}"] = fivenum([d[f"u_{ph}"] for d in grp])
            A[f"by_{key}"][cls] = e
    # 4b. pclass x rung counts (composition confounder view)
    A["pclass_rung_counts"] = {
        f"{pc}|{rg}": len(grp) for (pc, rg), grp in
        sorted(group_by(rows, lambda d: (d["pclass"], d["rung"])).items())}
    # 5. correlations (within rung, so N-scaling doesn't confound)
    corr = {}
    for rg, grp in sorted(group_by(rows, lambda d: d["rung"]).items()):
        if len(grp) < 12:
            continue
        c = {}
        for ph in PHASES:
            c[f"hit_vs_f_{ph}"] = round(spearman([d["hit"] for d in grp],
                                                 [d[f"f_{ph}"] for d in grp]), 3)
            c[f"hit_vs_u_{ph}"] = round(spearman([d["hit"] for d in grp],
                                                 [d[f"u_{ph}"] for d in grp]), 3)
        c["hit_vs_us"] = round(spearman([d["hit"] for d in grp],
                                        [d["us"] for d in grp]), 3)
        if any(d["pr_vs_base"] is not None for d in grp):
            g2 = [d for d in grp if d["pr_vs_base"] is not None]
            c["prvb_vs_f_p4"] = round(spearman([d["pr_vs_base"] for d in g2],
                                               [d["f_p4_select"] for d in g2]), 3)
            c["prvb_vs_hit"] = round(spearman([d["pr_vs_base"] for d in g2],
                                              [d["hit"] for d in g2]), 3)
        corr[rg] = c
    A["spearman_by_rung"] = corr
    # 6. dominant-phase census (per cell, no averaging)
    dom = defaultdict(int)
    for d in rows:
        dom[max(PHASES, key=lambda ph: d[f"f_{ph}"])] += 1
    A["dominant_phase_counts"] = dict(dom)
    A["p4_frac_overall"] = fivenum([d["f_p4_select"] for d in rows])

    (HERE / "phase_analysis.json").write_text(json.dumps(A, indent=1))

    # ---------- markdown ----------
    L = ["# 865-cell phase breakdown — statistical analysis (no averaging)", ""]
    v = A["validation"]
    L += [f"- cells {v['n']}; inexact {v['inexact']}; non-monotone {v['nonmono']}; "
          f"instr overhead med {v['ovh']['med']:+.3f} p75 {v['ovh']['p75']:+.3f} "
          f"max {v['ovh']['max']:+.3f}; |ovh|>7%: {v['ovh_gt7']}",
          f"- dominant phase census: " + ", ".join(
              f"{PLABEL[k]} {n}" for k, n in sorted(
                  A['dominant_phase_counts'].items(), key=lambda kv: -kv[1])),
          f"- P4 frac overall: med {A['p4_frac_overall']['med']:.3f} "
          f"[{A['p4_frac_overall']['min']:.3f}, {A['p4_frac_overall']['max']:.3f}]", ""]
    L += ["## per (model, ISL): frac med [p25, p75] per phase", ""]
    hdr = "| model/ISL | n | cs/T | us med [min,max] | " + \
        " | ".join(PLABEL[p] for p in PHASES) + " |"
    L += [hdr, "|" + "---|" * (4 + len(PHASES))]
    for key in sorted(per_mi, key=lambda k: (k.split("|")[0],
                                             ISL_ORDER.index(k.split("|")[1]))):
        e = per_mi[key]
        cells = [key, str(e["n"]), f"{e['cs']}/{e['T']}",
                 f"{e['us']['med']:.1f} [{e['us']['min']:.1f},{e['us']['max']:.1f}]"]
        for ph in PHASES:
            f = e[f"f_{ph}"]
            cells.append(f"{f['med']:.2f} [{f['p25']:.2f},{f['p75']:.2f}]")
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## classes", ""]
    for key, title in (("pclass", "PR-vs-base class (§7b frozen, classification only)"),
                       ("htier", "hint tier"), ("squart", "within-(model,ISL) speed quartile")):
        L += [f"### {title}", ""]
        for cls, e in A[f"by_{key}"].items():
            L.append(
                f"- **{cls}** n={e['n']} us med {e['us']['med']:.1f} "
                f"hit med {e['hit']['med']:.2f} | " + " ".join(
                    f"{PLABEL[ph].split()[0]} {e[f'f_{ph}']['med']:.2f}"
                    for ph in PHASES if e[f"f_{ph}"]["med"] >= 0.005))
        L.append("")
    L += ["## Spearman (within rung)", ""]
    for rg, c in corr.items():
        keep = {k: v for k, v in c.items() if abs(v) >= 0.3}
        L.append(f"- {rg}: " + (", ".join(f"{k}={v:+.2f}" for k, v in
                                          sorted(keep.items(), key=lambda kv: -abs(kv[1])))
                                or "no |rho|>=0.3"))
    (HERE / "PHASE_FULL_ANALYSIS.md").write_text("\n".join(L) + "\n")
    print("wrote phase_analysis.json + PHASE_FULL_ANALYSIS.md")


def group_by(rows, key):
    g = defaultdict(list)
    for d in rows:
        g[key(d)].append(d)
    return g


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
