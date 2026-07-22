# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse the compB BS-scaling sweep -> kf_bs.csv, then anchor-check the local
gvr_pr arm against REPORT rival_bs_layers.csv (b200-027 run) and emit the
joined comparison table kf_bs_joined.csv.

us      = mean cold-L2 kernel-sum inside the NVTX c| range (canonical,
          == every existing report CSV). For kf_compB at BS>1 this SUMS the
          BS sequential launches but EXCLUDES inter-launch gaps.
us_span = mean projected NVTX GPU range (includes launch gaps) — canonical
          for judging compB's sequential-launch tax at high BS.
Rivals are PR-arm-normalized: rival_us_local_equiv = rival_us_027 *
(gvr_pr_us_local / gvr_pr_us_027) per cell.
"""
import csv
import io
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
sys.path.insert(0, str(REPORT.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def parse_rep_span(rep):
    out = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_gpu_proj_sum", "--format", "csv",
         "--force-export=true", str(rep)], capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr = next((i for i, r in enumerate(rows)
                if r and r[0] in ("Range", "NVTX Range", "Name")), None)
    if hdr is None:
        return {}
    cols = rows[hdr]
    try:
        i_inst = next(i for i, c in enumerate(cols) if "Instances" in c)
        i_tot = next(i for i, c in enumerate(cols) if "Total" in c)
    except StopIteration:
        return {}
    res = {}
    for r in rows[hdr + 1:]:
        if not r or "|" not in r[0]:
            continue
        try:
            ninst = int(r[i_inst]); tot = float(r[i_tot])
        except (ValueError, IndexError):
            continue
        if ninst:
            res[r[0].lstrip(":")] = tot / ninst / 1e3
    return res


def main():
    out = ["model,isl,N,K,L,BS,hit,cs,op,us,us_span,exact"]
    n_err = 0
    jls = [(jl, HERE / "nsys_reps") for jl in sorted((HERE / "results").glob("bs_*.jsonl"))]
    jls += [(jl, HERE / "nsys_reps_rv") for jl in sorted((HERE / "results_rv").glob("rv_*.jsonl"))]
    for jl, repdir in jls:
        rep = repdir / f"{jl.stem}.nsys-rep"
        kern = parse_rep(rep) if rep.exists() else {}
        span = parse_rep_span(rep) if rep.exists() else {}
        fix = repdir / f"{jl.stem}_fix.nsys-rep"   # solo re-measure top-ups
        if fix.exists():
            kern.update(parse_rep(fix))
            span.update(parse_rep_span(fix))
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            if "error" in r:
                n_err += 1
                print(f"  ERR {jl.stem} {r['op']} BS={r['BS']}: {r['error']}")
                continue
            uc = kern.get(r["range_cold"])
            sc = span.get(r["range_cold"])
            if uc is None:
                n_err += 1
                print(f"  MISSING range {r['range_cold']} in {rep.name}")
                continue
            out.append(f"{r['model']},{r['isl']},{r['N']},{r['K']},{r['L']},"
                       f"{r['BS']},{r['hit']},{r.get('cs','')},{r['op']},"
                       f"{round(uc, 4)},{round(sc, 4) if sc is not None else ''},"
                       f"{r['exact']}")
    (HERE / "kf_bs.csv").write_text("\n".join(out) + "\n")
    print(f"wrote kf_bs.csv: {len(out) - 1} rows ({n_err} omitted)")

    # ---- join with REPORT rival table + anchor check -----------------------
    mine = {}
    for r in csv.DictReader(open(HERE / "kf_bs.csv")):
        mine[(r["model"], r["isl"], r["L"], r["BS"], r["op"])] = r
    riv = {}
    for r in csv.DictReader(open(REPORT / "rival_bs_layers.csv")):
        riv[(r["model"], r["isl"], r["L"], r["BS"], r["op"])] = r

    # ---- joined table: ALL arms measured locally on this node --------------
    # (cross-node PR-arm normalization was found unsafe: local gvr_pr runs
    #  1.07x..1.7x slower than the 027 rival run and the node effect is
    #  asymmetric across arm shapes; REPORT numbers kept only as diagnostics.)
    LOCAL = {"sgl": "sglang_v2", "radix": "radix_cutedsl", "fi": "flashinfer_topk"}
    joined = ["model,isl,N,K,L,BS,hit,compB_us,compB_span,gvr_pr_us,gvr_pr_span,"
              "sgl_us,radix_us,fi_us,sgl_span,radix_span,fi_span,"
              "gvr2_ratio,drift,exact_compB"]
    drifts, stab = [], []
    per_op_drift = {op: [] for op in ("gvr_pr", *LOCAL.values())}
    for key, m in sorted(mine.items(), key=lambda kv: (
            kv[0][0], kv[0][1], int(kv[0][2]), int(kv[0][3]))):
        model, isl, L, BS, op = key
        if op != "kf_compB":
            continue
        g = mine.get((model, isl, L, BS, "gvr_pr"))
        if not g:
            continue
        cell = (model, isl, L, BS)
        rg = riv.get((*cell, "gvr_pr"))
        drift = float(g["us"]) / float(rg["us"]) if rg else None
        if drift:
            drifts.append(drift)
            per_op_drift["gvr_pr"].append(drift)
        g2 = mine.get((*cell, "gvr_pr2"))
        gvr2_ratio = float(g2["us"]) / float(g["us"]) if g2 else None
        if gvr2_ratio:
            stab.append(gvr2_ratio)

        cols_us, cols_span = [], []
        for short, opn in LOCAL.items():
            lr = mine.get((*cell, opn))
            cols_us.append(f"{float(lr['us']):.4f}" if lr else "")
            cols_span.append(lr["us_span"] if lr and lr.get("us_span") else "")
            rr = riv.get((*cell, opn))
            if lr and rr:
                per_op_drift[opn].append(float(lr["us"]) / float(rr["us"]))
        joined.append(
            f"{model},{isl},{m['N']},{m['K']},{L},{BS},{m['hit']},"
            f"{m['us']},{m['us_span']},{g['us']},{g['us_span']},"
            f"{','.join(cols_us)},{','.join(cols_span)},"
            f"{round(gvr2_ratio, 4) if gvr2_ratio else ''},"
            f"{round(drift, 4) if drift else ''},{m['exact']}")
    (HERE / "kf_bs_joined.csv").write_text("\n".join(joined) + "\n")
    print(f"wrote kf_bs_joined.csv: {len(joined) - 1} rows")

    def q(vals, lbl):
        if not vals:
            return
        v = sorted(vals)
        print(f"  {lbl}: n={len(v)} med {st.median(v):.3f} "
              f"p5 {v[int(0.05 * (len(v) - 1))]:.3f} "
              f"p95 {v[int(0.95 * (len(v) - 1))]:.3f}")
    print("pass1-vs-pass2 stability gvr_pr2/gvr_pr:")
    q(stab, "gvr2/gvr")
    print("diagnostics: local(019)/REPORT(027) per op:")
    for opn, vals in per_op_drift.items():
        q(vals, opn)


if __name__ == "__main__":
    main()
