# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse the compB BS>1 extension minimal-validation sweep -> ext_bs.csv +
verdict table. All arms local on this node (umbriel-b200-039), paired per
cell. us = mean cold-L2 kernel-sum (canonical); us_span = NVTX GPU-projected
span (includes launch gaps -> the sequential/chunked-wave tax)."""
import csv
import io
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent.parent
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
    out = ["model,isl,N,K,L,BS,kind,hit,cs,op,path,team,cap,rpw,waves,"
           "us,us_span,exact"]
    n_err = 0
    for jl in sorted((HERE / "results").glob("ext_*.jsonl")):
        rep = HERE / "nsys_reps" / f"{jl.stem}.nsys-rep"
        kern = parse_rep(rep) if rep.exists() else {}
        span = parse_rep_span(rep) if rep.exists() else {}
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
            out.append(
                f"{r['model']},{r['isl']},{r['N']},{r['K']},{r['L']},"
                f"{r['BS']},{r['kind']},{r['hit']},{r.get('cs', '')},{r['op']},"
                f"{r.get('path', '')},{r.get('team', '')},{r.get('cap', '')},"
                f"{r.get('rpw', '')},{r.get('waves', '')},"
                f"{round(uc, 4)},{round(sc, 4) if sc is not None else ''},"
                f"{r['exact']}")
    (HERE / "ext_bs.csv").write_text("\n".join(out) + "\n")
    print(f"wrote ext_bs.csv: {len(out) - 1} rows ({n_err} omitted)\n")

    rows = list(csv.DictReader(open(HERE / "ext_bs.csv")))
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["isl"], r["L"]), {})[
            (r["op"], int(r["BS"]))] = r

    print(f"{'cell':<18}{'BS':>6} {'ext_us':>9} {'seq_us':>9} {'gvr_us':>9}"
          f" {'ext/seq x':>10} {'ext/gvr x':>10} {'span tax':>9}"
          f" {'waves':>6} exact")
    ratios_vs_gvr = {}
    for (m, isl, L), d in sorted(cells.items()):
        tag = f"{m}_{isl}_L{L}"
        bss = sorted({bs for op, bs in d if op == "kf_compB_ext"})
        for bs in bss:
            e = d.get(("kf_compB_ext", bs))
            s = d.get(("kf_compB", bs))
            g = d.get(("gvr_pr", bs))
            if not e:
                continue
            eu = float(e["us"])
            su = float(s["us"]) if s else None
            gu = float(g["us"]) if g else None
            tax = (float(e["us_span"]) / eu) if e["us_span"] else None
            if gu:
                ratios_vs_gvr.setdefault((e["kind"], bs), []).append(gu / eu)
            print(f"{tag:<18}{bs:>6} {eu:>9.2f} "
                  f"{su if su is None else format(su, '>9.2f')} "
                  f"{gu if gu is None else format(gu, '>9.2f')} "
                  f"{su / eu if su else 0:>10.3f} {gu / eu if gu else 0:>10.3f}"
                  f" {tax or 0:>9.3f} {e['waves'] or '':>6} {e['exact']}")
        print()

    print("pooled geomean ext speedup vs gvr_pr by (kind, BS):")
    for (kind, bs), v in sorted(ratios_vs_gvr.items()):
        gm = st.geometric_mean(v)
        wins = sum(1 for x in v if x >= 1.0)
        print(f"  kind={kind} BS={bs:<5} gm {gm:.3f}x  ({wins}/{len(v)} >=1.0)")


if __name__ == "__main__":
    main()
