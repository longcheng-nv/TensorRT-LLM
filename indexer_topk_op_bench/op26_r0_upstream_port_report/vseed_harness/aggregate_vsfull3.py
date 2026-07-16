# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the vseed full-envelope sweep -> vsfull.csv + regression list.

Reads the per-batch jsonls + nsys-reps under RESULTS (default /tmp/gvrval1/
vsfull_results), fills cold-L2 kernel us per cell/arm, writes:
  vsfull.csv          — one row per cell: base/pr/vs us + ratios + exact
  (stdout)            — regression list vs/pr < THRESH and summary geomeans

Run on the sweep machine (needs nsys in PATH). CSV lands in this dir (NFS,
local-only per report policy); nsys-reps stay in /tmp (token hygiene).
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gvrval1/vsfull_results")
THRESH = 0.98   # vs/pr below this = regression

# IMPORTANT: NVTX range names do NOT carry the scenario, so ranges from
# different batches collide by name. Parse each batch's rep separately and
# join it ONLY with that batch's own jsonl (tag == filename stem).
cells = defaultdict(dict)   # key -> arm -> (us, exact, extra)
meta = {}
def _rep_for(jl):
    # jsonl: synth_<scen>_<sweep>_K<K>_<dt> / real_<model>_<sweep>_<dt>
    # rep tag (driver): synth_<sweep>_<scen>_K<K>_<dt> / real_<sweep>_<model>_<dt>
    p = jl.stem.split("_")
    tag = "_".join([p[0], p[2], p[1]] + p[3:])
    return RESULTS / "nsys_reps" / (tag + ".nsys-rep")


for jl in sorted(RESULTS.glob("*.jsonl")):
    rep = _rep_for(jl)
    if not rep.exists():
        print(f"# WARN missing rep for {jl.name}", file=sys.stderr)
        continue
    us = parse_rep(rep)
    for line in jl.read_text().splitlines():
        r = json.loads(line)
        if r.get("error"):
            continue
        key = (r["family"], r["sweep"], r.get("scenario", ""), r.get("model", ""),
               r["K"], r["dtype"], r["N"], r["BS"], r.get("isl", ""))
        u = us.get(r["range_cold"])
        if u:
            cells[key][r["op"]] = (u, r.get("exact"), r.get("launch_cfg", ""))
            meta[key] = r

rows = []
for key, arms in sorted(cells.items()):
    if not all(a in arms for a in ("gvr_base", "gvr_pr", "gvr_vs")):
        continue
    fam, sw, scen, model, K, dt, N, BS, isl = key
    b, p, v = arms["gvr_base"][0], arms["gvr_pr"][0], arms["gvr_vs"][0]
    rows.append(dict(family=fam, sweep=sw, scenario=scen, model=model, K=K,
                     dtype=dt, N=N, BS=BS, isl=isl, hit=meta[key].get("hit"),
                     base=round(b, 3), pr=round(p, 3), vs=round(v, 3),
                     pr_vs_base=round(b / p, 4), vs_vs_base=round(b / v, 4),
                     vs_vs_pr=round(p / v, 4),
                     base_exact=arms["gvr_base"][1], pr_exact=arms["gvr_pr"][1],
                     vs_exact=arms["gvr_vs"][1], launch_cfg=arms["gvr_vs"][2]))

out = _HERE / (sys.argv[2] if len(sys.argv) > 2 else "vsfull.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"# wrote {out} ({len(rows)} cells)", file=sys.stderr)


def geo(vals):
    vals = [x for x in vals if x]
    return math.exp(sum(math.log(x) for x in vals) / len(vals)) if vals else float("nan")


print(f"\n== summary: vs vs pr (cells={len(rows)}) ==")
print(f"geomean vs/pr overall: {geo([r['vs_vs_pr'] for r in rows]):.4f}")
for fam in ("synth", "real"):
    sub = [r for r in rows if r["family"] == fam]
    print(f"  {fam}: {geo([r['vs_vs_pr'] for r in sub]):.4f} ({len(sub)} cells)")
ex_bad = [r for r in rows if r["vs_exact"] is False]
print(f"vs exactness fails: {len(ex_bad)}")
for r in ex_bad[:20]:
    print("  EXACT-FAIL", r)

reg = sorted((r for r in rows if r["vs_vs_pr"] < THRESH), key=lambda r: r["vs_vs_pr"])
print(f"\n== regressions vs/pr < {THRESH}: {len(reg)} cells ==")
for r in reg:
    tag = (f"{r['family']}/{r['scenario'] or r['model']}/K{r['K']}/{r['dtype']}"
           f"/N{r['N']}/BS{r['BS']}{('/' + r['isl']) if r['isl'] else ''}")
    print(f"  {tag:>52} vs/pr={r['vs_vs_pr']:.3f} vs/base={r['vs_vs_base']:.3f} "
          f"(pr {r['pr']}us -> vs {r['vs']}us)")

wins = sorted((r for r in rows if r["vs_vs_pr"] > 1.05), key=lambda r: -r["vs_vs_pr"])
print(f"\n== wins vs/pr > 1.05: {len(wins)} cells (top 15) ==")
for r in wins[:15]:
    tag = (f"{r['family']}/{r['scenario'] or r['model']}/K{r['K']}/{r['dtype']}"
           f"/N{r['N']}/BS{r['BS']}{('/' + r['isl']) if r['isl'] else ''}")
    print(f"  {tag:>52} vs/pr={r['vs_vs_pr']:.3f} vs/base={r['vs_vs_base']:.3f}")
