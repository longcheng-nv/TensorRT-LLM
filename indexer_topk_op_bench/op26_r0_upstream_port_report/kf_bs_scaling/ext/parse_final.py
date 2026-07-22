# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse the final campaign sweep -> final_bs.csv + gate evaluation."""
import json
import sys
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

ARM = {0: "v4", 1: "tp3", 2: "tp2"}


def main():
    rows = []
    for jl in sorted((HERE / "results").glob("final_*.jsonl")):
        stem = jl.stem.replace("_L", "_")
        rep = HERE / "nsys_reps" / f"{stem}.nsys-rep"
        kern = parse_rep(rep)
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            r["us"] = kern.get(r["range_cold"])
            rows.append(r)
    cells = {}
    for r in rows:
        cells.setdefault((r["kind"], f"{r['model']}_{r['isl']}_L{r['L']}"),
                         {})[(r["op"], r["BS"])] = r
    csv = ["kind,model,isl,N,K,L,BS,op,pick,us,exact"]
    print(f"{'cell':<20}{'BS':>5} {'gvr_us':>9} {'auto_us':>9} {'arm':>4}"
          f" {'speedup':>8} exact")
    tgt, gen = [], []
    for (kind, tag), d in sorted(cells.items(),
                                 key=lambda kv: (kv[0][0] != "target",
                                                 kv[0][1])):
        for BS in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            g, a = d[("gvr_pr", BS)], d[("auto", BS)]
            for r in (g, a):
                csv.append(
                    f"{r['kind']},{r['model']},{r['isl']},{r['N']},{r['K']},"
                    f"{r['L']},{BS},{r['op']},{r.get('pick', '')},"
                    f"{round(r['us'], 4)},{r['exact']}")
            sp = g["us"] / a["us"]
            (tgt if kind == "target" else gen).append((tag, BS, sp))
            print(f"{tag:<20}{BS:>5} {g['us']:>9.2f} {a['us']:>9.2f}"
                  f" {ARM[a['pick']]:>4} {sp:>8.3f} {a['exact']}")
        print()
    (HERE / "final_bs.csv").write_text("\n".join(csv) + "\n")
    for name, data in (("TARGET", tgt), ("GEN", gen)):
        sps = [x[2] for x in data]
        gm = st.geometric_mean(sps)
        mn = min(data, key=lambda x: x[2])
        nreg = sum(1 for x in sps if x < 1.0)
        print(f"{name}: gm {gm:.3f}x  min {mn[2]:.3f}x @{mn[0]} BS={mn[1]}"
              f"  regressions(<1.0): {nreg}/{len(sps)}")
    allsp = [x[2] for x in tgt + gen]
    print(f"ALL pooled: gm {st.geometric_mean(allsp):.3f}x"
          f" min {min(allsp):.3f}x")
    print("GATES(all cells): gm>=2.0:", st.geometric_mean(allsp) >= 2.0,
          "| min>=1.2:", min(allsp) >= 1.2,
          "| no-reg:", all(x >= 1.0 for x in allsp))
    print("exact:", sum(1 for r in rows if r["exact"]), "/", len(rows))


if __name__ == "__main__":
    main()
