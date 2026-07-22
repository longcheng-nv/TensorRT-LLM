# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-cell summary of the NCU P4 instruction accounting.

Inputs: ncu_reps/p4_<cell>_segs.json (parse_ncu_p4.py) + p4pipe_full.csv
(Experiment A cycles). Outputs ncu_p4_summary.json:
  per cell x P4 sub-stage: inst_executed + share, warp-stall-sample share
  (~time proxy), opcode-class mix, key-op counts (ATOMS/LDS/STS/BAR/MAPA...),
  and the Exp-A cycle share for the same stage (cross-validation);
  per cell: kernel-wide stall-reason table from the details page.
"""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUB = ["p4_peer_wait", "p4_dsmem_gather", "p4_minmax", "p4_coarse_hist",
       "p4_coarse_search", "p4_fine", "p4_scatter", "p4_tail"]
KEYOPS = ["ATOMS", "LDS", "STS", "BAR", "MAPA", "LDG", "STG", "LDSM",
          "VOTEU", "SHFL", "CS2R", "NANOSLEEP"]


def key_op_counts(top):
    out = {}
    for mn, n in top.items():
        base = mn.split(".")[0]
        if base in KEYOPS:
            out[base] = out.get(base, 0) + n
    return out


def stall_table(raw_csv):
    """Kernel-wide warp-stall breakdown (cycles per issue-active) from the
    raw page export."""
    if not raw_csv.exists():
        return {}
    lines = open(raw_csv).read().splitlines()
    if len(lines) < 3:
        return {}
    hdr = next(csv.reader([lines[0]]))
    val = next(csv.reader([lines[2]]))  # line 1 = units row
    st = {}
    for k, v in zip(hdr, val):
        m = re.match(
            r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio",
            k)
        if m:
            try:
                st[m.group(1)] = float(v)
            except ValueError:
                pass
    return dict(sorted(st.items(), key=lambda kv: -kv[1])[:10])


def main():
    exp_a = {r["uuid"]: r for r in csv.DictReader(open(HERE / "p4pipe_full.csv"))}
    cells = {}
    for f in sorted((HERE / "ncu_reps").glob("p4_*_segs.json")):
        cell = re.sub(r"^p4_|_segs\.json$", "", f.name)
        S = json.load(open(f))
        segs = {s["label"]: s for s in S["segments"]}
        a = exp_a.get(cell, {})
        tot_inst = sum(s["inst_executed"] for s in S["segments"])
        samp_col = None
        for s in S["segments"]:
            for k in s["column_sums"]:
                if "Warp Stall Sampling (All" in k:
                    samp_col = k
                    break
            if samp_col:
                break
        tot_samp = sum(s["column_sums"].get(samp_col, 0.0)
                       for s in S["segments"]) if samp_col else 0.0
        p4_cyc = float(a.get("cyc_p4_select", 0) or 0)
        rec = dict(
            cs=int(a["cs"]) if a else None, N=int(a["N"]) if a else None,
            K=int(a["K"]) if a else None, hit=float(a["hit"]) if a else None,
            n_stamps=S["n_stamps"], total_inst=tot_inst,
            stalls_kernelwide=stall_table(
                HERE / "ncu_reps" / f"p4_{cell}_raw.csv"),
            stages={},
        )
        for st in SUB:
            s = segs.get(st)
            if not s:
                continue
            samp = s["column_sums"].get(samp_col, 0.0) if samp_col else 0.0
            cyc = float(a.get(f"cyc_{st}", 0) or 0) if a else 0.0
            rec["stages"][st] = dict(
                inst=s["inst_executed"],
                inst_share=round(s["inst_executed"] / tot_inst, 4) if tot_inst else 0,
                stall_samples=samp,
                samp_share=round(samp / tot_samp, 4) if tot_samp else 0,
                expA_cyc_share_kernel=round(
                    cyc / float(a["window_cyc"]), 4) if a else None,
                expA_share_of_p4=round(cyc / p4_cyc, 4) if p4_cyc else None,
                opcode_class=s["opcode_class"],
                key_ops=key_op_counts(s["top_opcodes"]),
                top5=dict(list(s["top_opcodes"].items())[:5]),
            )
        cells[cell] = rec

    json.dump(cells, open(HERE / "ncu_p4_summary.json", "w"), indent=1)
    print(f"{len(cells)} cells -> ncu_p4_summary.json")
    for cell, rec in cells.items():
        print(f"\n== {cell} cs={rec['cs']} N={rec['N']} K={rec['K']} "
              f"hit={rec['hit']}")
        for st, v in rec["stages"].items():
            ko = " ".join(f"{k}={v['key_ops'][k]:.0f}" for k in
                          ("ATOMS", "LDS", "STS", "BAR", "MAPA")
                          if k in v["key_ops"])
            print(f"  {st.replace('p4_',''):>13s} inst={v['inst']:>9.0f} "
                  f"samp%={100*v['samp_share']:5.1f} "
                  f"cyc%P4={100*(v['expA_share_of_p4'] or 0):5.1f}  {ko}")


if __name__ == "__main__":
    main()
