# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse op33 iter0 CRUX ncu CSVs -> per-cell dominant-kernel attribution table."""
import csv
import glob
import os
import sys

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "results", "crux")

COLS = {
    "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "issue": "smsp__issue_active.avg.pct_of_peak_sustained_elapsed",
    "occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "reg": "launch__registers_per_thread",
    "grid": "launch__grid_size",
    "clu": "launch__cluster_size",
    "sec_req": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "t": "gpu__time_duration.sum",
    "name": "Kernel Name",
}


def fnum(x):
    try:
        return float(str(x).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def parse_csv(path):
    """Return dominant-kernel (max total gpu-time across rows) metric dict."""
    # ncu prepends non-CSV preamble (==PROF==, ninja...) to stdout; the real
    # CSV header is the first line starting with "ID".
    with open(path) as f:
        lines = f.readlines()
    hdr = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None)
    if hdr is None:
        return None
    rows = list(csv.DictReader(lines[hdr:]))
    rows = [r for r in rows if r.get(COLS["t"], "").strip() not in ("", None)]
    if not rows:
        return None
    # aggregate gpu-time per kernel name; pick the heaviest
    agg = {}
    for r in rows:
        nm = r.get(COLS["name"], "")[:40]
        agg.setdefault(nm, {"t": 0.0, "r": r})
        agg[nm]["t"] += fnum(r.get(COLS["t"], 0))
    dom = max(agg.values(), key=lambda a: a["t"])
    r = dom["r"]
    tot_t = sum(a["t"] for a in agg.values())
    return {
        "t_dom_us": dom["t"] / 1000.0,
        "t_all_us": tot_t / 1000.0,
        "nkern": len(agg),
        "dram": fnum(r.get(COLS["dram"], 0)),
        "sm": fnum(r.get(COLS["sm"], 0)),
        "issue": fnum(r.get(COLS["issue"], 0)),
        "occ": fnum(r.get(COLS["occ"], 0)),
        "reg": int(fnum(r.get(COLS["reg"], 0))),
        "grid": int(fnum(r.get(COLS["grid"], 0))),
        "clu": int(fnum(r.get(COLS["clu"], 0))),
        "sec_req": fnum(r.get(COLS["sec_req"], 0)),
    }


def main():
    files = sorted(glob.glob(os.path.join(OUTDIR, "*.csv")))
    print(f"{'cell':<44} {'us_all':>7} {'us_dom':>7} {'nk':>3} "
          f"{'dram%':>6} {'sm%':>6} {'iss%':>6} {'occ%':>6} "
          f"{'reg':>4} {'grid':>6} {'clu':>3} {'sec/rq':>6}  bound")
    print("-" * 130)
    for path in files:
        tag = os.path.basename(path)[:-4]
        m = parse_csv(path)
        if m is None:
            print(f"{tag:<44} (no data / failed)")
            continue
        # crude bottleneck label
        if m["dram"] > 55:
            bound = "MEM"
        elif m["issue"] > 60:
            bound = "ISSUE"
        elif m["occ"] < 35 and m["dram"] < 20 and m["issue"] < 45:
            bound = "LATENCY(struct)"
        else:
            bound = "mixed"
        print(f"{tag:<44} {m['t_all_us']:7.2f} {m['t_dom_us']:7.2f} "
              f"{m['nkern']:3d} {m['dram']:6.1f} {m['sm']:6.1f} {m['issue']:6.1f} "
              f"{m['occ']:6.1f} {m['reg']:4d} {m['grid']:6d} {m['clu']:3d} "
              f"{m['sec_req']:6.2f}  {bound}")


if __name__ == "__main__":
    main()
