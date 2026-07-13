#!/usr/bin/env python3
# op26 iter7 预研 — aggregate ncu reports from prof_lowbs_cell.py captures.
# Per report: SOL/duration/occupancy/stall breakdown + warp-stall sampling
# aggregated by SASS source location (top barrier sites).
# Usage: python3 analyze_iter7_ncu.py [results_iter7_prof]
import csv
import io
import os
import subprocess
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results_iter7_prof"
ENV = {k: v for k, v in os.environ.items()
       if k not in ("GITHUB_TOKEN", "HF_TOKEN")}

DETAIL_METRICS = [
    ("gpu__time_duration.sum", "dur_us", 1),             # csv already us
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm_pct", 1),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
     "mem_pct", 1),
    ("launch__occupancy_per_cluster_gpu", "clu_occ_pct", 100),
    ("smsp__average_warp_latency_per_inst_issued.ratio", "cyc_per_inst", 1),
    ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
     "stall_barrier", 1),
    ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
     "stall_longsb", 1),
    ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
     "stall_shortsb", 1),
    ("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
     "stall_wait", 1),
    ("smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
     "stall_membar", 1),
    ("smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
     "stall_branch", 1),
    ("smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
     "stall_drain", 1),
    ("smsp__average_warps_issue_stalled_selected_per_issue_active.ratio",
     "stall_selected", 1),
    ("smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
     "stall_notsel", 1),
    ("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
     "stall_math", 1),
]


def run(args):
    return subprocess.run(args, capture_output=True, text=True,
                          env=ENV).stdout


def raw_metrics(rep):
    out = run(["ncu", "--import", rep, "--page", "raw", "--csv"])
    rows = list(csv.reader(io.StringIO(out)))
    if len(rows) < 3:
        return []
    hdr = rows[0]
    per_launch = []
    for r in rows[1:]:
        if len(r) != len(hdr) or r[0].startswith("=="):
            continue
        d = dict(zip(hdr, r))
        if not d.get("ID", "").strip().isdigit():   # units row
            continue
        m = {}
        for metric, name, scale in DETAIL_METRICS:
            v = d.get(metric, "")
            try:
                m[name] = float(v.replace(",", "")) * scale
            except (ValueError, AttributeError):
                m[name] = None
            m["kernel"] = d.get("Kernel Name", "?")[:60]
        per_launch.append(m)
    return per_launch


def stall_by_line(rep, top=12):
    out = run(["ncu", "--import", rep, "--page", "source", "--csv"])
    rows = list(csv.reader(io.StringIO(out)))
    hdr_i = next((i for i, r in enumerate(rows)
                  if r and r[0] == "Address"), None)
    if hdr_i is None:
        return []
    hdr = rows[hdr_i]
    cand = [i for i, h in enumerate(hdr) if "Stall Sampling (All" in h]
    if not cand:
        return []
    i_samp = cand[0]
    i_src = hdr.index("Source") if "Source" in hdr else None
    agg = defaultdict(int)
    for r in rows[hdr_i + 1:]:
        if len(r) <= i_samp:
            continue
        try:
            s = int(r[i_samp].replace(",", "") or 0)
        except ValueError:
            continue
        if s <= 0:
            continue
        src = r[i_src] if i_src is not None and len(r) > i_src else "?"
        agg[src] += s
    total = sum(agg.values()) or 1
    return [(src, n, 100.0 * n / total)
            for src, n in sorted(agg.items(), key=lambda kv: -kv[1])[:top]]


def main():
    reps = sorted(f for f in os.listdir(ROOT) if f.endswith(".ncu-rep"))
    for f in reps:
        rep = os.path.join(ROOT, f)
        print(f"\n=== {f} ===")
        launches = raw_metrics(rep)
        if not launches:
            print("  (no raw metrics)")
            continue
        for m in launches:
            stalls = {k: v for k, v in m.items()
                      if k.startswith("stall_") and v}
            top3 = sorted(stalls.items(), key=lambda kv: -kv[1])[:4]
            frac = ""
            if m.get("cyc_per_inst"):
                frac = " | " + " ".join(
                    f"{k[6:]}={v:.1f}({100*v/m['cyc_per_inst']:.0f}%)"
                    for k, v in top3)
            def fmt(v, p=2):
                return f"{v:.{p}f}" if v is not None else "?"
            print(f"  {m['kernel'][:48]:48s} dur={fmt(m['dur_us'])}us "
                  f"sm={fmt(m['sm_pct'], 1)}% mem={fmt(m['mem_pct'], 1)}% "
                  f"cyc/inst={fmt(m['cyc_per_inst'], 1)}{frac}")
        print("  -- warp-stall sampling by source (whole report) --")
        for src, n, pct in stall_by_line(rep):
            print(f"    {pct:5.1f}%  {n:>8d}  {src[:90]}")


if __name__ == "__main__":
    main()
