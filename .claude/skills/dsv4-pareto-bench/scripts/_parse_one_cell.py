#!/usr/bin/env python3
"""Parse one bench log and append a row to results/per_cell.csv.

Idempotent: if the cell_id already has a row in the CSV, this overwrites
it (uncommon but possible when re-running after fixing a bug).

Called by 01_run_one_cell.sh after trtllm-bench finishes successfully.
"""

import argparse
import collections
import csv
import os
import re

PERF_PATTERNS = {
    "req_per_sec":          r"Request Throughput \(req/sec\):\s+([0-9.E+\-]+)",
    "total_output_tok_s":   r"Total Output Throughput \(tokens/sec\):\s+([0-9.E+\-]+)",
    "total_token_tok_s":    r"Total Token Throughput \(tokens/sec\):\s+([0-9.E+\-]+)",
    "total_latency_ms":     r"Total Latency \(ms\):\s+([0-9.E+\-]+)",
    "avg_req_latency_ms":   r"Average request latency \(ms\):\s+([0-9.E+\-]+)",
    "per_user_output_ctx":  r"Per User Output Throughput \[w/ ctx\] \(tps/user\):\s+([0-9.E+\-]+)",
    "per_gpu_output_tok_s": r"Per GPU Output Throughput \(tps/gpu\):\s+([0-9.E+\-]+)",
    "ttft_avg_ms":          r"Average time-to-first-token \[TTFT\] \(ms\):\s+([0-9.E+\-]+)",
    "tpot_avg_ms":          r"Average time-per-output-token \[TPOT\] \(ms\):\s+([0-9.E+\-]+)",
    "per_user_output_st":   r"Per User Output Speed \(tps/user\):\s+([0-9.E+\-]+)",
    "total_energy_j":       r"Total Energy \(J\):\s+([0-9.E+\-]+)",
    "tps_per_w":            r"Output Tokens per Second per Watt \(tps/W\):\s+([0-9.E+\-]+)",
    "avg_gpu_power_w":      r"Average GPU Power \(W\):\s+([0-9.E+\-]+)",
}

OUTPUT_FIELDS = [
    "cell_id", "ISL", "OSL", "BS", "MTP", "Mode", "GVR",
    *PERF_PATTERNS.keys(),
    "dispatch_path", "n_scheme_x_lines", "numRows_distribution",
    "log_relpath",
]


def parse_perf(text: str) -> dict:
    return {
        k: float(m.group(1)) if (m := re.search(p, text)) else None
        for k, p in PERF_PATTERNS.items()
    }


def parse_dispatch(text: str) -> tuple[str, int, str]:
    """Return (dispatch_path, n_scheme_x_lines, numRows_distribution_str).

    `dispatch_path` ∈ {"Heuristic", "Radix", "Mixed", "Unknown"}.
    `numRows_distribution` is a 'numRows=N:count,...' string.
    Requires TRTLLM_SCHEMEX_DEBUG=1 to have been set during the bench run.
    """
    scheme_lines = re.findall(r"\[Scheme X\][^\n]*-> (\w+) path", text)
    nrows = collections.Counter(re.findall(r"numRows=(\d+)", text))

    n_lines = len(scheme_lines)
    distribution = ",".join(f"{nr}:{c}" for nr, c in sorted(nrows.items(), key=lambda x: int(x[0])))

    paths = set(scheme_lines)
    if not paths:
        dispatch = "Unknown"
    elif paths == {"Heuristic"}:
        dispatch = "Heuristic"
    elif paths == {"Radix"}:
        dispatch = "Radix"
    else:
        dispatch = "Mixed"
    return dispatch, n_lines, distribution


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell-id", required=True)
    p.add_argument("--isl", type=int, required=True)
    p.add_argument("--osl", type=int, required=True)
    p.add_argument("--bs", type=int, required=True)
    p.add_argument("--mtp", type=int, required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--gvr", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    text = open(args.log, errors="replace").read()
    row = {
        "cell_id":  args.cell_id,
        "ISL":      args.isl,
        "OSL":      args.osl,
        "BS":       args.bs,
        "MTP":      args.mtp,
        "Mode":     args.mode,
        "GVR":      args.gvr,
        "log_relpath": os.path.relpath(args.log, start=os.path.dirname(args.out) + "/.."),
    }
    row.update(parse_perf(text))
    dispatch, n_lines, distribution = parse_dispatch(text)
    row["dispatch_path"] = dispatch
    row["n_scheme_x_lines"] = n_lines
    row["numRows_distribution"] = distribution

    existing: list[dict] = []
    write_header = not os.path.exists(args.out)
    if not write_header:
        with open(args.out) as f:
            r = csv.DictReader(f)
            existing = [r0 for r0 in r if r0.get("cell_id") != args.cell_id]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        for r0 in existing:
            w.writerow({k: r0.get(k, "") for k in OUTPUT_FIELDS})
        w.writerow(row)

    print(f"[parse] {args.cell_id}: req/s={row.get('req_per_sec')} "
          f"tpot_ms={row.get('tpot_avg_ms')} dispatch={dispatch} "
          f"scheme_x_lines={n_lines}")


if __name__ == "__main__":
    main()
