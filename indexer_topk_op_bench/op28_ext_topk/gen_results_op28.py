#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op28 — merge parsed results into CSV + comparison summary vs the op22
REPORT baselines.

Outputs:
  op28_bs_data.csv / op28_seqlen_data.csv / op28_bs_hugeN_data.csv
      per-cell cold/warm us for the 6 op28 arms (same-node, node 027)
      + us_span columns for the 2-kernel sglang_v2 cells
      + anchor-transferred op22rr baseline columns (op21_hls / op25_hls /
        op26_r0auto etc from op22rr_*_data.csv, rescaled per cell by
        gvr_cutedsl(orig)/gvr_cutedsl(op28) so cross-node µs are comparable)
  RESULTS_SUMMARY.md   geomean verdict tables

Usage: python3 gen_results_op28.py [<out_root>] default ../results_b200_op28
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP22 = HERE.parents[0] / "op22_temporal_fixed_hr_bench"

ARMS = ["gvr_cutedsl", "radix_cutedsl", "sglang_streaming",
        "sglang_v2", "flashinfer_topk", "flashinfer_topk_i32"]
# op22rr baseline columns to carry over (anchor-transferred)
RR_ARMS = ["op21_hls", "op25_hls", "op26_r0auto", "gvr_multicta_cutedsl",
           "radix_single_cuda", "radix_multi_cuda"]


def load_op28(out_root):
    """{(scenario, sweep, K, N, BS): {op: rec}}"""
    data = defaultdict(dict)
    for scen_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        for sub in ("seqlen_sweep", "bs_scaling", "bs_hugeN"):
            p = scen_dir / sub / "results.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" in r and "us" not in r:
                    continue
                key = (r["scenario"], r["sweep"], r["K"], r["N"], r["BS"])
                data[key][r["op"]] = r
    return data


def load_rr_csv(fn):
    """{(scenario, K, N, BS): row} from op22rr CSVs (fp32 rows only)."""
    out = {}
    p = OP22 / fn
    if not p.exists():
        return out
    for row in csv.DictReader(open(p)):
        if row.get("dtype") != "fp32":
            continue
        out[(row["scenario"], int(row["K"]), int(row["N"]),
             int(row["BS"]))] = row
    return out


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE.parents[0] / "results_b200_op28"
    data = load_op28(out_root)
    rr_bs = load_rr_csv("op22rr_bs_data.csv")
    rr_seq = load_rr_csv("op22rr_seqlen_data.csv")

    # ---- per-sweep CSVs ----
    sweeps = {"bs": "op28_bs_data.csv", "seqlen": "op28_seqlen_data.csv",
              "bs_hugeN": "op28_bs_hugeN_data.csv"}
    for sweep, fn in sweeps.items():
        keys = sorted(k for k in data if k[1] == sweep)
        if not keys:
            continue
        head = ["scenario", "K", "N", "BS", "hit_rate"]
        for a in ARMS:
            head += [f"{a}_cold_us", f"{a}_warm_us"]
        head += ["sglang_v2_cold_span_us", "sglang_v2_warm_span_us",
                 "anchor_ratio"]  # gvr(orig)/gvr(op28) per cell
        for a in RR_ARMS:
            head += [f"rr_{a}_cold_us_adj"]
        rows = []
        for key in keys:
            scen, _, K, N, BS = key
            recs = data[key]
            row = [scen, K, N, BS,
                   next((r.get("hit_rate") for r in recs.values()), None)]
            for a in ARMS:
                r = recs.get(a, {})
                row += [round(r["us_cold"], 3) if "us_cold" in r else None,
                        round(r["us_warm"], 3) if "us_warm" in r else None]
            sv = recs.get("sglang_v2", {})
            row += [round(sv["us_cold_span"], 3) if "us_cold_span" in sv else None,
                    round(sv["us_warm_span"], 3) if "us_warm_span" in sv else None]
            # anchor transfer
            rr = (rr_bs if sweep == "bs" else rr_seq).get((scen, K, N, BS))
            gvr28 = recs.get("gvr_cutedsl", {}).get("us_cold")
            ar = None
            if rr and gvr28:
                try:
                    gvr_orig = float(rr["gvr_cutedsl_cold_us"])
                    ar = gvr_orig / gvr28
                except (KeyError, ValueError, ZeroDivisionError):
                    ar = None
            row.append(round(ar, 4) if ar else None)
            for a in RR_ARMS:
                v = None
                if rr and ar:
                    try:
                        v = float(rr[f"{a}_cold_us"]) / ar  # onto op28 node scale
                    except (KeyError, ValueError):
                        v = None
                row.append(round(v, 3) if v else None)
            rows.append(row)
        with open(HERE / fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rows)
        print(f"wrote {fn} ({len(rows)} rows)")

    # ---- summary ----
    lines = ["# op28 — LATEST SGLang v2 & FlashInfer top_k vs in-tree ops",
             "",
             f"cells loaded: {len(data)}  (node umbriel-b200-027, fp32, "
             "op22rr byte-identical bundles, nsys cold-L2 20 reps / warm 50)",
             ""]
    anchor_ratios = []
    for key, recs in data.items():
        scen, sweep, K, N, BS = key
        rr = (rr_bs if sweep == "bs" else rr_seq).get((scen, K, N, BS))
        g = recs.get("gvr_cutedsl", {}).get("us_cold")
        if rr and g:
            try:
                anchor_ratios.append(float(rr["gvr_cutedsl_cold_us"]) / g)
            except (KeyError, ValueError):
                pass
    if anchor_ratios:
        s = sorted(anchor_ratios)
        lines += [f"anchor drift gvr_cutedsl orig/op28: med "
                  f"{s[len(s)//2]:.4f}  p10 {s[len(s)//10]:.4f}  "
                  f"p90 {s[9*len(s)//10]:.4f}  n={len(s)}", ""]

    def ratio_table(num_arm, den_arm, span_for_num=False):
        rows = []
        for scen in ("best", "worst", "real"):
            for K in (512, 1024, 2048):
                rs = []
                for key, recs in data.items():
                    if key[0] != scen or key[2] != K:
                        continue
                    a = recs.get(num_arm, {})
                    b = recs.get(den_arm, {})
                    ua = a.get("us_cold_span") if span_for_num and \
                        "us_cold_span" in a else a.get("us_cold")
                    ub = b.get("us_cold")
                    if ua and ub:
                        rs.append(ub / ua)   # >1 => num_arm faster
                g = geomean(rs)
                rows.append((scen, K, g, len(rs)))
        return rows

    pairs = [
        ("sglang_v2", "sglang_streaming", "SGLang v2 vs OLD StreamingTopK"),
        ("sglang_v2", "gvr_cutedsl", "SGLang v2 vs GVR(cuteDSL) baseline"),
        ("sglang_v2", "radix_cutedsl", "SGLang v2 vs Radix(cuteDSL)"),
        ("flashinfer_topk", "gvr_cutedsl", "FlashInfer top_k vs GVR(cuteDSL)"),
        ("flashinfer_topk", "radix_cutedsl", "FlashInfer top_k vs Radix(cuteDSL)"),
        ("flashinfer_topk", "sglang_v2", "FlashInfer top_k vs SGLang v2"),
        ("flashinfer_topk_i32", "flashinfer_topk", "FlashInfer i32-minimal vs public API"),
    ]
    for num, den, title in pairs:
        lines += [f"## {title}  (cold-L2 kernel-sum, t({den})/t({num}), "
                  ">1 => {} faster; geomean over all cells)".format(num), ""]
        lines += ["| scenario | K | geomean | n |", "|---|---|---|---|"]
        for scen, K, g, n in ratio_table(num, den):
            lines.append(f"| {scen} | {K} | "
                         f"{'%.3f' % g if g else '—'} | {n} |")
        lines.append("")

    # ---- vs anchor-transferred op22rr production arms ----
    def rr_ratio_table(num_arm, rr_arm):
        """t(rr_arm, anchor-transferred)/t(num_arm) — >1 => num_arm faster."""
        out = []
        for scen in ("best", "worst", "real"):
            for K in (512, 1024, 2048):
                rs = []
                for key, recs in data.items():
                    scen_k, sweep, Kk, N, BS = key
                    if scen_k != scen or Kk != K or sweep == "bs_hugeN":
                        continue
                    rr = (rr_bs if sweep == "bs" else rr_seq).get(
                        (scen, K, N, BS))
                    g = recs.get("gvr_cutedsl", {}).get("us_cold")
                    a = recs.get(num_arm, {}).get("us_cold")
                    if not (rr and g and a):
                        continue
                    try:
                        gvr_orig = float(rr["gvr_cutedsl_cold_us"])
                        rv = float(rr[f"{rr_arm}_cold_us"]) * g / gvr_orig
                    except (KeyError, ValueError, ZeroDivisionError):
                        continue
                    rs.append(rv / a)
                out.append((scen, K, geomean(rs), len(rs)))
        return out

    for num in ("sglang_v2", "flashinfer_topk"):
        for rr_arm in ("op21_hls", "op26_r0auto", "op25_hls",
                       "gvr_multicta_cutedsl"):
            lines += [f"## {num} vs {rr_arm} (op22rr arm, anchor-transferred "
                      f"onto node-027 scale; >1 => {num} faster)", "",
                      "| scenario | K | geomean | n |", "|---|---|---|---|"]
            for scen, K, g, n in rr_ratio_table(num, rr_arm):
                lines.append(f"| {scen} | {K} | "
                             f"{'%.3f' % g if g else '—'} | {n} |")
            lines.append("")

    lines += [
        "## Caveats",
        "- canonical `us` = per-range kernel-time SUM (comparable to all",
        "  prior report numbers). sglang_v2's persistent-cluster path",
        "  (N>=131072, 30<BS<=512) launches 2 PDL kernels: sum can",
        "  double-count overlap (observed up to 1.8x at N=262144 BS=64 where",
        "  span=0.56x sum) or miss the inter-kernel gap (span up to 1.2x sum",
        "  at N=131072); `*_span_us` columns carry the honest wall-clock.",
        "- sglang_v2 `topk_plan` runs untimed (production: once per step,",
        "  reused across ~61 layers; measured ~7us wall => ~0.11us/layer).",
        "- flashinfer public top_k returns (values fp32, indices int64) --",
        "  slightly larger output traffic than the int32-only in-tree",
        "  contract; flashinfer_topk_i32 is the contract-matched variant",
        "  (~2-5% faster).",
    ]

    (HERE / "RESULTS_SUMMARY.md").write_text("\n".join(lines))
    print("wrote RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()
