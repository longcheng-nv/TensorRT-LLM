#!/usr/bin/env python3
"""Iteration 2 (GA7 host precursor): BS=1..512 attribution + rho map.

Host-only sweep over the SCOPE_DSV4_MOE_BS1-512 shape/format/distribution matrix using
the Phase-0 harness. Produces the per-shape error-budget + trust-gate (rho) + flip-risk
table that the silicon iterations (GA5/GA7) will validate against measured latency/SOL.

This is the loop in action: PROPOSE (map the BS arc) -> EXECUTE (harness.sweep) ->
EVALUATE (rho / budget / regime) -> DECIDE. Every number is owned by harness.measure.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from harness import MeasureRequest, PrecisionPolicy, Shape, measure
from harness import _regime as _regime_true

DSV4 = dict(hidden=4096, moe_intermediate=2048, top_k=6)

# decode BS -> M_total = BS * top_k ; EP=32 -> 8 local experts
BS_LIST = [1, 8, 32, 64, 128, 256, 512]
EP_LOCAL = 8


def shapes_for_bs(bs: int):
    m = bs * DSV4["top_k"]
    return [
        Shape(f"decode_bs{bs}_fc1", "FC1", M_total=m, K=4096, N=4096, n_groups=EP_LOCAL),
        Shape(f"decode_bs{bs}_fc2", "FC2", M_total=m, K=2048, N=4096, n_groups=EP_LOCAL),
    ]


FORMATS = ["nvf4", "mxf8", "mxf4"]
DISTS = ["normal", "outlier_channel"]
SEEDS = [42, 137]

# DOCUMENTED SHRINK (not a silent cap): the host twin's rho / budget / flip_risk are
# scale-invariant *ratios* — by the law of large numbers they are essentially
# independent of M, K, N in distribution. So for the host metric sweep we compute them
# at reduced dims to keep the loop seconds-scale (PROGRAM.md "shrink shapes, not rigor").
# The TRUE M_total/K/N are preserved in the reported Shape and drive the roofline regime
# and the silicon GA5/GA7 latency/SOL iteration, where dimensions matter for real.
M_CAP, K_CAP, N_CAP = 128, 512, 512


def main():
    print(
        f"[shrink] host metric sweep computes ratios at M≤{M_CAP},K≤{K_CAP},N≤{N_CAP} "
        f"(ratios are dim-invariant; true dims drive regime + silicon GA5). seeds={SEEDS}"
    )
    rows = []
    for bs in BS_LIST:
        for shp_full in shapes_for_bs(bs):
            shp = Shape(
                shp_full.name,
                shp_full.gemm,
                M_total=min(shp_full.M_total, M_CAP),
                K=min(shp_full.K, K_CAP),
                N=min(shp_full.N, N_CAP),
                n_groups=shp_full.n_groups,
                phase=shp_full.phase,
            )
            for fmt in FORMATS:
                for dist in DISTS:
                    # average over seeds (deterministic per seed)
                    accs = {
                        "measured_rel": [],
                        "rho": [],
                        "rho_xt": [],
                        "flip": [],
                        "topbudget": [],
                    }
                    top_src = None
                    for seed in SEEDS:
                        r = measure(
                            MeasureRequest(
                                shape=shp,
                                policy=PrecisionPolicy(fmt, out_dtype="bf16"),
                                distribution=dist,
                                ref_dtype="fp64",
                                seed=seed,
                            )
                        )
                        rxt = measure(
                            MeasureRequest(
                                shape=shp,
                                policy=PrecisionPolicy(fmt, out_dtype="bf16"),
                                distribution=dist,
                                ref_dtype="fp64",
                                seed=seed,
                                escalation="cross_term",
                            )
                        )
                        accs["measured_rel"].append(r.measured_rel)
                        accs["rho"].append(r.rho)
                        accs["rho_xt"].append(rxt.rho)
                        accs["flip"].append(r.flip_risk or 0.0)
                        accs["topbudget"].append(r.budget_per_source[0].budget)
                        top_src = r.budget_per_source[0].source

                    def avg(k):
                        return sum(accs[k]) / len(accs[k])

                    rows.append(
                        dict(
                            bs=bs,
                            gemm=shp.gemm,
                            M_total_true=shp_full.M_total,
                            M_computed=shp.M_total,
                            fmt=fmt,
                            dist=dist,
                            regime=_regime_true(shp_full),
                            top_source=top_src,
                            measured_rel=avg("measured_rel"),
                            rho=avg("rho"),
                            rho_crossterm=avg("rho_xt"),
                            flip_risk=avg("flip"),
                            top_budget=avg("topbudget"),
                        )
                    )

    outdir = Path(__file__).resolve().parent / "results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / "iter2_bs_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # findings
    print("=" * 100)
    print("Iteration 2 — DSV4 MoE BS=1..512 attribution + rho map (host twin, fp64 ref)")
    print("=" * 100)
    print(
        f"{'shape':<20}{'fmt':<6}{'dist':<18}{'regime':<9}{'meas_rel':>10}{'rho':>9}{'rho_xt':>9}{'flip':>7}{'top_src':>16}"
    )
    for r in rows:
        if r["dist"] == "normal":  # print benign rows; full matrix in CSV
            print(
                f"{r['gemm'] + ' bs' + str(r['bs']):<20}{r['fmt']:<6}{r['dist']:<18}"
                f"{r['regime']:<9}{r['measured_rel']:>10.2e}{r['rho']:>9.2e}"
                f"{r['rho_crossterm']:>9.1e}{r['flip_risk']:>7.2f}{r['top_source']:>16}"
            )

    # auto-findings (machine-derived, no model assertions)
    fc1 = [r for r in rows if r["gemm"] == "FC1"]
    fc2 = [r for r in rows if r["gemm"] == "FC2"]
    [r["rho_crossterm"] for r in fc2 if r["fmt"] == "nvf4"]
    findings = {
        "fc2_linear_crossterm_closes": max(r["rho_crossterm"] for r in fc2 if r["fmt"] == "nvf4"),
        "fc1_swiglu_rho_max": max(r["rho"] for r in fc1),
        "fc1_flip_risk_max": max(r["flip_risk"] for r in fc1),
        "regimes_seen": sorted({r["regime"] for r in rows}),
        "outlier_vs_benign_rho_fc2_nvf4": {
            "benign": next(
                r["rho"]
                for r in fc2
                if r["fmt"] == "nvf4" and r["dist"] == "normal" and r["bs"] == 512
            ),
            "outlier": next(
                r["rho"]
                for r in fc2
                if r["fmt"] == "nvf4" and r["dist"] == "outlier_channel" and r["bs"] == 512
            ),
        },
    }
    (outdir / "iter2_findings.json").write_text(json.dumps(findings, indent=2, default=float))
    print("\nAuto-findings (harness-owned):")
    print(json.dumps(findings, indent=2, default=float))
    print(f"\nwrote {outdir / 'iter2_bs_sweep.csv'} and iter2_findings.json")


if __name__ == "__main__":
    main()
