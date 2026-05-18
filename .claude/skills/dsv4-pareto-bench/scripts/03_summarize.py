#!/usr/bin/env python3
"""Aggregate per_cell.csv into a human-readable REPORT.md + key figures.

Reads PERFDIR/results/per_cell.csv (path overridable via env PERFDIR);
writes:
  PERFDIR/results/REPORT.md
  PERFDIR/results/figures/dtpot_isl_bs_TEP_M3.png   (paired ΔTPOT% heatmap)
  PERFDIR/results/figures/dtpot_isl_bs_DEP_M3.png

Safe to re-run as more cells finish.
"""

import collections
import csv
import os

PERFDIR = os.environ.get(
    "PERFDIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RESULTS = os.path.join(PERFDIR, "results", "per_cell.csv")
REPORT = os.path.join(PERFDIR, "results", "REPORT.md")
FIG_DIR = os.path.join(PERFDIR, "results", "figures")


def load_rows() -> list[dict]:
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS) as f:
        return list(csv.DictReader(f))


def pair_on_off(rows: list[dict]) -> dict:
    """Pair (ISL, BS, MTP, Mode) cells across GVR={true, false}."""
    pairs: dict = collections.defaultdict(dict)
    for row in rows:
        key = (row["ISL"], row["BS"], row["MTP"], row["Mode"])
        gvr = "on" if row["GVR"] in ("True", "true", "1") else "off"
        pairs[key][gvr] = row
    return pairs


def fmt_delta(on: float, off: float) -> str:
    if off == 0:
        return "n/a"
    delta = (on - off) / off * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f} %"


def safe_float(v) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main() -> None:
    rows = load_rows()
    if not rows:
        print(f"no rows in {RESULTS}; run more cells first")
        return
    pairs = pair_on_off(rows)
    full_pairs = {k: v for k, v in pairs.items() if "on" in v and "off" in v}
    print(f"{len(rows)} cells loaded; {len(full_pairs)} have both feature ON and OFF")

    os.makedirs(FIG_DIR, exist_ok=True)

    lines: list[str] = []
    lines += [
        "# DSv4 paired-feature sweep — Current Results",
        "",
        f"Generated from `{os.path.relpath(RESULTS, PERFDIR)}`.",
        f"Cells: {len(rows)} total; {len(full_pairs)} complete ON+OFF pairs.",
        "",
        "## Per-pair ΔTPOT summary",
        "",
        "| ISL | OSL | BS | MTP | Mode | TPOT-ON (ms) | TPOT-OFF (ms) | ΔTPOT | Δreq/s | Δttft | dispatch | numRows |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|:---:|:---|",
    ]
    for key in sorted(full_pairs.keys(), key=lambda k: (int(k[0]), int(k[1]), int(k[2]), k[3])):
        on, off = full_pairs[key]["on"], full_pairs[key]["off"]
        tpot_on, tpot_off = safe_float(on["tpot_avg_ms"]), safe_float(off["tpot_avg_ms"])
        req_on, req_off = safe_float(on["req_per_sec"]), safe_float(off["req_per_sec"])
        ttft_on, ttft_off = safe_float(on["ttft_avg_ms"]), safe_float(off["ttft_avg_ms"])
        lines.append(
            "| {isl} | {osl} | {bs} | {mtp} | {mode} | {to:.2f} | {tf:.2f} | {dt} | {dr} | {dt2} | {disp} | {nr} |".format(
                isl=key[0], osl=on["OSL"], bs=key[1], mtp=key[2], mode=key[3],
                to=tpot_on or 0.0, tf=tpot_off or 0.0,
                dt=fmt_delta(tpot_on, tpot_off) if tpot_on and tpot_off else "n/a",
                dr=fmt_delta(req_on, req_off) if req_on and req_off else "n/a",
                dt2=fmt_delta(ttft_on, ttft_off) if ttft_on and ttft_off else "n/a",
                disp=on["dispatch_path"],
                nr=on["numRows_distribution"][:40],
            )
        )

    dispatch_summary = collections.Counter(
        r["dispatch_path"] for r in rows if r["GVR"] in ("True", "true", "1")
    )
    lines += [
        "",
        "## Dispatch path (feature=ON cells only)",
        "",
        "| Path | Cells | Note |",
        "|---|---:|---|",
    ]
    for k in ("Heuristic", "Mixed", "Radix", "Unknown"):
        n = dispatch_summary.get(k, 0)
        note = {
            "Heuristic": "feature fired as expected",
            "Mixed":     "some launches fell back (gate failed sometimes)",
            "Radix":     "feature never fired (gate failed every launch)",
            "Unknown":   "[Scheme X] debug line missing — check TRTLLM_SCHEMEX_DEBUG was set",
        }[k]
        lines.append(f"| {k} | {n} | {note} |")

    plan_path = os.path.join(PERFDIR, "plan.csv")
    if os.path.exists(plan_path):
        with open(plan_path) as f:
            plan_total = sum(1 for _ in f) - 1
        lines += [
            "",
            "## Progress",
            "",
            f"- plan.csv cells: **{plan_total}**",
            f"- completed (in per_cell.csv): **{len(rows)}** ({100.0*len(rows)/plan_total:.1f} %)",
            "- failed (in failed.txt): see `failed.txt`",
        ]

    os.makedirs(os.path.dirname(REPORT) or ".", exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {REPORT}")

    try:
        plot_figures(rows, full_pairs)
    except ImportError as e:
        print(f"[warn] skipping figures: {e}")


def plot_figures(rows: list[dict], full_pairs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode in ("TEP", "DEP"):
        isls = sorted({int(k[0]) for k in full_pairs if k[3] == mode and int(k[2]) == 3})
        bses = sorted({int(k[1]) for k in full_pairs if k[3] == mode and int(k[2]) == 3})
        if not isls or not bses:
            continue
        Z = [[None] * len(bses) for _ in isls]
        for i_idx, isl in enumerate(isls):
            for b_idx, bs in enumerate(bses):
                key = (str(isl), str(bs), "3", mode)
                if key in full_pairs and "on" in full_pairs[key] and "off" in full_pairs[key]:
                    on = safe_float(full_pairs[key]["on"]["tpot_avg_ms"])
                    off = safe_float(full_pairs[key]["off"]["tpot_avg_ms"])
                    if on and off:
                        Z[i_idx][b_idx] = (on - off) / off * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        Z_arr = [[(z if z is not None else 0) for z in row] for row in Z]
        im = ax.imshow(Z_arr, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
        ax.set_xticks(range(len(bses)))
        ax.set_xticklabels(bses)
        ax.set_yticks(range(len(isls)))
        ax.set_yticklabels([f"{i//1024}K" for i in isls])
        ax.set_xlabel("BS")
        ax.set_ylabel("ISL")
        ax.set_title(f"ΔTPOT% (ON − OFF), {mode}, MTP=3 — blue=ON wins")
        for i_idx in range(len(isls)):
            for b_idx in range(len(bses)):
                v = Z[i_idx][b_idx]
                if v is not None:
                    ax.text(b_idx, i_idx, f"{v:+.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="ΔTPOT %")
        fig.tight_layout()
        path = os.path.join(FIG_DIR, f"dtpot_isl_bs_{mode}_M3.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
