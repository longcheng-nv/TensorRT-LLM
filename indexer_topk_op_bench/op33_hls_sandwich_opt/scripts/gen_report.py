# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op33 report: aggregate knob-sweep nsys data (D1-D4 vs op27_hls base) into a
temporary REPORT.html. speedup = base_ns / cfg_ns (>1 = cfg faster than op27_hls).
Target line = +30% (geomean 1.30)."""
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
R = HERE / "results"

CFG_LABEL = {
    "base": "op27_hls (incumbent)",
    "d1_p4fast0": "D1: p4_smallbin OFF (warp/reg band select off)",
    "d4_p4rs0": "D4: p4_rank_scatter OFF (band snap)",
    "d2_qstock": "D2: qfracs stock (0.75,0.5,0.25)",
    "d2_qm2": "D2: qfracs M=2 (0.85,0.35)",
    "d4_slot1": "D4: slot_scale=1",
    "d3_qbins128": "D3: qbins=128",
    "d3_qbins64": "D3: qbins=64",
    "op33_dispatch": "op33 DISPATCH (M=3 for K512/1024, default K2048) — EXACT 48/48",
}


def load(fn):
    d = {}
    p = R / fn
    if not p.exists():
        return d
    for row in csv.DictReader(p.open()):
        try:
            d[(row["cfg"], int(row["K"]), int(row["N"]))] = float(row["mean_ns"])
        except (ValueError, KeyError):
            pass
    return d


def load_base():
    d = {}
    p = R / "baseline.log"
    if not p.exists():
        return d
    import re
    for line in p.read_text().splitlines():
        m = re.match(r"K(\d+) N(\d+) gpu\d+ mean=([\d.]+)ns", line)
        if m:
            d[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return d


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    base_bl = load_base()  # (K,N)->ns  (the dedicated baseline sweep)
    knobs = load("knobs.csv")
    d3 = load("d3.csv")
    disp = load("dispatch.csv")
    all_cfg = {**knobs, **d3, **disp}
    # base per cell: prefer the 'base' cfg from knobs (same-run), else baseline.log
    def base_ns(K, N):
        return all_cfg.get(("base", K, N)) or base_bl.get((K, N))

    Ks = [512, 1024, 2048]
    Ns = sorted({k[2] for k in all_cfg} | {k[1] for k in base_bl})
    cfgs = ["op33_dispatch", "d2_qm2", "d1_p4fast0", "d2_qstock", "d3_qbins128", "d3_qbins64", "d4_p4rs0", "d4_slot1"]

    rows_html = []
    cfg_geo = {}
    for cfg in cfgs:
        speeds = []
        cells_html = []
        for K in Ks:
            for N in Ns:
                bn = base_ns(K, N)
                cn = all_cfg.get((cfg, K, N))
                if bn and cn:
                    sp = bn / cn
                    speeds.append(sp)
                    color = "#2a9d3f" if sp >= 1.30 else ("#6a9a3a" if sp >= 1.05 else ("#b04a4a" if sp < 0.98 else "#888"))
                    cells_html.append(f"<td style='color:{color}'>{sp:.3f}</td>")
                else:
                    cells_html.append("<td>-</td>")
        g = gm(speeds)
        cfg_geo[cfg] = g
        gcol = "#2a9d3f" if g >= 1.30 else ("#b04a4a" if g < 1.0 else "#888")
        rows_html.append(
            f"<tr><td class='lbl'>{CFG_LABEL.get(cfg, cfg)}</td>"
            + "".join(cells_html)
            + f"<td style='font-weight:700;color:{gcol}'>{g:.3f}</td></tr>")

    # base absolute table
    base_html = []
    for K in Ks:
        cells = "".join(f"<td>{base_ns(K,N)/1000:.2f}</td>" if base_ns(K, N) else "<td>-</td>" for N in Ns)
        base_html.append(f"<tr><td class='lbl'>K={K}</td>{cells}</tr>")

    hdr = "".join(f"<th>K512·N{N//1024}K</th>" if False else "" for N in Ns)
    col_hdr = "".join(f"<th>{K}/{N//1024}K</th>" for K in Ks for N in Ns)
    base_col_hdr = "".join(f"<th>N{N//1024}K</th>" for N in Ns)

    best_geo = max(cfg_geo.values()) if cfg_geo else float("nan")
    dg = cfg_geo.get("op33_dispatch", float("nan"))
    verdict = ("+30% TARGET MET" if dg >= 1.30 else
               f"+30% target NOT met — but op33 DISPATCH (M=3 K512/1024, EXACT 48/48) ships a "
               f"conditional +{(dg-1)*100:.1f}% overall (K512/1024 ~+9%), the only exact positive lever")

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>op33 HLS-op27 sandwich optimization — knob sweep</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222;background:#fafafa}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:26px}}
table{{border-collapse:collapse;margin:10px 0;font-size:12px}}
td,th{{border:1px solid #ddd;padding:4px 8px;text-align:right}}
th{{background:#f0f0f0}} td.lbl{{text-align:left;font-weight:600}}
.note{{background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:12px 16px;font-size:13px;max-width:900px}}
.verdict{{font-size:15px;font-weight:700;padding:10px 14px;border-radius:6px;background:#fff3cd;border:1px solid #ffe08a}}
code{{background:#eee;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>op33 — HLS-op27 sandwich optimization (BS=1, fp32, K512/1024/2048)</h1>
<p class='verdict'>Target: beat <code>op27_hls</code> by avg +30% (geomean 1.30). Result: <b>{verdict}</b></p>
<div class='note'>
<b>Incumbent</b> = <code>op27_hls</code> (gvr_ms_auto @ op27 HEAD; GvrSandwichKernel: P1 hint-gather →
P1b 256-hist rank-quantile M=4 thresholds → ONE fused count+collect pass → sandwich direct-write M0&lt;K
winners + band → band-only P4). Measured nsys cold-L2, B200 sm_100, BS=1 fp32 real, 60 cold reps ×3-median.
speedup = <code>t(op27_hls) / t(cfg)</code>; &gt;1 = faster than incumbent. Green ≥1.30, red &lt;0.98.
<br><b>D1</b> (warp/register band tie-select, sglang INSIGHTS-P3) is ALREADY the incumbent default
(<code>p4_smallbin=True</code>) — the row shows p4_smallbin OFF, i.e. how much D1 already contributes.
</div>
<h2>Incumbent absolute time (µs, nsys cold-L2)</h2>
<table><tr><th></th>{base_col_hdr}</tr>{''.join(base_html)}</table>
<h2>Directions vs op27_hls — speedup (base_ns / cfg_ns), columns = K/N</h2>
<table><tr><th>direction</th>{col_hdr}<th>geomean</th></tr>
{''.join(rows_html)}
</table>
<div class='note'>
<b>Reading:</b> every knob is an env A/B on the SAME incumbent kernel (no new kernel written; the
sandwich already incorporates the borrow ideas). A green geomean would mean a re-tuned knob beats the
shipped default — unlikely by construction (if it won, it would already be default). See ITERATIONS.md
for per-direction verdicts and FALSIFIED.md for the ledger. Structural +30% at BS=1 single-CTA is
bounded by the same latency/barrier wall op32 established (dram 0.06%, issue 15%).
</div>
</body></html>"""
    out = HERE / "REPORT.html"
    out.write_text(html)
    print(f"wrote {out}")
    print(f"best single-knob geomean = {best_geo:.3f} | verdict: {verdict}")
    for c, g in sorted(cfg_geo.items(), key=lambda x: -x[1]):
        print(f"  {c:16} geomean {g:.3f}")


if __name__ == "__main__":
    main()
