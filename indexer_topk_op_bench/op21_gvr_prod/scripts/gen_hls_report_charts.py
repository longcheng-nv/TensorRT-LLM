# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the interactive-data block (sections 6.6/6.7) of
HLS_VALIDATION_REPORT.html from the authoritative nsys artifacts.

Data sources (all committed / NFS-resident):
  results/nsys/iter13_ab_hls/       iter13 log-falsi A/B   (alpha=0.2)
  results/nsys/iter13_ab_hls_a01/   iter13 alpha=0.1 probe
  results/nsys/iter13_ab_hls_dist/  iter14 distributed A/B
  ../count_ge_multi_bench/results_m3.csv   tau(M) same-silicon sweep
  P0 spot medians: hardcoded from the measured runs (ITERATIONS iter13/14).

Output: replaces the block between <!-- HLS-CHARTS-BEGIN --> and
<!-- HLS-CHARTS-END --> in HLS_VALIDATION_REPORT.html (both language
variants inside), and ensures the plotly CDN tag exists in <head>.
Chart interaction follows the op22 REPORT.html conventions (plotly +
checkbox/radio controls). Bilingual labels via .en/.zh spans.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP21 = HERE.parents[0]
BENCH = OP21.parents[0]
sys.path.insert(0, str(BENCH / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

RUNS = {
    "i13": OP21 / "results/nsys/iter13_ab_hls",
    "a01": OP21 / "results/nsys/iter13_ab_hls_a01",
    "i14": OP21 / "results/nsys/iter13_ab_hls_dist",
}
SCENS = ["best", "worst", "real"]

P0_SPOT = {  # cell label -> {run: [old_us, new_us]}
    "K512 fp32 262K":  {"i13": [18.14, 17.38], "i14": [17.47, 18.21]},
    "K1024 fp32 65K":  {"i13": [15.07, 14.29], "i14": [14.30, 14.82]},
    "K1024 fp32 262K": {"i13": [19.87, 19.01], "i14": [18.98, 19.78]},
    "K2048 fp32 262K": {"i13": [18.29, 19.33], "i14": [19.26, 19.10]},
    "K1024 bf16 262K": {"i13": [14.50, 14.59], "i14": [14.56, 14.56]},
}


def load_ab():
    """-> {run: {scen: [ {K,N,BS,ms_path,old,new,exact} ]}}"""
    out = {}
    for run, d in RUNS.items():
        out[run] = {}
        for scen in SCENS:
            rep = d / f"ab_{scen}_fp32.nsys-rep"
            jl = d / f"ab_{scen}_fp32.jsonl"
            if not rep.exists() or not jl.exists():
                continue
            kern = parse_rep(rep)
            cells = []
            for line in jl.read_text().splitlines():
                r = json.loads(line)
                if "error" in r:
                    continue
                base = (f"{r['scenario']}|{r['K']}|{r['dtype']}|{r['N']}"
                        f"|{r['BS']}")
                uo = kern.get(f"c|old|{base}")
                un = kern.get(f"c|new|{base}")
                if uo is None or un is None:
                    continue
                cells.append({"K": r["K"], "N": r["N"], "BS": r["BS"],
                              "path": r.get("ms_path"),
                              "old": round(uo, 2), "new": round(un, 2),
                              "exact": f"{r.get('exact_old','?')}/"
                                       f"{r.get('exact_new','?')}"})
            out[run][scen] = cells
    return out


def load_tau():
    """-> {dtype: {N: {M: us}}} from results_m3.csv"""
    out = {}
    p = BENCH / "count_ge_multi_bench" / "results_m3.csv"
    for r in csv.DictReader(open(p)):
        out.setdefault(r["dtype"], {}).setdefault(int(r["N"]), {})[
            int(r["M"])] = float(r["us_med"])
    return out


def bi(en, zh):
    # inline spans; the charts-block CSS forces .zh display:inline in this
    # scope (the report default is block, which would break control rows)
    return f'<span class="en">{en}</span><span class="zh">{zh}</span>'


def detail_table(cells, run_label):
    rows = []
    for c in cells:
        ratio = c["old"] / c["new"] if c["new"] else float("nan")
        cls = ' class="hl-good"' if ratio > 1.02 else (
            ' class="hl-bad"' if ratio < 0.98 else "")
        rows.append(
            f"<tr><td>K{c['K']}</td><td class=num>{c['N']}</td>"
            f"<td class=num>{c['BS']}</td><td>{c['path']}</td>"
            f"<td class=num>{c['old']:.2f}</td><td class=num>{c['new']:.2f}"
            f"</td><td class=num{cls}>{ratio:.3f}</td>"
            f"<td>{c['exact']}</td></tr>")
    return ("<table><tr><th>K</th><th class=num>N</th><th class=num>BS</th>"
            "<th>path</th><th class=num>old µs</th><th class=num>new µs</th>"
            f"<th class=num>old/new</th><th>exact</th></tr>"
            + "".join(rows) + "</table>")


def main():
    ab = load_ab()
    tau = load_tau()
    data_js = json.dumps({"ab": ab, "tau": tau, "p0": P0_SPOT},
                         separators=(",", ":"))

    # ---------------- section 6.6 (charts) ----------------
    ctl1 = (
        '<div class="ctl">'
        + bi("<b>scenarios</b>", "<b>场景</b>") + " "
        + "".join(f'<label class="ck"><input type="checkbox" class="hsck" '
                  f'value="{s}"{" checked" if s != "worst" else ""}>'
                  f'{s.upper()}</label> ' for s in SCENS)
        + " · " + bi("<b>run</b>", "<b>轮次</b>") + " "
        '<label class="ck"><input type="radio" name="hrun" value="i13">'
        'iter13 log-falsi</label> '
        '<label class="ck"><input type="radio" name="hrun" value="a01">'
        'iter13 α=0.1</label> '
        '<label class="ck"><input type="radio" name="hrun" value="i14" '
        'checked>iter14 distributed</label><br>'
        + "".join(f'<label class="ck"><input type="radio" name="hkk" '
                  f'value="{k}"{" checked" if k == 2048 else ""}>K={k}'
                  f'</label> ' for k in (512, 1024, 2048))
        + '</div>')
    charts = f"""
<h3>6.6 {bi("Interactive charts (nsys cold-L2, BS=1 seqlen cells)",
            "交互图表(nsys 冷 L2,BS=1 seqlen 格)")}</h3>
<p class="small">{bi(
  "Check scenarios / pick run + K. Left: absolute µs vs N (log-log; old "
  "dashed, new solid — pairs are same-process interleaved, so within-run "
  "comparisons are throttle-immune; absolute µs do NOT transfer across "
  "runs). Right: the paired old/new ratio vs N (cross-run safe; >1 = new "
  "wins). BS=16 cells live in the 6.7 tables.",
  "勾选场景/选择轮次与 K。左:绝对 µs 对 N(双对数;old 虚线、new 实线 —— "
  "两臂同进程交错,轮内比较对节流免疫;绝对 µs 不跨轮可比)。右:配对 "
  "old/new 比值对 N(跨轮安全;>1 = new 胜)。BS=16 格见 6.7 明细表。")}</p>
<div class="card">{ctl1}
<div class="row"><div id="h_abs" class="plt"></div>
<div id="h_ratio" class="plt"></div></div></div>

<p class="small">{bi(
  "Width-tax tau(M) measured same-silicon (Step 0) — tau(3) sits on the "
  "interpolated 1.2 at large N. P0 spot: the fast-path cost of each "
  "iteration's code mass on op21 synth data (light/no fallback).",
  "宽度税 tau(M) 同硅实测(Step 0)—— 大 N 处 tau(3) 落在插值 1.2 上。"
  "P0 spot:两轮代码质量对 op21 synth 数据(轻/无 fallback)fast path "
  "的代价。")}</p>
<div class="card"><div class="ctl">
<label class="ck"><input type="checkbox" class="htck" value="2" checked>M=2</label>
<label class="ck"><input type="checkbox" class="htck" value="3" checked>M=3</label>
<label class="ck"><input type="checkbox" class="htck" value="4" checked>M=4</label>
 · <label class="ck"><input type="checkbox" class="htdt" value="fp32" checked>fp32</label>
<label class="ck"><input type="checkbox" class="htdt" value="fp16" checked>fp16</label>
</div>
<div class="row"><div id="h_tau" class="plt"></div>
<div id="h_p0" class="plt"></div></div></div>
"""

    # ---------------- section 6.7 (detail tables) ----------------
    det = [f"<h3>6.7 {bi('Full per-cell data', '逐格全量数据')}</h3>"]
    names = {"i13": ("iter13 log-falsi A/B (α=0.2)", "iter13 log-falsi A/B(α=0.2)"),
             "a01": ("iter13 α=0.1 probe", "iter13 α=0.1 探针"),
             "i14": ("iter14 distributed-fallback A/B", "iter14 分布式 fallback A/B")}
    for run in ("i13", "a01", "i14"):
        for scen in SCENS:
            cells = ab.get(run, {}).get(scen)
            if not cells:
                continue
            en, zh = names[run]
            det.append(
                f"<details><summary>{bi(en, zh)} — {scen} "
                f"({len(cells)} cells)</summary>"
                + detail_table(cells, run) + "</details>")
    tau_rows = []
    for dt in ("fp32", "fp16"):
        for N in sorted(tau.get(dt, {})):
            ms = tau[dt][N]
            if 1 not in ms:
                continue
            tau_rows.append(
                f"<tr><td>{dt}</td><td class=num>{N}</td>"
                + "".join(f"<td class=num>{ms.get(m, float('nan')):.3f}</td>"
                          for m in (1, 2, 3, 4))
                + "".join(f"<td class=num>{ms[m]/ms[1]:.3f}</td>"
                          if m in ms else "<td class=num>—</td>"
                          for m in (2, 3, 4))
                + "</tr>")
    det.append(
        f"<details><summary>{bi('tau(M) same-silicon sweep (Step 0)', 'tau(M) 同硅扫描(Step 0)')}"
        "</summary><table><tr><th>dtype</th><th class=num>N</th>"
        "<th class=num>M1 µs</th><th class=num>M2 µs</th>"
        "<th class=num>M3 µs</th><th class=num>M4 µs</th>"
        "<th class=num>τ(2)</th><th class=num>τ(3)</th><th class=num>τ(4)</th></tr>"
        + "".join(tau_rows) + "</table></details>")
    p0_rows = []
    for cell, d in P0_SPOT.items():
        r13 = d["i13"][0] / d["i13"][1]
        r14 = d["i14"][0] / d["i14"][1]
        p0_rows.append(
            f"<tr><td>{cell}</td>"
            f"<td class=num>{d['i13'][0]:.2f}</td><td class=num>{d['i13'][1]:.2f}</td>"
            f"<td class=num>{r13:.3f}</td>"
            f"<td class=num>{d['i14'][0]:.2f}</td><td class=num>{d['i14'][1]:.2f}</td>"
            f"<td class=num>{r14:.3f}</td></tr>")
    det.append(
        f"<details><summary>{bi('P0 no-regress spot (op21 synth)', 'P0 无回归 spot(op21 synth)')}"
        "</summary><table><tr><th>cell</th>"
        "<th class=num>i13 old</th><th class=num>i13 new</th><th class=num>old/new</th>"
        "<th class=num>i14 old</th><th class=num>i14 new</th><th class=num>old/new</th></tr>"
        + "".join(p0_rows) + "</table></details>")

    wiring = """
<script>
const HD = %DATA%;
const NS = [16384, 65536, 262144, 1048576];
const SCOL = {best: "#0f6bb3", worst: "#a12a1e", real: "#1a7a4a"};
const LAYOUT = (title, ylab, ylog) => ({
  title: {text: title, font: {size: 13}}, height: 360,
  margin: {l: 55, r: 10, t: 36, b: 42},
  xaxis: {type: "log", title: "N", tickvals: NS,
          ticktext: ["16K", "64K", "256K", "1M"]},
  yaxis: {type: ylog ? "log" : "linear", title: ylab},
  legend: {orientation: "h", font: {size: 10}}, showlegend: true});
function hSel(cls) {
  return Array.from(document.querySelectorAll("input." + cls + ":checked"))
    .map(e => e.value);
}
function hDraw() {
  const scens = hSel("hsck");
  const run = document.querySelector('input[name="hrun"]:checked').value;
  const K = +document.querySelector('input[name="hkk"]:checked').value;
  const tAbs = [], tRat = [];
  for (const s of scens) {
    const cells = ((HD.ab[run] || {})[s] || []).filter(
      c => c.K === K && c.BS === 1);
    cells.sort((a, b) => a.N - b.N);
    const xs = cells.map(c => c.N);
    tAbs.push({x: xs, y: cells.map(c => c.old), name: s + " old",
               mode: "lines+markers", line: {color: SCOL[s], dash: "dash"}});
    tAbs.push({x: xs, y: cells.map(c => c.new), name: s + " new",
               mode: "lines+markers", line: {color: SCOL[s]}});
    tRat.push({x: xs, y: cells.map(c => c.old / c.new), name: s,
               mode: "lines+markers", line: {color: SCOL[s]}});
  }
  tRat.push({x: [NS[0], NS[NS.length - 1]], y: [1, 1], name: "parity",
             mode: "lines", line: {color: "#888", dash: "dot"},
             showlegend: false});
  Plotly.react("h_abs",
    tAbs, LAYOUT(run + "  K=" + K + "  cold-L2 µs (old dashed / new solid)",
                 "µs", true));
  Plotly.react("h_ratio",
    tRat, LAYOUT(run + "  K=" + K + "  paired old/new ratio (>1 = new wins)",
                 "old/new", false));
}
function hDrawTau() {
  const ms = hSel("htck").map(Number);
  const dts = hSel("htdt");
  const tr = [];
  const mcol = {2: "#0f6bb3", 3: "#1a7a4a", 4: "#a12a1e"};
  for (const dt of dts) {
    const byN = HD.tau[dt] || {};
    const xs = Object.keys(byN).map(Number).sort((a, b) => a - b)
      .filter(n => byN[n]["1"] !== undefined);
    for (const m of ms) {
      const y = xs.map(n => byN[n][m] !== undefined
                            ? byN[n][m] / byN[n]["1"] : null);
      tr.push({x: xs, y: y, name: dt + " τ(" + m + ")",
               mode: "lines+markers",
               line: {color: mcol[m], dash: dt === "fp16" ? "dash" : "solid"}});
    }
  }
  tr.push({x: [4096, 1048576], y: [1.2, 1.2], name: "HLS τ(3)=1.2",
           mode: "lines", line: {color: "#1a7a4a", dash: "dot", width: 1}});
  const lay = LAYOUT("width tax τ(M) = t(M)/t(M=1), cold-L2", "τ", false);
  lay.xaxis.tickvals = [4096, 16384, 65536, 262144];
  lay.xaxis.ticktext = ["4K", "16K", "64K", "256K"];
  Plotly.react("h_tau", tr, lay);
  const cells = Object.keys(HD.p0);
  const bars = [
    {x: cells, y: cells.map(c => 100 * (HD.p0[c].i13[0] / HD.p0[c].i13[1] - 1)),
     name: "iter13", type: "bar", marker: {color: "#0f6bb3"}},
    {x: cells, y: cells.map(c => 100 * (HD.p0[c].i14[0] / HD.p0[c].i14[1] - 1)),
     name: "iter14 (dist forced)", type: "bar", marker: {color: "#a12a1e"}}];
  Plotly.react("h_p0", bars, {
    title: {text: "P0 fast-path spot: paired gain % (neg = code-mass tax)",
            font: {size: 13}},
    height: 360, margin: {l: 55, r: 10, t: 36, b: 90},
    yaxis: {title: "old/new − 1  (%)"}, barmode: "group",
    legend: {orientation: "h", font: {size: 10}}});
}
for (const e of document.querySelectorAll(
     "input.hsck, input[name=hrun], input[name=hkk]"))
  e.addEventListener("change", hDraw);
for (const e of document.querySelectorAll("input.htck, input.htdt"))
  e.addEventListener("change", hDrawTau);
hDraw(); hDrawTau();
</script>
""".replace("%DATA%", data_js)

    css = """
<style>
  .card { background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: .7rem .9rem; margin: 1rem 0; }
  .ctl { font-size: .84rem; margin-bottom: .4rem; }
  .ck { margin-right: .55rem; white-space: nowrap; cursor: pointer; }
  .row { display: flex; flex-wrap: wrap; gap: .6rem; }
  .plt { flex: 1 1 440px; min-width: 380px; }
  details { margin: .5rem 0; }
  details summary { cursor: pointer; font-size: .9rem; color: var(--accent); }
  details table { font-size: .8rem; }
  /* charts block: bilingual spans must stay INLINE (report default for
     .zh is display:block, which breaks control rows / summaries) */
  #lang-zh:checked ~ .wrap .ctl .zh,
  #lang-zh:checked ~ .wrap summary .zh,
  #lang-zh:checked ~ .wrap h3 .zh,
  #lang-zh:checked ~ .wrap p.small .zh { display: inline; }
</style>"""

    frag = ("<!-- HLS-CHARTS-BEGIN -->" + css + charts + "".join(det)
            + wiring + "\n<!-- HLS-CHARTS-END -->")

    p = OP21 / "HLS_VALIDATION_REPORT.html"
    t = p.read_text()
    if "HLS-CHARTS-BEGIN" in t:
        pre, rest = t.split("<!-- HLS-CHARTS-BEGIN -->", 1)
        _, post = rest.split("<!-- HLS-CHARTS-END -->", 1)
        t = pre + frag + post
    else:
        anchor = "<footer>"
        t = t.replace(anchor, frag + "\n" + anchor, 1)
    if "cdn.plot.ly" not in t:
        t = t.replace("</title>",
                      '</title>\n<script src="https://cdn.plot.ly/'
                      'plotly-2.35.2.min.js"></script>', 1)
    p.write_text(t)
    n_cells = sum(len(c) for r in ab.values() for c in r.values())
    print(f"inserted charts block: {n_cells} A/B cells, "
          f"{len(tau_rows)} tau rows, {len(P0_SPOT)} p0 cells")


if __name__ == "__main__":
    main()
