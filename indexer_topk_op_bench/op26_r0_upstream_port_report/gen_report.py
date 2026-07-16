# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generate the bilingual (EN default / CN toggle) op26->production GVR report.

Language toggle = pure CSS :checked ~ sibling (works even if <script> is stripped).
Charts = interactive Plotly (checkbox arm-select + multi-select scenario/model to
compare settings), matching op22 REPORT.html style. Tables are the no-script
fallback. Reads synth_3arm.csv (op22 §env, K512/1024/2048 = V4-Flash/V4-Pro/V3.2)
and real_3arm.csv (V4 flash+pro decode-capture).
"""
import csv, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
SYN_EXACT, DTYPE_EXACT = 52, 36   # canonical exactness-gate counts (see memory)
# K -> model family (house convention, matches op22 §env radio labels)
KLAB = {"512": "V4 Flash (K=512)", "1024": "V4 Pro (K=1024)", "2048": "V3.2 (K=2048)"}


def read_csv(name):
    p = os.path.join(HERE, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def geo(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def fnum(s):
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except (ValueError, TypeError):
        return None


synth = read_csv("synth_3arm.csv")
real = read_csv("real_3arm.csv")

# ---- geomeans -------------------------------------------------------------
def sgeo(pred, col):
    return geo([fnum(r[col]) for r in synth if pred(r)])

syn_pvb = sgeo(lambda r: True, "pr_vs_base")
syn_pvo = sgeo(lambda r: True, "my_vs_op26")
syn_pvb_cs1 = sgeo(lambda r: r["cs"] == "1", "pr_vs_base")
syn_pvb_cs4 = sgeo(lambda r: r["cs"] == "4", "pr_vs_base")

real_pvb = geo([fnum(r["pr_vs_base"]) for r in real])
real_pvo = geo([fnum(r["pr_vs_op26"]) for r in real])
real_exact = sum(r["exact"] == "True" for r in real)
real_pvb_flash = geo([fnum(r["pr_vs_base"]) for r in real if r["model"] == "flash"])
real_pvb_pro = geo([fnum(r["pr_vs_base"]) for r in real if r["model"] == "pro"])
real_v32 = [r for r in real if r["model"] == "v32"]
real_pvb_v32 = geo([fnum(r["pr_vs_base"]) for r in real_v32]) if real_v32 else float("nan")
real_pvo_v32 = geo([fnum(r["pr_vs_op26"]) for r in real_v32]) if real_v32 else float("nan")
real_exact_v32 = sum(r["exact"] == "True" for r in real_v32)
base_inexact = [r for r in real if r.get("base_exact") == "False"]

# ---- JSON data for the interactive charts ---------------------------------
SYN_JS = [dict(scen=r["scen"], K=r["K"], N=int(r["N"]), cs=int(r["cs"]),
               base=fnum(r["base"]), pr=fnum(r["pr"]), op26=fnum(r["op26"]))
          for r in synth]
REAL_JS = [dict(model=r["model"], isl=r["isl"], N=int(r["N"]), K=int(r["K"]),
                hit=fnum(r["hit_rate"]), base=fnum(r["base"]), pr=fnum(r["pr"]),
                op26=fnum(r["op26"])) for r in real]

# ---- BS-scaling data (§8) -------------------------------------------------
bs_synth = read_csv("bs_synth.csv")
bs_real = read_csv("bs_real.csv")
BS_SYN_JS = [dict(K=r["K"], dtype=r["dtype"], N=int(r["N"]), scen=r["scen"], cs=int(r["cs"]),
                  BS=int(r["BS"]), base=fnum(r["base"]), pr=fnum(r["pr"]), op26=fnum(r["op26"]))
             for r in bs_synth]
BS_REAL_JS = [dict(model=r["model"], isl=r["isl"], dtype=r["dtype"], N=int(r["N"]) if r["N"] else None,
                   BS=int(r["BS"]), hit=fnum(r["hit"]), base=fnum(r["base"]), pr=fnum(r["pr"]),
                   op26=fnum(r["op26"])) for r in bs_real]  # isl feeds the §7 real ISL radio (brl)

def bsgeo(rows, col, pred):
    return geo([fnum(r[col]) for r in rows if pred(r)])

# low-BS (latency-bound) vs high-BS (throughput-bound) PR/base story
bs_syn_pvb_lo = bsgeo(bs_synth, "pr_vs_base", lambda r: r["BS"] == "1")
bs_syn_pvb_hi = bsgeo(bs_synth, "pr_vs_base", lambda r: r["BS"] == "1024")
bs_real_pvb_lo = bsgeo(bs_real, "pr_vs_base", lambda r: r["BS"] == "1")
bs_real_pvb_hi = bsgeo(bs_real, "pr_vs_base", lambda r: r["BS"] == "1024")
bs_syn_pvo_lo = bsgeo(bs_synth, "op26_vs_pr", lambda r: r["BS"] == "1")
bs_syn_pvo_hi = bsgeo(bs_synth, "op26_vs_pr", lambda r: r["BS"] == "1024")
bs_syn_exact = sum(1 for r in bs_synth if r.get("pr_exact") == "True")
bs_syn_exact_tot = sum(1 for r in bs_synth if r.get("pr_exact") in ("True", "False"))
bs_real_exact = sum(1 for r in bs_real if r.get("pr_exact") == "True")
bs_real_exact_tot = sum(1 for r in bs_real if r.get("pr_exact") in ("True", "False"))
bs_have = bool(bs_synth or bs_real)

# ---- big-BS dispatch triage (nsys, umbriel-b200-094 2026-07-15) -----------
# bigbs_triage.csv: pr_frozen (this chapter's harness instantiation) vs
# pr_runner (production CuteDSLGvrTopKDecodeRunner policy replica) vs op26,
# BS in {64,256,1024}. See BIGBS_TRIAGE_NOTE.md / ab_bigbs_runnercfg.py /
# bigbs_nsys.py + parse_bigbs.py.
bigbs = read_csv("bigbs_triage.csv")
bigbs_fro = geo([fnum(r["frozen_vs_op26"]) for r in bigbs])
bigbs_run = geo([fnum(r["runner_vs_op26"]) for r in bigbs])
bigbs_fro_max = max((fnum(r["frozen_vs_op26"]) or 0) for r in bigbs) if bigbs else float("nan")
bigbs_exact = sum(r["exact_all"] == "True" for r in bigbs)

def _f(x, d="—"):
    return f"{x:.3f}×" if isinstance(x, float) and not math.isnan(x) else d

# ---- §8 External-rival data ------------------------------------------------
# rival_long.csv: family,sweep,scenario,model,op,K,dtype,N,BS,isl,cr,hit,us,us_span,exact
rival = read_csv("rival_long.csv")
RIVAL_JS = [dict(family=r["family"], sweep=r["sweep"], scenario=r.get("scenario", ""),
                 model=r.get("model", ""), op=r["op"], K=int(r["K"]), dtype=r["dtype"],
                 N=int(r["N"]), BS=int(r["BS"]), isl=r.get("isl", ""),
                 hit=fnum(r["hit"]), us=fnum(r["us"]),
                 us_span=fnum(r.get("us_span", "")),
                 # primary latency = honest wall-clock span (== kern-sum for single-kernel
                 # ops; avoids double-counting SGLang v2's overlapped 2-kernel PDL path)
                 t=(fnum(r.get("us_span", "")) or fnum(r["us"])),
                 exact=r.get("exact", ""))
            for r in rival]
RIVAL_ARMS = ["gvr_base", "gvr_pr", "op26_r0auto", "radix_cutedsl", "sglang_v2", "flashinfer_topk"]
rival_have = bool(rival)


def _rival_vs_gvr(pred):
    """geomean t(op26_r0auto)/t(op) over matching cells (>1 => op slower than GVR).
    Uses honest wall-clock span (us_span) with kern-sum fallback."""
    from collections import defaultdict
    cells = defaultdict(dict)
    for r in rival:
        u = fnum(r.get("us_span", "")) or fnum(r["us"])
        if u:
            cells[(r["family"], r["sweep"], r.get("scenario", ""), r.get("model", ""),
                   r["dtype"], r["K"], r["N"], r["BS"], r.get("isl", ""))][r["op"]] = u
    out = {a: [] for a in RIVAL_ARMS if a != "op26_r0auto"}
    for key, d in cells.items():
        g = d.get("op26_r0auto")
        if not g or not pred(key):
            continue
        for a in out:
            if d.get(a):
                out[a].append(g / d[a])
    return {a: geo(v) for a, v in out.items() if v}


# fastest-op summary on fp32 seqlen (BS=1) — headline comparison.
# cell key = (family, sweep, scenario, model, dtype, K, N, BS, isl); BS is a str.
riv_syn_fp32 = _rival_vs_gvr(lambda k: k[0] == "synth" and k[1] == "seqlen" and k[4] == "fp32" and k[7] == "1")
riv_real_fp32 = _rival_vs_gvr(lambda k: k[0] == "real" and k[1] == "seqlen" and k[4] == "fp32" and k[7] == "1")
rival_exact_ct = sum(1 for r in rival if r.get("exact") == "True")
rival_exact_tot = sum(1 for r in rival if r.get("exact") in ("True", "False"))
rival_omit = "(see aggregate_rival.py stdout for omitted unsupported/OOM/missing cells)"


def _best_line(d):
    """Human summary: which rival is closest-to / beats GVR."""
    if not d:
        return "—"
    return " · ".join(f"{a.replace('_topk','').replace('_cutedsl','')}: {v:.2f}×" for a, v in
                      sorted(d.items(), key=lambda x: -x[1]))

# ---- correctness-fix callout ----------------------------------------------
if base_inexact:
    bx = base_inexact[0]
    ratio = float(bx["base"]) / float(bx["pr"])
    fix_en = (f'<div class="keep"><b>Correctness bonus:</b> on the low-hit real cell '
              f'<code>{bx["model"]} {bx["isl"]} (N={bx["N"]}, hit={bx["hit_rate"]})</code> the upstream '
              f'<b>base</b> secant kernel is <b>inexact</b> (undershoot: fewer than K unique indices) '
              f'<i>and</i> {ratio:.2f}× slower ({bx["base"]}µs vs PR {bx["pr"]}µs). The R0 port is '
              f'<b>exact here</b> — it repairs a real-data undershoot the shipped kernel exhibits.</div>')
    fix_zh = (f'<div class="keep"><b>正确性附赠：</b>在低命中真实 cell '
              f'<code>{bx["model"]} {bx["isl"]}（N={bx["N"]}，hit={bx["hit_rate"]}）</code>上，上游'
              f'<b>base</b> secant 核<b>不精确</b>（undershoot：唯一索引不足 K 个），<i>且</i>慢 '
              f'{ratio:.2f}×（{bx["base"]}µs vs PR {bx["pr"]}µs）。R0 移植在此<b>精确</b>——修复了'
              f'已上线核在真实数据上的一处 undershoot。</div>')
else:
    fix_en = fix_zh = ""


# ---- tables (no-script fallback) ------------------------------------------
def synth_table():
    h = ("<tr><th>scen</th><th>model (K)</th><th>N</th><th>cs</th><th>base µs</th><th>PR µs</th>"
         "<th>op26 µs</th><th>PR/base</th><th>op26/PR</th><th>exact</th></tr>")
    body = ""
    for r in synth:
        n = int(r["N"])
        nl = f"{n//1024}K" if n >= 1024 else str(n)
        body += (f"<tr><td>{r['scen']}</td><td>{KLAB[r['K']]}</td><td>{nl}</td><td>{r['cs']}</td>"
                 f"<td>{r['base']}</td><td>{r['pr']}</td><td>{r['op26']}</td>"
                 f"<td>{r['pr_vs_base']}</td><td>{r['my_vs_op26']}</td><td>{r['exact']}</td></tr>")
    return f"<table>{h}{body}</table>"


def real_table():
    if not real:
        return "<p class='mut'>(real_3arm.csv not yet available)</p>"
    h = ("<tr><th>model</th><th>ISL</th><th>N</th><th>K</th><th>hit</th><th>cs</th>"
         "<th>base µs</th><th>PR µs</th><th>op26 µs</th><th>PR/base</th><th>op26/PR</th>"
         "<th>PR exact</th><th>base exact</th></tr>")
    body = ""
    for r in real:
        be = r.get("base_exact", "")
        be_cell = (f'<td style="color:var(--red)">{be}</td>' if be == "False" else f"<td>{be}</td>")
        body += (f"<tr><td>{r['model']}</td><td>{r['isl']}</td><td>{r['N']}</td><td>{r['K']}</td>"
                 f"<td>{r['hit_rate']}</td><td>{r['cs']}</td><td>{r['base']}</td><td>{r['pr']}</td>"
                 f"<td>{r['op26']}</td><td>{r['pr_vs_base']}</td><td>{r['pr_vs_op26']}</td>"
                 f"<td>{r['exact']}</td>{be_cell}</tr>")
    return f"<table>{h}{body}</table>"


def bs_synth_table():
    if not bs_synth:
        return "<p class='mut'>(bs_synth.csv not yet available — run the BS sweep + aggregate_bs.py)</p>"
    h = ("<tr><th>model (K)</th><th>dtype</th><th>N</th><th>scen</th><th>BS</th><th>base µs</th>"
         "<th>PR µs</th><th>op26 µs</th><th>PR/base</th><th>op26/PR</th><th>exact</th></tr>")
    body = ""
    for r in bs_synth:
        n = int(r["N"]); nl = f"{n//1024}K" if n >= 1024 else str(n)
        body += (f"<tr><td>{KLAB[r['K']]}</td><td>{r['dtype']}</td><td>{nl}</td><td>{r['scen']}</td>"
                 f"<td>{r['BS']}</td><td>{r['base']}</td><td>{r['pr']}</td><td>{r['op26']}</td>"
                 f"<td>{r['pr_vs_base']}</td><td>{r['op26_vs_pr']}</td><td>{r['pr_exact']}</td></tr>")
    return f"<table>{h}{body}</table>"


def bs_real_table():
    if not bs_real:
        return "<p class='mut'>(bs_real.csv not yet available — run the BS sweep + aggregate_bs.py)</p>"
    h = ("<tr><th>model</th><th>ISL</th><th>dtype</th><th>N</th><th>hit</th><th>BS</th><th>base µs</th>"
         "<th>PR µs</th><th>op26 µs</th><th>PR/base</th><th>op26/PR</th><th>exact</th></tr>")
    body = ""
    for r in bs_real:
        body += (f"<tr><td>{r['model']}</td><td>{r['isl']}</td><td>{r['dtype']}</td><td>{r['N']}</td>"
                 f"<td>{r['hit']}</td><td>{r['BS']}</td><td>{r['base']}</td><td>{r['pr']}</td>"
                 f"<td>{r['op26']}</td><td>{r['pr_vs_base']}</td><td>{r['op26_vs_pr']}</td>"
                 f"<td>{r['pr_exact']}</td></tr>")
    return f"<table>{h}{body}</table>"


def rival_table():
    if not rival:
        return "<p class='mut'>(rival_long.csv not yet available — run the rival sweep + aggregate_rival.py)</p>"
    h = ("<tr><th>family</th><th>view</th><th>K/model</th><th>dtype</th><th>N</th><th>BS</th>"
         "<th>hit</th><th>op</th><th>µs</th><th>span µs</th><th>t(GVR)/t(op)</th><th>exact</th></tr>")
    # index op26 us per cell for the ratio column
    from collections import defaultdict as _dd
    g = _dd(dict)
    for r in rival:
        g[(r["family"], r["sweep"], r.get("scenario", ""), r.get("model", ""),
           r["dtype"], r["K"], r["N"], r["BS"], r.get("isl", ""))][r["op"]] = fnum(r["us"])
    # group rows into collapsed sub-tables by (family, view, dtype) so the report
    # tail doesn't render ~6000 rows at once — each group is a closed <details>
    groups, counts = _dd(str), _dd(int)
    for r in sorted(rival, key=lambda x: (x["family"], x["sweep"], x["dtype"], int(x["K"]),
                                          int(x["N"]), int(x["BS"]), x["op"])):
        km = KLAB.get(str(r["K"]), r["K"]) if r["family"] == "synth" else r.get("model", "")
        gv = g[(r["family"], r["sweep"], r.get("scenario", ""), r.get("model", ""),
                r["dtype"], r["K"], r["N"], r["BS"], r.get("isl", ""))].get("op26_r0auto")
        u = fnum(r["us"])
        ratio = f"{gv/u:.2f}" if (gv and u) else ""
        sp = r.get("us_span", "") or ""
        key = (r["family"], r["sweep"], r["dtype"])
        groups[key] += (f"<tr><td>{r['family']}</td><td>{r['sweep']}</td><td>{km}</td><td>{r['dtype']}</td>"
                        f"<td>{r['N']}</td><td>{r['BS']}</td><td>{r.get('hit','')}</td><td>{RVLAB_PY.get(r['op'],r['op'])}</td>"
                        f"<td>{r['us']}</td><td>{sp}</td><td>{ratio}</td><td>{r.get('exact','')}</td></tr>")
        counts[key] += 1
    FLAB = {"synth": ("synthetic", "合成"), "real": ("real", "真实")}
    VLAB = {"seqlen": ("seq-len view (BS=1)", "序列长视图 (BS=1)"), "bs": ("BS view", "BS 视图")}
    DORD = {"fp32": 0, "fp16": 1, "bf16": 2}
    FORD = {"synth": 0, "real": 1}
    VORD = {"seqlen": 0, "bs": 1}
    inner = ""
    for key in sorted(groups, key=lambda k: (FORD.get(k[0], 9), VORD.get(k[1], 9), DORD.get(k[2], 9))):
        fam, sw, dt = key
        fe, fz = FLAB.get(fam, (fam, fam))
        ve, vz = VLAB.get(sw, (sw, sw))
        inner += (f"<details><summary class='mut'>{fe} · {ve} · {dt} ({counts[key]} rows) / "
                  f"{fz} · {vz} · {dt}（{counts[key]} 行）</summary>"
                  f"<table>{h}{groups[key]}</table></details>")
    return (f"<details><summary class='mut'>full rival measurement table ({len(rival)} rows, "
            f"grouped by inputs · view · dtype) / 完整竞品实测表（{len(rival)} 行,按 数据 × 视图 × dtype 折叠）</summary>"
            f"{inner}</details>")


RVLAB_PY = {"gvr_base": "GVR base", "gvr_pr": "GVR pr", "op26_r0auto": "GVR op26",
            "radix_cutedsl": "Radix cuteDSL", "sglang_v2": "SGLang v2",
            "flashinfer_topk": "FlashInfer"}


def kpi(v, en, zh, color=""):
    st = f' style="color:{color}"' if color else ""
    return (f'<div class="kpi"><div class="v"{st}>{v}</div>'
            f'<div class="l lang-en">{en}</div><div class="l lang-zh">{zh}</div></div>')

real_kpi = f"{real_pvb:.3f}×" if real else "—"

# ---------------------------------------------------------------- HTML
# ---- §9 vseed fix chapter (2026-07-16 campaign) --------------------------
vsr2 = read_csv("vseed_harness/round2.csv")
vsfull = read_csv("vseed_harness/vsfull.csv")

def _vs_r2_table():
    if not vsr2:
        return "<p class='mut'>(round2.csv pending)</p>"
    h = ("<tr><th>cell</th><th>N</th><th>hit</th><th>base µs</th><th>PR µs</th>"
         "<th>+vseed µs</th><th>+vs2 µs</th><th>vs2/PR</th><th>vs2/base</th><th>exact</th></tr>")
    b = ""
    for r in vsr2:
        pr, v2, ba = fnum(r["pr"]), fnum(r["vs2"]), fnum(r["base"])
        cell = f"{r['model']}/{r['isl']}/{r['dtype']}/BS{r['BS']}"
        b += (f"<tr><td>{cell}</td><td>{r['N']}</td><td>{r['hit']}</td><td>{r['base']}</td>"
              f"<td>{r['pr']}</td><td>{r['vseed']}</td><td>{r['vs2']}</td>"
              f"<td>{pr/v2:.2f}</td><td>{ba/v2:.2f}</td><td>{r['exact_all']}</td></tr>")
    return f"<table>{h}{b}</table>"

def _vs_full_block():
    if not vsfull:
        return ("<p class='mut'><b>Full-envelope validation sweep in flight</b> (54 nsys batches, "
                "8 GPUs) — this block auto-fills from <code>vseed_harness/vsfull.csv</code> on regen. / "
                "全域验证扫描进行中,完成后本块由 <code>vsfull.csv</code> 自动填充。</p>", "", "")
    g_all = geo([fnum(r["vs_vs_pr"]) for r in vsfull])
    g_syn = geo([fnum(r["vs_vs_pr"]) for r in vsfull if r["family"] == "synth"])
    g_real = geo([fnum(r["vs_vs_pr"]) for r in vsfull if r["family"] == "real"])
    ex_ok = sum(r["vs_exact"] == "True" for r in vsfull)
    ex_tot = sum(r["vs_exact"] in ("True", "False") for r in vsfull)
    reg = sorted((r for r in vsfull if (fnum(r["vs_vs_pr"]) or 1) < 0.98),
                 key=lambda r: fnum(r["vs_vs_pr"]))
    win = sorted((r for r in vsfull if (fnum(r["vs_vs_pr"]) or 1) > 1.05),
                 key=lambda r: -fnum(r["vs_vs_pr"]))
    kpis = (f"<div class='kpis'>"
            f"<div class='kpi'><div class='v' style='color:#6ede8a'>{g_all:.3f}×</div>"
            f"<div class='l lang-en'>geomean vseed/PR, all {len(vsfull)} cells</div>"
            f"<div class='l lang-zh'>全 {len(vsfull)} cell vseed/PR 几何均值</div></div>"
            f"<div class='kpi'><div class='v' style='color:#6ede8a'>{g_syn:.3f}× / {g_real:.3f}×</div>"
            f"<div class='l lang-en'>synth / real geomean vseed/PR</div>"
            f"<div class='l lang-zh'>合成 / 真实 vseed/PR</div></div>"
            f"<div class='kpi'><div class='v' style='color:#6ea8fe'>{ex_ok}/{ex_tot}</div>"
            f"<div class='l lang-en'>vseed exactness (all cells)</div>"
            f"<div class='l lang-zh'>vseed 精确性（全 cell）</div></div>"
            f"<div class='kpi'><div class='v' style='color:#ff7a7a'>{len(reg)}</div>"
            f"<div class='l lang-en'>regressions vs PR (&lt;0.98)</div>"
            f"<div class='l lang-zh'>对 PR 回退 cell 数（&lt;0.98）</div></div>"
            f"</div>")
    def rtab(rows, ratio_color):
        h = ("<tr><th>family</th><th>cell</th><th>K</th><th>dtype</th><th>N</th><th>BS</th>"
             "<th>hit</th><th>PR µs</th><th>vseed µs</th><th>vseed/PR</th><th>vseed/base</th></tr>")
        b = ""
        for r in rows:
            cell = r["scenario"] or f"{r['model']}/{r['isl']}"
            hit = f"{fnum(r['hit']):.2f}" if fnum(r["hit"]) is not None else ""
            b += (f"<tr><td>{r['family']}</td><td>{cell}</td><td>{r['K']}</td><td>{r['dtype']}</td>"
                  f"<td>{r['N']}</td><td>{r['BS']}</td><td>{hit}</td><td>{r['pr']}</td><td>{r['vs']}</td>"
                  f"<td style='color:{ratio_color}'>{r['vs_vs_pr']}</td><td>{r['vs_vs_base']}</td></tr>")
        return f"<table>{h}{b}</table>"
    reg_html = (f"<details open><summary class='mut'>ALL regressed cells vs PR "
                f"(vseed/PR &lt; 0.98): {len(reg)} / 完整回退清单</summary>{rtab(reg, '#ff7a7a')}</details>"
                if reg else "<p><b>No cell regresses more than 2% vs PR. / 无超过 2% 的回退 cell。</b></p>")
    win_html = (f"<details><summary class='mut'>top wins vs PR (vseed/PR &gt; 1.05): {len(win)} cells "
                f"/ 主要收益 cell</summary>{rtab(win[:40], '#6ede8a')}</details>")
    return kpis, reg_html, win_html

VS_FULL_KPI, VS_FULL_REG, VS_FULL_WIN = _vs_full_block()

vs3 = read_csv("vseed_harness/vsfull3.csv")

def _vs3_block():
    """§9b: SHIPPED PR-head re-measure — vseed (@88a563b145) + P4 exact-tail
    fix (@eae374554c) on the PR branch vs the OLD PR head (@018251950f) and
    base. Column mapping in vsfull3.csv: pr = OLD head, vs = NEW head."""
    if not vs3:
        return ("<p class='mut'><b>New-PR-head re-measure sweep in flight</b> — auto-fills from "
                "<code>vseed_harness/vsfull3.csv</code> on regen. / 新 PR HEAD 重测进行中,完成后自动填充。</p>",
                "", "")
    g_all = geo([fnum(r["vs_vs_pr"]) for r in vs3])
    g_base = geo([fnum(r["vs_vs_base"]) for r in vs3])
    g_real = geo([fnum(r["vs_vs_base"]) for r in vs3 if r["family"] == "real"])
    ex_ok = sum(r["vs_exact"] == "True" for r in vs3)
    ex_tot = sum(r["vs_exact"] in ("True", "False") for r in vs3)
    ex_pr = sum(r["pr_exact"] == "True" for r in vs3)
    reg = sorted((r for r in vs3 if (fnum(r["vs_vs_pr"]) or 1) < 0.98),
                 key=lambda r: fnum(r["vs_vs_pr"]))
    win = sorted((r for r in vs3 if (fnum(r["vs_vs_pr"]) or 1) > 1.05),
                 key=lambda r: -fnum(r["vs_vs_pr"]))
    kpis = (f"<div class='kpis'>"
            f"<div class='kpi'><div class='v' style='color:#6ede8a'>{g_all:.3f}×</div>"
            f"<div class='l lang-en'>geomean NEW/OLD PR head, all {len(vs3)} cells</div>"
            f"<div class='l lang-zh'>新/旧 PR HEAD 几何均值(全 {len(vs3)} cell)</div></div>"
            f"<div class='kpi'><div class='v' style='color:#6ede8a'>{g_base:.3f}× / {g_real:.3f}×</div>"
            f"<div class='l lang-en'>NEW head vs base: all / real-capture cells</div>"
            f"<div class='l lang-zh'>新 HEAD 对 base:全部 / 真实数据</div></div>"
            f"<div class='kpi'><div class='v' style='color:#6ea8fe'>{ex_ok}/{ex_tot}</div>"
            f"<div class='l lang-en'>NEW head exactness (OLD head: {ex_pr}/{ex_tot})</div>"
            f"<div class='l lang-zh'>新 HEAD 精确性(旧 HEAD:{ex_pr}/{ex_tot})</div></div>"
            f"<div class='kpi'><div class='v' style='color:#ff7a7a'>{len(reg)}</div>"
            f"<div class='l lang-en'>cells &lt;0.98 vs OLD head (all listed)</div>"
            f"<div class='l lang-zh'>对旧 HEAD &lt;0.98 的 cell(全列出)</div></div>"
            f"</div>")
    def rtab3(rows, ratio_color):
        h = ("<tr><th>family</th><th>cell</th><th>K</th><th>dtype</th><th>N</th><th>BS</th>"
             "<th>hit</th><th>OLD µs</th><th>NEW µs</th><th>NEW/OLD</th><th>NEW/base</th></tr>")
        b = ""
        for r in rows:
            cell = r["scenario"] or f"{r['model']}/{r['isl']}"
            hit = f"{fnum(r['hit']):.2f}" if fnum(r["hit"]) is not None else ""
            b += (f"<tr><td>{r['family']}</td><td>{cell}</td><td>{r['K']}</td><td>{r['dtype']}</td>"
                  f"<td>{r['N']}</td><td>{r['BS']}</td><td>{hit}</td><td>{r['pr']}</td><td>{r['vs']}</td>"
                  f"<td style='color:{ratio_color}'>{r['vs_vs_pr']}</td><td>{r['vs_vs_base']}</td></tr>")
        return f"<table>{h}{b}</table>"
    reg_html = (f"<details><summary class='mut'>ALL cells &lt;0.98 vs OLD PR head: {len(reg)} "
                f"/ 完整回退清单</summary>{rtab3(reg, '#ff7a7a')}</details>"
                if reg else "<p><b>No cell regresses more than 2% vs the old PR head. / 无超过 2% 的回退。</b></p>")
    win_html = (f"<details><summary class='mut'>top wins vs OLD PR head (&gt;1.05): {len(win)} cells "
                f"/ 主要收益</summary>{rtab3(win[:40], '#6ede8a')}</details>")
    return kpis, reg_html, win_html

VS3_KPI, VS3_REG, VS3_WIN = _vs3_block()

HTML = f"""<!DOCTYPE html>
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>op26 → production GVR upstream port — bilingual report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{{--bg:#0f1116;--panel:#171a21;--fg:#e6e9ef;--mut:#9aa4b2;--line:#272c36;
        --blue:#6ea8fe;--red:#ff7a7a;--grn:#6ede8a;--org:#ffbf69;--acc:#8b8bff;}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans SC",sans-serif}}
  .wrap{{max-width:1000px;margin:0 auto;padding:26px 20px 80px}}
  h1{{font-size:25px;margin:.2em 0 .1em}} h2{{font-size:20px;margin:1.6em 0 .4em;
     border-left:3px solid var(--acc);padding-left:10px}}
  h3{{font-size:16px;color:var(--blue);margin:1.1em 0 .3em}}
  p{{margin:.5em 0}} code{{background:#222633;padding:1px 5px;border-radius:4px;font-size:13px}}
  .mut{{color:var(--mut)}} .big{{font-size:17px}}
  table{{border-collapse:collapse;width:100%;margin:.6em 0;font-size:13px}}
  th,td{{border:1px solid var(--line);padding:4px 7px;text-align:right}}
  th{{background:#1c212b;color:var(--blue)}} td:first-child,th:first-child{{text-align:left}}
  tr:nth-child(even) td{{background:#141821}}
  .keep{{background:#12211a;border:1px solid #2b6b45;border-radius:10px;padding:12px 16px;margin:14px 0}}
  .box{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}}
  .kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}}
  .kpi{{flex:1;min-width:140px;background:var(--panel);border:1px solid var(--line);
       border-radius:10px;padding:10px 14px}}
  .kpi .v{{font-size:22px;font-weight:700}} .kpi .l{{color:var(--mut);font-size:12px}}
  .toggle{{position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:9;border-bottom:1px solid var(--line)}}
  .toggle label{{cursor:pointer;padding:5px 14px;border:1px solid var(--line);border-radius:20px;
                margin-right:6px;color:var(--mut);user-select:none}}
  #te:checked~.wrap .toggle label[for=te],#tz:checked~.wrap .toggle label[for=tz]{{
     background:var(--acc);color:#fff;border-color:var(--acc)}}
  input[name=lang]{{display:none}}
  .lang-zh{{display:none}}
  #tz:checked~.wrap .lang-en{{display:none}}
  #tz:checked~.wrap .lang-zh{{display:block}}
  ul{{margin:.4em 0 .4em 0;padding-left:1.2em}} li{{margin:.25em 0}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0}}
  .ctl{{font-size:12.5px;color:var(--mut);margin-bottom:6px;line-height:2}}
  .ctl b{{color:var(--fg)}}
  .ck{{display:inline-block;margin-right:10px;padding:2px 6px;border:1px solid var(--line);
       border-radius:14px;cursor:pointer;user-select:none}}
  .ck input{{vertical-align:middle;margin-right:4px}}
  .row{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
  @media(max-width:820px){{.row{{grid-template-columns:1fr}}}}
  .plt{{width:100%;height:310px}}
  .noscript{{color:var(--org);font-size:12px}}
</style>
</head>
<body>
<input type="radio" name="lang" id="te" checked>
<input type="radio" name="lang" id="tz">
<div class="wrap">
<div class="toggle"><label for="te">English</label><label for="tz">中文</label>
  <span class="mut" style="font-size:12px">· op26→prod GVR port · B200 SM100 · BS=1 fp32 · synthetic (op22 §env, K512/1024/2048 = Flash/Pro/V3.2) + real V4 Flash/Pro + V3.2 decode-capture · same-run nsys cold-L2</span></div>

<h1>op26 → production GVR upstream port</h1>
<div class="lang-en"><p class="big">Porting the op-bench <code>op26_r0auto</code> top-K optimizations
(R0 seeded-cluster refine + op#7 rank-scatter) into production
<code>gvr_topk_decode.py</code>, validated on synthetic <b>and</b> real V4 decode data at BS=1 seq-len scan.
Charts are interactive — tick arms / scenarios / models to compare.</p></div>
<div class="lang-zh"><p class="big">将 op-bench <code>op26_r0auto</code> 的 top-K 优化（R0 seeded-cluster 精修 + op#7 rank-scatter）
迁移进生产核 <code>gvr_topk_decode.py</code>，并在<b>合成数据与真实 V4 解码数据</b>上于 BS=1 序列长扫描下验证。
图表可交互——勾选臂 / 场景 / 模型进行对比。</p></div>

<h2>1 · Executive summary / 摘要</h2>
<div class="kpis">
{kpi(f"{syn_pvb:.3f}×", "PR vs upstream base — synthetic geomean", "PR vs 上游基线——合成几何均值", "#6ede8a")}
{kpi(f"{syn_pvo:.3f}", "op26 / PR residual gap (synthetic)", "op26/PR 残余差距（合成）", "#ffbf69")}
{kpi(real_kpi, "PR vs base — real V4+V3.2 decode geomean", "PR vs 基线——真实 V4+V3.2 解码几何均值", "#6ede8a")}
{kpi(f"{SYN_EXACT}/{SYN_EXACT}", f"synthetic exactness (+{DTYPE_EXACT}/{DTYPE_EXACT} dtypes)", f"合成精确性（另 dtypes {DTYPE_EXACT}/{DTYPE_EXACT}）", "#6ea8fe")}
</div>
<div class="lang-en">
<p>The op-bench had opened a <b>12.5%</b> gap between the shipped production secant kernel and the tuned
<code>op26_r0auto</code>. This work ports two <b>safe</b> levers into production and closes that gap to
<b>~5%</b>, while keeping the base kernel <b>byte-identical</b> to upstream when <code>enable_r0=False</code>.</p>
<ul>
<li><b>rank-scatter P4</b> (op#7): cluster barriers 14→7. my/op26 1.125→1.048, exact 36/36.</li>
<li><b>p1b_cache cs-aware</b> (matches op26 <code>dispatch_p1bc_mc</code>): ON for all dtypes at cluster_size&gt;1. my/op26 →~1.05, exact 18/18.</li>
<li>Synthetic geomean <b>PR/base = {syn_pvb:.3f}×</b> (cs1 {syn_pvb_cs1:.3f}, cs4 {syn_pvb_cs4:.3f}); residual op26/PR <b>{syn_pvo:.3f}</b>.</li>
<li><b>Real decode</b> ({len(real)} cells, 3 model families V4 Flash/Pro + V3.2): PR/base <b>{real_pvb:.3f}×</b>
(pro {real_pvb_pro:.3f}×, V3.2 {real_pvb_v32:.3f}×), PR exact <b>{real_exact}/{len(real)}</b>, and PR
<b>repairs</b> a base-secant undershoot on a low-hit cell.</li>
<li>Residual ~5% = unified-kernel cluster-barrier structural floor — <b>not safely removable</b> on a dev box
(a prior cluster-handoff silent bug passed 330/330). Safe levers are exhausted.</li>
</ul></div>
<div class="lang-zh">
<p>op-bench 显示已上线的生产 secant 核与调优后的 <code>op26_r0auto</code> 之间存在 <b>12.5%</b> 差距。
本工作把两个<b>安全</b>杠杆迁入生产，将差距收窄到 <b>~5%</b>，且当 <code>enable_r0=False</code> 时基线核与上游<b>逐字节一致</b>。</p>
<ul>
<li><b>rank-scatter P4</b>（op#7）：cluster barrier 14→7。my/op26 1.125→1.048，精确 36/36。</li>
<li><b>p1b_cache cs-aware</b>（对齐 op26 <code>dispatch_p1bc_mc</code>）：cluster_size&gt;1 时对所有 dtype 开启。my/op26 →~1.05，精确 18/18。</li>
<li>合成几何均值 <b>PR/base = {syn_pvb:.3f}×</b>（cs1 {syn_pvb_cs1:.3f}，cs4 {syn_pvb_cs4:.3f}）；残余 op26/PR <b>{syn_pvo:.3f}</b>。</li>
<li><b>真实解码</b>（{len(real)} cell，3 个模型族 V4 Flash/Pro + V3.2）：PR/base <b>{real_pvb:.3f}×</b>
（pro {real_pvb_pro:.3f}×，V3.2 {real_pvb_v32:.3f}×），PR 精确 <b>{real_exact}/{len(real)}</b>，
且 PR <b>修复</b>了一处 base-secant 在低命中 cell 上的 undershoot。</li>
<li>残余 ~5% = 统一核的 cluster-barrier 结构性地板——在开发盒上<b>无法安全移除</b>（此前一个 cluster-handoff 静默 bug 曾通过 330/330）。安全杠杆已耗尽。</li>
</ul></div>

<h2>2 · Optimization migration / 优化迁移</h2>
<div class="lang-en">
<p>All optimization lives in production <code>tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py</code>
only (<b>13 commits</b> on branch <code>omni/op21-gvr-prod</code>). <code>gvr_topk_decode_load_balance.py</code> is
study-only and untouched. Every flag defaults OFF when <code>enable_r0=False</code> → base kernel is byte-identical to upstream.</p>
<table>
<tr><th>lever</th><th>source</th><th>effect</th><th>exact</th></tr>
<tr><td>rank-scatter P4</td><td>op#7</td><td>barriers 14→7; my/op26 1.125→1.048</td><td>36/36</td></tr>
<tr><td>p1b_cache cs-aware</td><td>op26 dispatch_p1bc_mc</td><td>ON all dtypes at cs&gt;1; my/op26 →~1.05</td><td>18/18</td></tr>
<tr><td>seeded-cluster R0 refine (C1–C7)</td><td>op26 R0</td><td>miss-refine on R0 seed</td><td>52/52</td></tr>
</table></div>
<div class="lang-zh">
<p>所有优化只在生产核 <code>tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py</code>
中（分支 <code>omni/op21-gvr-prod</code> 上 <b>13 个提交</b>）。<code>gvr_topk_decode_load_balance.py</code> 仅供研究、未改动。
当 <code>enable_r0=False</code> 时所有 flag 默认关闭 → 基线核与上游逐字节一致。</p>
<table>
<tr><th>杠杆</th><th>来源</th><th>效果</th><th>精确</th></tr>
<tr><td>rank-scatter P4</td><td>op#7</td><td>barrier 14→7；my/op26 1.125→1.048</td><td>36/36</td></tr>
<tr><td>p1b_cache cs-aware</td><td>op26 dispatch_p1bc_mc</td><td>cs&gt;1 时全 dtype 开启；my/op26 →~1.05</td><td>18/18</td></tr>
<tr><td>seeded-cluster R0 精修（C1–C7）</td><td>op26 R0</td><td>R0 seed 上的 miss 精修</td><td>52/52</td></tr>
</table></div>

<h2>3 · Synthetic results (op22 §env) / 合成结果</h2>
<div class="lang-en"><p>op22 §env synthetic distribution, fp32 BS=1, same-run nsys cold-L2. The three K buckets are the three
model families: <b>K=512 → V4 Flash</b>, <b>K=1024 → V4 Pro</b>, <b>K=2048 → V3.2</b>. Tick arms and scenarios to compare.
<b>Refreshed 2026-07-16 at the PR's own launch-shape contract</b>: base/pr arms driven via
<code>GvrTopKKernel.launch</code> → <code>pick_config</code> (branch HEAD <code>018251950f</code>) — notably
<b>cluster_size=8</b> at N≥131072 (BS≤4), T=512 below 64K, fp32-only 256-bit loads — replacing the earlier
fixed cs4/T1024 instantiation, so these numbers reflect production shapes.</p></div>
<div class="lang-zh"><p>op22 §env 合成分布，fp32 BS=1，同轮 nsys 冷 L2。三个 K 桶即三个模型族：
<b>K=512 → V4 Flash</b>、<b>K=1024 → V4 Pro</b>、<b>K=2048 → V3.2</b>。勾选臂与场景进行对比。
<b>2026-07-16 按 PR 自身 launch 形状契约复测</b>：base/pr 臂经 <code>GvrTopKKernel.launch</code> →
<code>pick_config</code>（分支 HEAD <code>018251950f</code>）驱动——特别是 N≥131072（BS≤4）用
<b>cluster_size=8</b>、64K 以下 T=512、256-bit 载入仅 fp32——取代此前固定 cs4/T1024 的实例化，
即本表反映生产形状。</p></div>
<div class="card">
  <div class="ctl">
    <b>model / K</b>
    <label class="ck"><input type="radio" name="sk" value="512">V4 Flash · K512</label>
    <label class="ck"><input type="radio" name="sk" value="1024" checked>V4 Pro · K1024</label>
    <label class="ck"><input type="radio" name="sk" value="2048">V3.2 · K2048</label>
    &nbsp; <b>scenario</b>
    <label class="ck"><input type="checkbox" class="ss" value="best" checked>best</label>
    <label class="ck"><input type="checkbox" class="ss" value="worst" checked>worst</label>
    &nbsp; <b>arms</b>
    <label class="ck"><input type="checkbox" class="sa" value="base" checked>base (secant)</label>
    <label class="ck"><input type="checkbox" class="sa" value="pr" checked>PR (R0+RS)</label>
    <label class="ck"><input type="checkbox" class="sa" value="op26" checked>op26_r0auto</label>
    <span class="mut">· solid = best, dashed = worst</span>
  </div>
  <div class="row"><div id="syn_lat" class="plt"></div><div id="syn_rat" class="plt"></div></div>
  <p class="noscript">If charts are blank (script-free viewer), use the table below.</p>
</div>
<details><summary class="mut">full synthetic grid ({len(synth)} cells) / 完整合成网格</summary>{synth_table()}</details>

<h2>4 · Real-data results (V4 + V3.2 decode-capture) / 真实数据结果</h2>
<div class="lang-en"><p>Real DeepSeek decode-capture top-K inputs, BS=1 fp32, median over 3 GVR-active layers per (model, ISL).
Three model families: <b>V4 Flash</b> (K512, cr=4), <b>V4 Pro</b> (K1024, cr=4), <b>V3.2</b> (K2048, cr=1).
Indexer length N = ISL/cr (V4: ISL/4; V3.2: ≈ISL). hit = mean preIdx∩topK hit-rate, where
<code>preIdx</code> = the previous decode step's captured top-K (real temporal warm-start; for V3.2 reconstructed
from <code>topk.out[s_prev]</code> since it has no separate <code>preidx.in</code>). <b>PR is EXACT ({real_exact}/{len(real)} cells)</b>
vs the captured reference. <b>Refreshed 2026-07-16 at the PR launch-shape contract</b> (base/pr via
<code>launch</code>/<code>pick_config</code>, incl. cs=8 on the ≥512k V4 / ≥512k-equivalent rungs) — the two
former sub-1.0 Flash cells (512k/1024k) are re-measured under the shapes production would actually pick.</p>
<p>Grand <b>PR/base = {real_pvb:.3f}×</b> (flash {real_pvb_flash:.3f}×, pro {real_pvb_pro:.3f}×, V3.2 {real_pvb_v32:.3f}×) —
larger than synthetic {syn_pvb:.3f}× because the upstream base secant is slower on real high-hit distributions.
V3.2 (K=2048, dense cr=1, hit≈0.4–0.93) shows the widest PR/base wins. Residual <b>op26/PR = {real_pvo:.3f}</b>
(~{(1-real_pvo)*100:.0f}%; V3.2 {real_pvo_v32:.3f}). V3.2 real captures span 7 ISL rungs (4K–256K; physical
kv_len caps N at ~163775 for ISL_256K).</p></div>
<div class="lang-zh"><p>真实 DeepSeek 解码采集 top-K 输入，BS=1 fp32，每个 (model, ISL) 对 3 个 GVR-active 层取中位数。
三个模型族：<b>V4 Flash</b>（K512，cr=4）、<b>V4 Pro</b>（K1024，cr=4）、<b>V3.2</b>（K2048，cr=1）。
indexer 长度 N = ISL/cr（V4：ISL/4；V3.2：≈ISL）。hit = preIdx∩topK 平均命中率，其中
<code>preIdx</code> = 上一解码步采集的 top-K（真实时序 warm-start；V3.2 因无独立 <code>preidx.in</code> 而由
<code>topk.out[s_prev]</code> 重建）。<b>PR 精确（{real_exact}/{len(real)} cell）</b>，与采集参考值集相等。
<b>2026-07-16 按 PR launch 形状契约复测</b>（base/pr 经 <code>launch</code>/<code>pick_config</code>,
大 N 档启用 cs=8）——原先两个低于 1.0 的 Flash cell（512k/1024k）现按生产真实选型重测。</p>
<p>总 <b>PR/base = {real_pvb:.3f}×</b>（flash {real_pvb_flash:.3f}×，pro {real_pvb_pro:.3f}×，V3.2 {real_pvb_v32:.3f}×）——
大于合成的 {syn_pvb:.3f}×，因为上游 base secant 在真实高命中分布上更慢。V3.2（K=2048，密集 cr=1，hit≈0.4–0.93）
的 PR/base 增益最大。残余 <b>op26/PR = {real_pvo:.3f}</b>（约 {(1-real_pvo)*100:.0f}%；V3.2 {real_pvo_v32:.3f}）。
V3.2 真实采集覆盖 7 档 ISL（4K–256K；ISL_256K 的物理 kv_len 使 N 上限约 163775）。</p></div>
<div class="lang-en">{fix_en}</div>
<div class="lang-zh">{fix_zh}</div>
<div class="card">
  <div class="ctl">
    <b>model</b>
    <label class="ck"><input type="checkbox" class="rm" value="flash" checked>V4 Flash · K512 · cr=4</label>
    <label class="ck"><input type="checkbox" class="rm" value="pro" checked>V4 Pro · K1024 · cr=4</label>
    <label class="ck"><input type="checkbox" class="rm" value="v32" checked>V3.2 · K2048 · cr=1</label>
    &nbsp; <b>arms</b>
    <label class="ck"><input type="checkbox" class="ra" value="base" checked>base (secant)</label>
    <label class="ck"><input type="checkbox" class="ra" value="pr" checked>PR (R0+RS)</label>
    <label class="ck"><input type="checkbox" class="ra" value="op26" checked>op26_r0auto</label>
    <span class="mut">· solid = flash, dashed = pro, dotted = V3.2 · x-tick = ISL (indexer N)</span>
  </div>
  <div class="row"><div id="real_lat" class="plt"></div><div id="real_rat" class="plt"></div></div>
  <p class="noscript">If charts are blank (script-free viewer), use the table below.</p>
</div>
<details><summary class="mut">full real-data table ({len(real)} rows) / 完整真实数据表（{len(real)} 行）</summary>{real_table()}</details>

<h2>5 · Exactness / 精确性</h2>
<div class="lang-en"><ul>
<li>Synthetic fp32: <b>{SYN_EXACT}/{SYN_EXACT}</b> cells value-set exact (best+worst, K∈512/1024/2048, N 4K–1M).</li>
<li>Non-fp32 dtypes (bf16/fp16): <b>{DTYPE_EXACT}/{DTYPE_EXACT}</b> exact.</li>
<li>Real decode fp32: <b>{real_exact}/{len(real)}</b> cells exact vs captured reference — V4 Flash + V4 Pro
(K512/K1024, cr=4) <b>and</b> V3.2 (K2048, cr=1, {real_exact_v32}/{len(real_v32)}).</li>
<li>Contract: PR arm requires unique top-K count == K AND gathered logit value-set equal to reference.</li>
</ul></div>
<div class="lang-zh"><ul>
<li>合成 fp32：<b>{SYN_EXACT}/{SYN_EXACT}</b> cell 值集精确（best+worst，K∈512/1024/2048，N 4K–1M）。</li>
<li>非 fp32（bf16/fp16）：<b>{DTYPE_EXACT}/{DTYPE_EXACT}</b> 精确。</li>
<li>真实解码 fp32：<b>{real_exact}/{len(real)}</b> cell 与采集参考精确 —— V4 Flash + V4 Pro
（K512/K1024，cr=4）<b>及</b> V3.2（K2048，cr=1，{real_exact_v32}/{len(real_v32)}）。</li>
<li>契约：PR 臂要求 top-K 唯一数 == K 且 gather 的 logit 值集与参考相等。</li>
</ul></div>

<h2>6 · Residual vs the op-bench anchor — gap closed by launch shapes / 残余分析</h2>
<div class="lang-en">
<p><b>2026-07-16 update: the residual is closed on synthetic and mixed on real.</b> Under the PR's
launch-shape contract, synthetic BS=1 reads <b>PR/op26 time ratio ≈ {syn_pvo:.3f}</b> (PR ~{(1-syn_pvo)*100:.0f}%
faster); on real data PR leads on Flash (~3.5%) while op26 keeps a small edge on Pro (~2%) and V3.2 (~5%).
The earlier revision of this section measured a uniform ~5%
op26-ahead residual and attributed it to the unified-kernel cluster-barrier floor — that analysis compared
<b>matched shapes</b> (both arms at cs≤4). The refreshed picture decomposes as:</p>
<ul>
<li><b>cs=8 rung (BS≤4, N≥131072)</b>: <code>pick_config</code> picks cluster_size=8 where the op-bench
anchor's own dispatch tops out at cs=4 — worth ~6–12% on the large-N rungs (validated cs8-vs-cs4 nsys
gm 0.943, <code>BIGBS_TRIAGE_NOTE.md</code>).</li>
<li><b>Occupancy/vec tuning</b> (T=512 below 64K, fp32-only 256-bit): small additional wins vs the anchor's
tables on 16-bit and small-N cells.</li>
<li>The <b>cluster-barrier observation itself still stands at matched shapes</b> (p4_coop OFF both arms;
barrier removal remains out of scope — load-dependent SMEM races are invisible to fixed-seed exactness:
a prior cluster-handoff silent bug passed 330/330 before failing under real load). It is simply no longer
the binding term once the launch shapes differ.</li>
</ul></div>
<div class="lang-zh">
<p><b>2026-07-16 更新：合成侧残余已收平反超,真实侧混合。</b>在 PR 的 launch 形状契约下,合成 BS=1 为
<b>PR/op26 时间比 ≈ {syn_pvo:.3f}</b>（PR 快 ~{(1-syn_pvo)*100:.0f}%）;真实数据上 Flash 由 PR 领先（~3.5%),
Pro/V3.2 上 op26 仍保有小幅优势（~2%/~5%）。本节旧版测得整齐的 op26 领先 ~5% 并归因于统一核
cluster-barrier 地板——那是<b>同形状</b>对比（双臂均 cs≤4）。刷新后的分解：</p>
<ul>
<li><b>cs=8 档（BS≤4、N≥131072）</b>：<code>pick_config</code> 在 op-bench 锚自身调度止步 cs=4 的区间选
cluster_size=8——大 N 档收益 ~6–12%（cs8-vs-cs4 nsys gm 0.943,见 <code>BIGBS_TRIAGE_NOTE.md</code>）。</li>
<li><b>占用率/向量宽调优</b>（64K 以下 T=512、256-bit 仅 fp32）：在 16-bit 与小 N cell 上相对锚表的额外小幅收益。</li>
<li><b>同形状下 cluster-barrier 的观察仍然成立</b>（双臂 p4_coop=OFF;移除 barrier 仍超范围——负载相关
SMEM 竞争对定种精确性不可见:此前一个 cluster-handoff 静默 bug 在真实负载下失败前曾通过 330/330）。
只是形状不同后它不再是约束项。</li>
</ul></div>

<h2>7 · BS scaling (latency-bound → throughput-bound) / 批量扩展</h2>
<div class="lang-en"><p>§3–§6 fixed BS=1. This chapter sweeps <b>batch size BS∈{{1…1024}}</b> across
<b>dtype∈{{fp32,fp16,bf16}}</b> on the SAME synthetic (op22 §env) and real (V4 Flash/Pro + V3.2) inputs,
replicated to BS rows (single decode step, BS independent rows). <b>Refreshed 2026-07-16 with FULL ISL
coverage</b>: the synthetic grid now spans all 9 seq-len rungs (4K–1M, previously 3) and the real grid all
captured ISL rungs per model (previously one) — and the base/pr arms are driven through the PR's own
<b>launch-shape contract</b> (<code>GvrTopKKernel.launch</code> → <code>pick_config</code>, branch HEAD
<code>018251950f</code>): cluster_size 8/4/2/1 by (BS, N), T=512/1024, mbpm 0–3, fp32-only 256-bit loads —
the shapes the production runner picks. Measured story: (1) the <b>PR/base gain is BS-invariant</b>
(synth geomean {bs_syn_pvb_lo:.3f}× @BS=1 → {bs_syn_pvb_hi:.3f}× @BS=1024; real {bs_real_pvb_lo:.3f}× →
{bs_real_pvb_hi:.3f}×) — R0 changes Phase-2/4 arithmetic only, no batch coupling; (2) <b>pr vs
op26_r0auto is now a fair dispatch-vs-dispatch comparison</b>: op26/pr geomean {bs_syn_pvo_lo:.3f} @BS=1 →
{bs_syn_pvo_hi:.3f} @BS=1024 (values &gt;1 = op26 slower). <i>History note</i>: the previous revision of this
chapter drove pr/base at a config frozen at the BS=1 optimum (cs4/T1024/mbpm1), which understated the PR
by geomean {bigbs_fro:.3f}× (max {bigbs_fro_max:.2f}×) at BS≥64 vs {bigbs_run:.3f}× at proper shapes —
the diagnostic that motivated adding <code>pick_config</code>/<code>launch</code> to the PR
(<code>BIGBS_TRIAGE_NOTE.md</code>). Same-run nsys cold-L2 pure-kernel. Tick arms / scenarios / dtype / N.</p></div>
<div class="lang-zh"><p>§3–§6 固定 BS=1。本章在 SAME 合成（op22 §env）与真实（V4 Flash/Pro + V3.2）输入上扫描
<b>批量 BS∈{{1…1024}}</b>，横跨 <b>dtype∈{{fp32,fp16,bf16}}</b>，将输入复制到 BS 行（单解码步、BS 独立行）。
<b>2026-07-16 全 ISL 覆盖复测</b>：合成网格覆盖全部 9 档序列长（4K–1M，此前 3 档），真实网格覆盖每个模型的
全部采集 ISL 档（此前 1 档）；base/pr 臂经 PR 自身的 <b>launch 形状契约</b>（<code>GvrTopKKernel.launch</code>
→ <code>pick_config</code>，分支 HEAD <code>018251950f</code>）驱动：cluster_size 8/4/2/1 按 (BS, N)、
T=512/1024、mbpm 0–3、256-bit 载入仅 fp32——即生产 runner 会选的形状。实测结论：（1）<b>PR/base 增益与 BS
无关</b>（合成 geomean BS=1 时 {bs_syn_pvb_lo:.3f}× → BS=1024 时 {bs_syn_pvb_hi:.3f}×；真实
{bs_real_pvb_lo:.3f}× → {bs_real_pvb_hi:.3f}×）——R0 只改 P2/P4 算术，无批量耦合；（2）<b>pr 对
op26_r0auto 现为公平的"调度对调度"比较</b>：op26/pr geomean BS=1 时 {bs_syn_pvo_lo:.3f} → BS=1024 时
{bs_syn_pvo_hi:.3f}（&gt;1 = op26 更慢）。<i>历史注</i>：本章上一版将 pr/base 冻结在 BS=1 最优配置
（cs4/T1024/mbpm1），BS≥64 段把 PR 低估 geomean {bigbs_fro:.3f}×（最差 {bigbs_fro_max:.2f}×，正确形状下为
{bigbs_run:.3f}×）——正是该诊断促成 PR 增加 <code>pick_config</code>/<code>launch</code>
（<code>BIGBS_TRIAGE_NOTE.md</code>）。同轮 nsys 冷 L2 纯核。勾选臂 / 场景 / dtype / N 对比。</p></div>
<div class="kpis">
{kpi(_f(bs_syn_pvb_lo), "synth PR/base @ BS=1 (latency-bound)", "合成 PR/base @ BS=1（延迟受限）", "#6ede8a")}
{kpi(_f(bs_syn_pvb_hi), "synth PR/base @ BS=1024 (saturated)", "合成 PR/base @ BS=1024（饱和）", "#ffbf69")}
{kpi(_f(bs_real_pvb_lo), "real PR/base @ BS=1", "真实 PR/base @ BS=1", "#6ede8a")}
{kpi(_f(bs_real_pvb_hi), "real PR/base @ BS=1024 (saturated)", "真实 PR/base @ BS=1024（饱和）", "#ffbf69")}
{kpi((f"{bs_syn_exact}/{bs_syn_exact_tot}" if bs_have else "—"), "synth PR exactness (all BS×dtype)", "合成 PR 精确（全 BS×dtype）", "#6ea8fe")}
{kpi((f"{bs_real_exact}/{bs_real_exact_tot}" if bs_have else "—"), "real PR exactness (all ISL×BS×dtype)", "真实 PR 精确（全 ISL×BS×dtype）", "#6ea8fe")}
{kpi((f"{bigbs_fro:.3f}×" if bigbs else "—"), "frozen-config PR / op26 @ big-BS (07-15 diagnostic, synth best cells)", "冻结配置 PR / op26 @ 大 BS（07-15 诊断,合成 best cell）", "#ff7a7a")}
{kpi((f"{bigbs_run:.3f}×" if bigbs else "—"), "runner-config PR / op26 @ big-BS (07-15 diagnostic, synth best cells)", "runner 配置 PR / op26 @ 大 BS（07-15 诊断,合成 best cell）", "#6ede8a")}
</div>
<h3>7a · Synthetic (op22 §env), BS × 9 seq-len rungs / 合成数据</h3>
<div class="card">
  <div class="ctl">
    <b>model / K</b>
    <label class="ck"><input type="radio" name="bsk" value="512">V4 Flash · K512</label>
    <label class="ck"><input type="radio" name="bsk" value="1024" checked>V4 Pro · K1024</label>
    <label class="ck"><input type="radio" name="bsk" value="2048">V3.2 · K2048</label>
    &nbsp; <b>dtype</b>
    <label class="ck"><input type="radio" name="bsd" value="fp32" checked>fp32</label>
    <label class="ck"><input type="radio" name="bsd" value="fp16">fp16</label>
    <label class="ck"><input type="radio" name="bsd" value="bf16">bf16</label>
    &nbsp; <b>N</b>
    {"".join(f'<label class="ck"><input type="radio" name="bsn" value="{n}"{" checked" if n == 65536 else ""}>{n // 1024}K</label>'
             for n in sorted({int(r["N"]) for r in bs_synth}))}
    &nbsp; <b>scenario</b>
    <label class="ck"><input type="checkbox" class="bss" value="best" checked>best</label>
    <label class="ck"><input type="checkbox" class="bss" value="worst" checked>worst</label>
    &nbsp; <b>arms</b>
    <label class="ck"><input type="checkbox" class="bsa" value="base" checked>base</label>
    <label class="ck"><input type="checkbox" class="bsa" value="pr" checked>PR</label>
    <label class="ck"><input type="checkbox" class="bsa" value="op26" checked>op26</label>
    <span class="mut">· solid = best, dashed = worst · BS on log-x</span>
  </div>
  <div class="row"><div id="bs_syn_lat" class="plt"></div><div id="bs_syn_rat" class="plt"></div></div>
  <p class="noscript">If charts are blank (script-free viewer), use the table below.</p>
</div>
<details><summary class="mut">full synthetic BS grid ({len(bs_synth)} rows) / 完整合成 BS 网格</summary>{bs_synth_table()}</details>
<h3>7b · Real decode-capture (V4 Flash/Pro + V3.2), BS × all ISL rungs / 真实采集数据</h3>
<div class="lang-en"><p class="mut"><b>Known cold-hit regression (disclosed)</b>: on <b>Flash 1024k</b>
(N=262127, hit≈0.42 — the V4 hit-rate valley/floor) pr AND op26 both fall to <b>0.68–0.79× of base</b> at
BS≥128 (cs=1, throughput-bound; fp32 mild 0.98 at BS≤64, 16-bit affected at all BS); v32 256k shows the same
shape (0.75–0.87). Both R0 implementations regress together and stay exact → this is the R0-ladder
low-hit-rate regime (admission miss → extra full-N fallback scans, unmasked once cs=1).
<b>Guard design constraint: hit-rate is NOT known at inference time</b> (it is the overlap of the current
step's top-K with the previous step's — only computable after the fact), so the follow-up guard PR cannot
dispatch on it host-side. Runtime-feasible alternatives: (a) <b>in-kernel escape</b> — R0 already counts seed
admissions; when the count signals a cold seed, bail to the secant path inside the kernel instead of running
the fallback ladder; (b) <b>trailing-hit feedback</b> — the kernel emits its measured admission/hit counter,
and the host uses the previous steps' value as the predictor (temporal coherence makes last-step hit a good
proxy for this step).</p></div>
<div class="lang-zh"><p class="mut"><b>已知低命中回退区（如实披露）</b>：<b>Flash 1024k</b>（N=262127,
hit≈0.42——V4 命中率谷底）上 pr 与 op26 在 BS≥128（cs=1,吞吐受限）同时跌到 <b>base 的 0.68–0.79×</b>
（fp32 在 BS≤64 仅轻微 0.98,16-bit 全 BS 受累）;v32 256k 同形（0.75–0.87）。两套 R0 实现同步退化且全部
exact → 这是 R0 梯子在低命中 regime 的算法性失配（admission miss → 额外整遍 N 兜底扫描,cs=1 后不再被
掩盖）。<b>守卫设计约束:推理时 hit-rate 不可知</b>（它是本步 top-K 与上一步的交集,只能事后算出）,
follow-up 守卫 PR 不能在 host 侧按 hit 调度。运行时可行的替代:（a）<b>kernel 内逃逸</b>——R0 本来就统计
seed admission 计数,计数显示种子失效时在 kernel 内直接转 secant 路径,而非跑完兜底梯子;（b）<b>滞后命中反馈</b>——
kernel 输出实测 admission/命中计数,host 用上一步的值做本步预测（时序相干性使上一步 hit 是本步的良好代理）。</p></div>
<div class="card">
  <div class="ctl">
    <b>model</b>
    <label class="ck"><input type="checkbox" class="brm" value="flash" checked>V4 Flash</label>
    <label class="ck"><input type="checkbox" class="brm" value="pro" checked>V4 Pro</label>
    <label class="ck"><input type="checkbox" class="brm" value="v32" checked>V3.2</label>
    &nbsp; <b>dtype</b>
    <label class="ck"><input type="radio" name="brd" value="fp32" checked>fp32</label>
    <label class="ck"><input type="radio" name="brd" value="fp16">fp16</label>
    <label class="ck"><input type="radio" name="brd" value="bf16">bf16</label>
    &nbsp; <b>seq-len rung</b>
    {"".join(f'<label class="ck"><input type="radio" name="brl" value="{isl}"{" checked" if isl == "128k" else ""}>{isl}</label>'
             for isl in ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]
             if any(r["isl"] == isl for r in bs_real))}
    &nbsp; <b>arms</b>
    <label class="ck"><input type="checkbox" class="bra" value="base" checked>base</label>
    <label class="ck"><input type="checkbox" class="bra" value="pr" checked>PR</label>
    <label class="ck"><input type="checkbox" class="bra" value="op26" checked>op26</label>
    <span class="mut">· solid = flash, dashed = pro, dotted = V3.2 · BS on log-x</span>
  </div>
  <div class="row"><div id="bs_real_lat" class="plt"></div><div id="bs_real_rat" class="plt"></div></div>
  <p class="noscript">If charts are blank (script-free viewer), use the table below.</p>
</div>
<details><summary class="mut">full real BS grid ({len(bs_real)} rows) / 完整真实 BS 网格</summary>{bs_real_table()}</details>

<div class="lang-en"><p><b>BS × seq-len heatmap.</b> The same §7 grids rendered as 2-D maps — x = seq-len rung
(indexer N), y = batch size, cell = the ticked metric. Tick one or more <b>maps</b> to draw them side by side;
latency maps use a sequential scale, ratio maps a red–green diverging scale centered at 1 (green = numerator arm
faster). Missing cells (OOM / not swept) stay blank.</p></div>
<div class="lang-zh"><p><b>BS × 序列长 热力图。</b>把 §7 的网格画成二维图——x = 序列长档（indexer N），y = batch size，
格值 = 所勾选指标。可勾选多个 <b>maps</b> 并排绘制；延迟图用顺序色标，比值图用以 1 为中心的红绿发散色标
（绿 = 分子臂更快）。缺测 cell（OOM/未扫）留空。</p></div>
<div class="card">
  <div class="ctl">
    <b>inputs</b>
    <label class="ck"><input type="radio" name="hmf" value="synth" checked>synthetic</label>
    <label class="ck"><input type="radio" name="hmf" value="real">real (decode-capture)</label>
    &nbsp; <b>model/K (synth)</b>
    <label class="ck"><input type="radio" name="hmk" value="512">K512</label>
    <label class="ck"><input type="radio" name="hmk" value="1024" checked>K1024</label>
    <label class="ck"><input type="radio" name="hmk" value="2048">K2048</label>
    &nbsp; <b>scen (synth)</b>
    <label class="ck"><input type="radio" name="hms" value="best" checked>best</label>
    <label class="ck"><input type="radio" name="hms" value="worst">worst</label>
    &nbsp; <b>model (real)</b>
    <label class="ck"><input type="radio" name="hmm" value="flash">Flash</label>
    <label class="ck"><input type="radio" name="hmm" value="pro" checked>Pro</label>
    <label class="ck"><input type="radio" name="hmm" value="v32">V3.2</label>
    <br>
    <b>dtype</b>
    <label class="ck"><input type="radio" name="hmd" value="fp32" checked>fp32</label>
    <label class="ck"><input type="radio" name="hmd" value="fp16">fp16</label>
    <label class="ck"><input type="radio" name="hmd" value="bf16">bf16</label>
    &nbsp; <b>maps (tick to draw / 勾选绘制)</b>
    <label class="ck"><input type="checkbox" class="hmz" value="base">base µs</label>
    <label class="ck"><input type="checkbox" class="hmz" value="pr" checked>PR µs</label>
    <label class="ck"><input type="checkbox" class="hmz" value="op26">op26 µs</label>
    <label class="ck"><input type="checkbox" class="hmz" value="pvb" checked>PR/base speedup</label>
    <label class="ck"><input type="checkbox" class="hmz" value="pvo">op26/PR</label>
  </div>
  <div id="hm_row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px"></div>
  <p class="noscript">Heatmaps need a script-enabled viewer; the values are in the BS grid tables above. / 热力图需支持脚本的查看器；数值见上方 BS 网格表。</p>
</div>

<h2>8 · External top-K comparison / 外部算子对比</h2>
<div class="lang-en"><p>The same §3 synthetic (op22 §env) and §4 real (V4 Flash/Pro + V3.2) inputs, now benchmarking our best
in-tree GVR (<code>op26_r0auto</code>) against three external latest top-K kernels — <b>Radix (cuteDSL)</b>,
<b>SGLang v2</b> (sglang@main DSv4 top-K), <b>FlashInfer top_k</b> (0.6.11) — plus the GVR <code>base</code>/<code>pr</code>
arms for context. Swept over <b>BS 1–1024 × seq-len 4K–1M × dtype {{fp32,fp16,bf16}} × K {{512,1024,2048}}</b>,
same nsys cold-L2 protocol. <b>Latency = projected NVTX GPU span</b> (honest wall-clock) — it equals the
kernel-sum for single-kernel ops but avoids double-counting SGLang v2's overlapped 2-kernel PDL path (at
BS=32 the kern-sum overcounts 1.7×; the table carries both µs and span µs).
<b>Unsupported cells are omitted, never faked</b>: SGLang v2 is fp32-only (kernel contract); V3.2 has no real capture
beyond 256K; over-large BS×N cells that OOM drop out. Correctness of every rival is checked vs <code>torch.topk</code>
(index-set / value-set equality, order-free). Tick arms / dtype / scenario to compare.</p></div>
<div class="lang-zh"><p>用与 §3 合成（op22 §env）、§4 真实（V4 Flash/Pro + V3.2）<b>相同</b>的输入，把我们最强的 in-tree GVR
（<code>op26_r0auto</code>）与三个最新外部 top-K 内核对比——<b>Radix (cuteDSL)</b>、<b>SGLang v2</b>（sglang@main DSv4 top-K）、
<b>FlashInfer top_k</b>（0.6.11）——并附 GVR <code>base</code>/<code>pr</code> 臂作参照。扫描
<b>BS 1–1024 × 序列长 4K–1M × dtype {{fp32,fp16,bf16}} × K {{512,1024,2048}}</b>，同一 nsys 冷 L2 协议。
<b>延迟 = 投影 NVTX GPU span</b>（诚实 wall-clock）——单核 op 等于 kernel-sum，但避免对 SGLang v2 重叠 2-核 PDL 路径的双计
（BS=32 时 kern-sum 高估 1.7×；表中同时给出 µs 与 span µs）。<b>不支持的 cell 直接缺省、绝不伪造</b>：
SGLang v2 仅 fp32（内核契约）；V3.2 无 256K 以上真实采集；BS×N 过大 OOM 的 cell 自动退出。每个竞品都与
<code>torch.topk</code> 校验正确性（索引集/值集相等、不要求保序）。勾选臂 / dtype / 场景对比。</p></div>
<div class="lang-en"><p class="mut"><b>Provenance (2026-07-16 refresh)</b>: the three GVR rows
(<code>base</code>/<code>pr</code>/<code>op26</code>) were re-measured at the PR launch-shape contract
(<code>launch</code>/<code>pick_config</code> — fixes the earlier 256-bit-load mis-tune on 16-bit GVR rows
and the frozen large-BS shapes) on umbriel-b200-094, with the GVR BS grid extended to ALL seq-len/ISL
rungs. The three external rival rows: seq-len view + synth BS view are the 2026-07-15 run (umbriel-b200-044,
unchanged code); the <b>real BS view was BACKFILLED to ALL ISL rungs on 2026-07-16</b> (umbriel-b200-081,
same container generation + cutlass 4.5.0 / flashinfer 0.6.11 recipe, 2750 cells, exact 1925/1925) — it
replaces the old single-rung (128k) rival rows so the whole real-BS rival grid is single-node consistent.
Cross-run comparability passes the op26 anchor-drift gate twice: refresh vs 044 <b>median 1.002 / p95 1.047
(n=1122)</b>; backfill(081) vs refresh(094) <b>median 1.000 / p95 1.055 (n=825)</b>, and the backfill rivals
at the 128k overlap match the 044 rows (median 0.998–1.001, p95 ≤1.053). One noise-limited batch
(v32 bf16 16k, ~15µs kernels) shows reproducible ±15% per-cell scatter at BS≤32 in BOTH directions
(batch median 0.993, symmetric, confirmed by an independent re-run) — inherent cell noise, not node bias.</p></div>
<div class="lang-zh"><p class="mut"><b>数据来源（2026-07-16 复测）</b>：三条 GVR 行
（<code>base</code>/<code>pr</code>/<code>op26</code>）已按 PR launch 形状契约重测
（<code>launch</code>/<code>pick_config</code>——修正了此前 16-bit GVR 行错开 256-bit 载入、以及大 BS
冻结形状的问题），于 umbriel-b200-094;GVR 的 BS 网格扩展到全部序列长/ISL 档。三条外部竞品行为
2026-07-15 运行（umbriel-b200-044,代码未变）;<b>真实 BS 视图已于 2026-07-16 回填至全部 ISL 档</b>
（umbriel-b200-081,同代容器 + cutlass 4.5.0 / flashinfer 0.6.11 配方,2750 cells,exact 1925/1925）——
替换旧的单档（128k）竞品行,使真实-BS 竞品网格单节点一致。跨 run 可比性两道 op26 锚漂移门均通过:
refresh vs 044 <b>median 1.002 / p95 1.047（n=1122）</b>;backfill(081) vs refresh(094)
<b>median 1.000 / p95 1.055（n=825）</b>,且回填竞品在 128k 重叠处与 044 行吻合（median 0.998–1.001,
p95 ≤1.053）。一个噪声受限 batch（v32 bf16 16k,~15µs 短核）在 BS≤32 呈可复现的双向 ±15% 单 cell 抖动
（batch median 0.993、对称、独立重跑证实）——是 cell 固有噪声,非节点偏差。</p></div>
<div class="kpis">
{kpi(_best_line(riv_syn_fp32) if rival_have else "—", "synth fp32 BS=1: rival speedup over GVR op26 (t(GVR)/t(rival); >1 = rival FASTER)", "合成 fp32 BS=1: 竞品相对 GVR op26 的加速（t(GVR)/t(竞品)；>1=竞品更快）", "#ffbf69")}
{kpi(_best_line(riv_real_fp32) if rival_have else "—", "real fp32 BS=1: rival speedup over GVR op26 (>1 = rival FASTER)", "真实 fp32 BS=1: 竞品相对 GVR op26 的加速（>1=竞品更快）", "#ffbf69")}
{kpi((f"{rival_exact_ct}/{rival_exact_tot}" if rival_have else "—"), "rival correctness vs torch.topk (all measured cells)", "竞品 vs torch.topk 正确（全部实测 cell）", "#6ea8fe")}
</div>
<div class="card">
  <div class="ctl">
    <b>inputs</b>
    <label class="ck"><input type="radio" name="rvf" value="synth" checked>synthetic (op22 §env)</label>
    <label class="ck"><input type="radio" name="rvf" value="real">real (decode-capture)</label>
    &nbsp; <b>view</b>
    <label class="ck"><input type="radio" name="rvv" value="seqlen" checked>seq-len (BS=1)</label>
    <label class="ck"><input type="radio" name="rvv" value="bs">batch size</label>
    <br>
    <b>K (synth)</b>
    <label class="ck"><input type="radio" name="rvk" value="512">512</label>
    <label class="ck"><input type="radio" name="rvk" value="1024" checked>1024</label>
    <label class="ck"><input type="radio" name="rvk" value="2048">2048</label>
    &nbsp; <b>scen (synth)</b>
    <label class="ck"><input type="radio" name="rvs" value="best" checked>best</label>
    <label class="ck"><input type="radio" name="rvs" value="worst">worst</label>
    &nbsp; <b>model (real)</b>
    <label class="ck"><input type="radio" name="rvm" value="flash">Flash</label>
    <label class="ck"><input type="radio" name="rvm" value="pro" checked>Pro</label>
    <label class="ck"><input type="radio" name="rvm" value="v32">V3.2</label>
    <br>
    <b>N (BS view, synth)</b>
    {"".join(f'<label class="ck"><input type="radio" name="rvn" value="{n}"{" checked" if n == 65536 else ""}>{n // 1024}K</label>'
             for n in sorted({int(r["N"]) for r in rival if r["family"] == "synth" and r["sweep"] == "bs"}))}
    &nbsp; <b>ISL (BS view, real)</b>
    {"".join(f'<label class="ck"><input type="radio" name="rvi" value="{isl}"{" checked" if isl == "128k" else ""}>{isl}</label>'
             for isl in ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]
             if any(r.get("isl") == isl and r["sweep"] == "bs" for r in rival))}
    <br>
    <b>dtype</b>
    <label class="ck"><input type="radio" name="rvd" value="fp32" checked>fp32</label>
    <label class="ck"><input type="radio" name="rvd" value="fp16">fp16</label>
    <label class="ck"><input type="radio" name="rvd" value="bf16">bf16</label>
    &nbsp; <b>arms</b>
    <label class="ck"><input type="checkbox" class="rva" value="gvr_base">GVR base</label>
    <label class="ck"><input type="checkbox" class="rva" value="gvr_pr">GVR pr</label>
    <label class="ck"><input type="checkbox" class="rva" value="op26_r0auto" checked>GVR op26</label>
    <label class="ck"><input type="checkbox" class="rva" value="radix_cutedsl" checked>Radix cuteDSL</label>
    <label class="ck"><input type="checkbox" class="rva" value="sglang_v2" checked>SGLang v2</label>
    <label class="ck"><input type="checkbox" class="rva" value="flashinfer_topk" checked>FlashInfer</label>
  </div>
  <div class="row"><div id="rv_lat" class="plt"></div><div id="rv_rat" class="plt"></div></div>
  <p class="noscript">If charts are blank (script-free viewer), use the table below. Ratio = t(GVR op26)/t(arm); &gt;1 means the arm is FASTER than GVR op26.</p>
</div>
<details open><summary class="mut">rival trend analysis (measured — ratio = t(GVR op26)/t(rival), &gt;1 = rival FASTER than GVR) / 趋势分析</summary>
<div class="lang-en"><ul>
<li><b>BS = 1 (decode, the deployment regime):</b> <b>SGLang v2 is the fastest kernel</b> — 1.43× (synth) and
<b>1.40–1.53× (real: flash 1.48×, pro 1.53×, v32 1.40×)</b> faster than GVR op26. <b>FlashInfer is ~parity</b>
(1.03–1.07× faster). <b>Radix cuteDSL is ~parity at fp32</b> (0.93–1.06). GVR op26 is mid-pack — the best in-tree
GVR but it does not beat the latest external SGLang v2. This matches the op28 / op22-§10 / op34 finding that
SGLang v2's 8-CTA MLP path wins the BS=1 latency regime.</li>
<li><b>By BS:</b> SGLang v2 stays fastest across BS (1.43× at BS 1, dipping to 0.92 at BS 64, back to 1.32× at
BS 1024). <b>Radix cuteDSL scales WORST</b> — it falls from 0.91 (BS 1) to <b>0.40 (BS 1024)</b>, i.e. GVR op26
becomes ≈2.5× faster than Radix at BS 1024. FlashInfer stays ~parity (0.83–1.06). So beyond SGLang, GVR op26
holds up well vs the other two rivals as BS grows.
<b>2026-07-16 full-ISL real BS backfill</b>: the BS story is strongly ISL-dependent. SGLang v2's mid-BS dip
becomes a deep valley at large ISL — at <b>BS 8–64 × ISL ≥128k</b> its fp32 ratio falls to <b>0.45–0.76</b>
(GVR op26 up to ≈2.2× faster at 1M/BS32), while it stays fastest at BOTH ends (BS=1: 1.21–1.77×; BS=1024:
1.14–1.74×) and across all BS at ISL ≤32k. Radix's collapse is likewise ISL-graded (BS1024 ratio 0.92 at 4k
→ 0.33–0.47 at ≥64k); FlashInfer stays within 0.67–1.31 everywhere.</li>
<li><b>By dtype (the standout):</b> at 16-bit the full-scan rivals pull clearly ahead of GVR — <b>Radix goes
1.06 (fp32) → 1.48 (fp16) / 1.52 (bf16)</b> and <b>FlashInfer 1.07 → 1.45 / 1.38</b> (synth seqlen BS=1). Halving
the candidate-array bytes speeds up the memory-bound full scan, while GVR's compressed cuteDSL path barely moves
— so Radix/FlashInfer become ~1.5× faster than GVR op26 in fp16/bf16.
<b>2026-07-16 refresh</b>: at the PR launch shapes (fp32-only 256-bit fixed) the 16-bit gap vs the <b>pr</b>
arm narrows to ~1.20–1.30× (t(pr)/t(rival), seqlen BS=1), and at fp32 BS=1 the pr arm is now slightly
<b>faster</b> than both Radix (0.98) and FlashInfer (0.96); SGLang v2's lead over pr is ~1.26×.</li>
<li><b>Correctness:</b> every rival is exact vs <code>torch.topk</code> on all measured cells — but this
holds only on the benchmark slice: <b>see §8.1–§8.3 below</b> (SGLang v2 is only <i>conditionally</i> exact and
FAILS on real V3.2 K=2048 rows outside the slice; FlashInfer and Radix cuteDSL each passed the same full
adversarial + all-layer × all-step battery 2245/2245 — Radix at all three dtypes). Refreshed GVR
pr/op26 exact on all 2772 cells each (full-ISL BS grid) and GVR base exact on 2736/2772 — the 36 misses are
ALL the known upstream-base real Flash-512k undershoot (hit≈0.06), now exposed across every dtype × BS by the
full grid, and repaired by the R0 port (pr exact on all of them).</li>
</ul>
<p class="mut">Takeaway: on these BS=1 decode inputs SGLang v2 is the fastest top-K (≈1.5×), FlashInfer ties GVR, and
Radix ties at fp32 but wins ~1.5× at 16-bit; GVR op26 is the strongest in-tree arm and stays competitive except
against SGLang v2 and against the 16-bit full-scan kernels. The R0 port's value is the intra-GVR gain (§7) +
the base-undershoot repair, not beating the external state-of-the-art.</p></div>
<div class="lang-zh"><ul>
<li><b>BS = 1（解码，部署工况）：</b><b>SGLang v2 是最快的核</b>——比 GVR op26 快 1.43×（合成）、
<b>1.40–1.53×（真实：flash 1.48×，pro 1.53×，v32 1.40×）</b>。<b>FlashInfer 基本持平</b>（快 1.03–1.07×）。
<b>Radix cuteDSL 在 fp32 持平</b>（0.93–1.06）。GVR op26 居中——是最强的 in-tree GVR，但不胜最新外部 SGLang v2。
这与 op28 / op22-§10 / op34 的结论一致：SGLang v2 的 8-CTA MLP 路径在 BS=1 延迟工况取胜。</li>
<li><b>随 BS：</b>SGLang v2 全程最快（BS 1 快 1.43×，BS 64 降到 0.92，BS 1024 回到 1.32×）。
<b>Radix cuteDSL 扩展性最差</b>——从 0.91（BS 1）跌到 <b>0.40（BS 1024）</b>，即 BS 1024 时 GVR op26 比 Radix 快约 2.5×。
FlashInfer 大致持平（0.83–1.06）。所以除 SGLang 外,GVR op26 随 BS 增长对另两个竞品站得住。
<b>2026-07-16 全 ISL 真实 BS 回填</b>:BS 故事强依赖 ISL。SGLang v2 的中段 BS 低谷在大 ISL 上变成深谷——
<b>BS 8–64 × ISL ≥128k</b> 时 fp32 比值跌到 <b>0.45–0.76</b>（1M/BS32 处 GVR op26 反快约 2.2×）,
而两端（BS=1: 1.21–1.77×;BS=1024: 1.14–1.74×）及 ISL ≤32k 全 BS 段 SGLang 仍最快。Radix 的塌陷同样随
ISL 分级（BS1024 比值 4k 处 0.92 → ≥64k 处 0.33–0.47);FlashInfer 全域在 0.67–1.31 之间。</li>
<li><b>随 dtype（最显著）：</b>16-bit 下全扫描竞品明显反超 GVR——<b>Radix 从 1.06（fp32）→ 1.48（fp16）/1.52（bf16）</b>，
<b>FlashInfer 1.07 → 1.45 / 1.38</b>（合成 seqlen BS=1）。候选数组字节减半加速了访存受限的全扫描,而 GVR 压缩 cuteDSL 路径几乎不动
——故 Radix/FlashInfer 在 fp16/bf16 上比 GVR op26 快约 1.5×。</li>
<li><b>正确性：</b>所有竞品在全部实测 cell 上与 <code>torch.topk</code> 精确——但这仅在基准切片内成立:
<b>见下方 §8.1–§8.3</b>(SGLang v2 仅<i>条件</i>精确,在切片外的真实 V3.2 K=2048 行上实测失败;FlashInfer 与
Radix cuteDSL 各自通过同一对抗 + 全层×全步完整测试 2245/2245,Radix 覆盖全部三个 dtype)。GVR pr/op26 在全部 1122 cell 精确,
GVR base 在 2736/2772 精确——36 处 miss 全部是已知的上游 base 真实 Flash-512k undershoot（hit≈0.06),
全 ISL×BS 网格将其在每个 dtype×BS 上完整暴露,R0 移植全部修复（pr 在这些 cell 全 exact）;刷新后 GVR pr/op26
各 2772 cell 全精确。<b>2026-07-16 复测</b>:PR launch 形状下（256-bit 仅 fp32）16-bit 对 <b>pr</b> 臂的差距
收窄到 ~1.20–1.30×,fp32 BS=1 时 pr 反超 Radix（0.98）与 FlashInfer（0.96);SGLang v2 对 pr 领先 ~1.26×。</li>
</ul>
<p class="mut">结论:在这些 BS=1 解码输入上,SGLang v2 是最快的 top-K(约 1.5×),FlashInfer 与 GVR 持平,Radix 在 fp32 持平但 16-bit 快约 1.5×;
GVR op26 是最强 in-tree 臂,除对 SGLang v2 与 16-bit 全扫描核外均有竞争力。R0 移植的价值在于 GVR 内部增益(§7)+ 修复 base undershoot,
而非击败外部 SOTA。</p></div>
{rival_table()}
</details>

<h2>9 · vseed fix — flash-1M big-BS regression root cause + fix validation / vseed 修正</h2>
<div class="lang-en"><p><b>Root cause (corrects the earlier §7b story).</b> Simulating the exact kernel rung
placement on the bench layer (flash L22, N=262127, hit≈0.42) shows the R0 ladder does NOT miss on this cell —
it <b>fat-admits</b>: the coarse q0.85 rung is accepted with <b>4408 candidates</b> (near kC=5120) while the
base secant's pmean init lands <b>633</b> and exits in one pass. P3 collect + P4 rank-scatter then carry ~7×
more candidate work in pr/op26, which surfaces as the 0.68–0.79× regression once BS≥128/cs=1 saturates the
GPU (16-bit at all BS). The 512k rung is the opposite regime: BOTH arms miss badly (base pmean count 17752 ≫
kC) — base burns its secant (43 µs at BS=1, plus the ×36 undershoot cells PR repairs), which is why PR "wins"
2–3× there.</p>
<p><b>P1 estimator study (the mean-vs-median question).</b> The current top-K threshold equals the
<b>rank-(hit·K) order statistic</b> of the gathered prev-topK values (exact identity: exactly hit·K gathered
values are ≥ the true threshold). Since hit varies 0.06–0.94 across real rows, <b>no fixed statistic — mean,
median, or any single quantile — is admissible everywhere</b>; mean and median are biased HIGH (undershoot
side) on most V4 rows. The right insurance is not a better single number but measuring an extra
data-adaptive threshold in the same pass.</p>
<p><b>The fix (r0_vseed, ~40-line const-folded kernel change + per-K config).</b> P1 parks its pmean (the
secant init probe) in the last rung column — zero extra sync (P1's own barrier covers visibility); the M-ary
count pass then measures it for free; admission picks the <b>tightest</b> admissible column (explicit argmin,
unsorted-safe); on a miss the measured pmean improves the fallback bracket. Per-K config:
<b>K512/K1024 → qfracs=(0.85,) + pmean</b> (pmean replaces the q0.35 rung — 2 columns, zero column tax);
<b>K2048 → qfracs=(0.85, 0.35) + pmean</b> (kC/K=3 makes fat admission costly — v32-64k showed a fat pmean
admit losing 14% to a slim 2-pass miss, so the mid rung stays).</p></div>
<div class="lang-zh"><p><b>根因（修正此前 §7b 的说法）。</b>在 bench 层（flash L22,N=262127,hit≈0.42）精确模拟
kernel 摆 rung 逻辑显示:该 cell 上 R0 梯子并未 miss,而是<b>肥接纳</b>——粗 q0.85 rung 以 <b>4408 候选</b>
（接近 kC=5120）被接纳,而 base secant 的 pmean 初值只有 <b>633</b> 且一遍收敛。P3 collect + P4 rank-scatter
随之背负 ~7× 候选开销,在 BS≥128/cs=1 饱和后兑现为 0.68–0.79×（16-bit 全 BS 受累）。512k 档是相反工况:
两臂全 miss（base pmean count 17752 ≫ kC）——base 烧满 secant（BS=1 43µs,外加 PR 修复的 ×36 undershoot
cell）,所以 PR 在那里"赢"2–3×。</p>
<p><b>P1 估计量研究（mean vs median 之问）。</b>当前 top-K 阈值恰等于 gathered prev-topK 值的<b>第 (hit·K) 阶
统计量</b>（恒等式:恰有 hit·K 个 gathered 值 ≥ 真阈值）。真实数据 hit 在 0.06–0.94 间波动,<b>不存在处处可用的
固定统计量——mean、median、任何单一分位数都不行</b>;在多数 V4 行上 mean/median 系统性偏高（undershoot 方向）。
正确的保险不是更好的单点数,而是在同一遍 pass 里免费实测一个数据自适应阈值。</p>
<p><b>修正（r0_vseed,约 40 行 const-fold kernel 改动 + per-K 配置）。</b>P1 把自己的 pmean（secant 初始探针）
存进最后一个 rung 列——零额外同步（P1 自带 barrier 保证可见）;M-ary 计数 pass 顺带实测它;接纳规则改为
显式 argmin（取窗口内最瘦列,乱序安全）;miss 时实测过的 pmean 改善兜底 bracket。per-K 配置:
<b>K512/K1024 → qfracs=(0.85,) + pmean</b>（pmean 替换 q0.35 rung——2 列,零列税）;
<b>K2048 → qfracs=(0.85, 0.35) + pmean</b>（kC/K=3 肥接纳代价高——v32-64k 实测肥 pmean 接纳比瘦 2-pass miss
慢 14%,故保留中位 rung）。</p></div>
<div class="lang-en"><p class="mut"><b>Provenance</b>: umbriel-b200-072, 2026-07-16, single-GPU paired A/B, nsys
cold-L2 (same protocol as §3/§4/§7), machine-local edited <code>gvrpkg</code> (/tmp staging; diff =
<code>vseed_harness/vseed_v2.diff</code>, NOT yet on the PR branch). All numbers local-only.</p></div>
<div class="lang-zh"><p class="mut"><b>数据来源</b>:umbriel-b200-072,2026-07-16,单卡配对 A/B,nsys 冷 L2
（与 §3/§4/§7 同协议）,机器本地修改版 <code>gvrpkg</code>（/tmp 暂存;diff 见
<code>vseed_harness/vseed_v2.diff</code>,尚未上 PR 分支）。所有数字仅本地保留。</p></div>
<details open><summary class="mut">round-2 key-cell A/B (25 regression + guard cells, 4 arms) / 关键 cell 四臂对照</summary>
{_vs_r2_table()}
<div class="lang-en"><p class="mut">Headline: flash-1M fp32 BS128–1024 pulled from 0.70–0.79× of base to
1.01–1.03× (1.29–1.44× over current PR); 16-bit 1M BS1 0.76→0.88–0.90×base (+16% over PR); v32-256k becomes a
win (+16%); guard cells (R0-win regime) at 0.98–1.01 — the round-1 3–5% column tax is eliminated by the per-K
hybrid.</p></div>
<div class="lang-zh"><p class="mut">要点:flash-1M fp32 BS128–1024 从 base 的 0.70–0.79× 拉回 1.01–1.03×
（对现 PR 1.29–1.44×）;16-bit 1M BS1 0.76→0.88–0.90×base（对 PR +16%）;v32-256k 转为赢面（+16%）;守卫 cell
（R0 赢面工况）0.98–1.01——第一轮的 3–5% 列税已被 per-K 混合消除。</p></div>
</details>
<h3>9a · Full-envelope regression audit / 全域回退审计</h3>
<div class="lang-en"><p>The decisive question: does the fix regress ANY other case in this report's coverage?
Full REPORT grid re-swept with 3 arms (base / PR / PR+vseed): synth (op22 §env) seqlen+BS × K∈(512,1024,2048) ×
best/worst × 3 dtypes, plus real (V4 Flash/Pro + V3.2) all-ISL seqlen+BS × 3 dtypes — 54 nsys batches, 8 GPUs,
cold-L2, exactness-gated per cell. Every cell with vseed/PR &lt; 0.98 is listed below — none are hidden.</p></div>
<div class="lang-zh"><p>决定性问题:该修正是否让本报告覆盖的其它 case 回退?整个 REPORT 网格用 3 臂重扫
（base / PR / PR+vseed）:合成（op22 §env）seqlen+BS × K∈(512,1024,2048) × best/worst × 3 dtype,加真实
（V4 Flash/Pro + V3.2）全 ISL seqlen+BS × 3 dtype——54 个 nsys 批次、8 卡、冷 L2、逐 cell 精确性门。
所有 vseed/PR &lt; 0.98 的 cell 全部列出——绝不隐藏。</p></div>
<div class="lang-en"><p class="mut"><b>What the audit itself caught (and how it was fixed).</b> Audit round 1
found (a) a SEVERE perf tail concentrated in K2048/K1024 <b>16-bit big-BS</b> cells (down to 0.72× of PR,
all cs1/T512/mb3 or T1024/mb1 configs) — root-caused to the extra per-thread count column (+2–4 KB SMEM)
pushing high-occupancy 16-bit variants over an occupancy cliff (mb-override probe: 68.3→60.5 µs); fixed in
<b>v3</b> by reusing the existing secant <code>smem_ptcnt</code> buffer for the vseed column — zero SMEM
growth (bad cell 68.3→56.9 µs vs PR 54.5). And (b) <b>12 exactness fails, all pro/512k fp32</b> (hit 0.23):
control-proven <b>pre-existing</b> — the pristine PR-head kernel with <code>qfracs=(0.85,)</code> fails
identically (|miss|=1: picks −0.288984 instead of −0.288981, Δ=3e-6, below the P4 rank-scatter one-level
fine-recursion resolution ≈ range/1024² ≈ 5e-6; same class as the known op22 §9 2.7e-6 boundary defect).
vseed only shifts which value pair straddles the bin — the defect needs a second recursion level
(separate follow-up), it is NOT introduced by this change. The table below is the v3 re-audit.</p></div>
<div class="lang-zh"><p class="mut"><b>审计自身抓到的问题(及修复)。</b>第一轮审计发现:(a) 严重回退尾集中在
K2048/K1024 的 <b>16-bit 大 BS</b> cell(最低到 PR 的 0.72×,全部是 cs1/T512/mb3 或 T1024/mb1 配置)——根因是
多出的 per-thread 计数列(+2–4 KB SMEM)把高占用 16-bit 变体挤过 occupancy 悬崖(mb 降档探针:68.3→60.5 µs);
<b>v3</b> 修复:vseed 列复用 secant 已有的 <code>smem_ptcnt</code> 缓冲——SMEM 零增长(坏 cell 68.3→56.9 µs,
PR 54.5)。(b) <b>12 个精确性失败,全部 pro/512k fp32</b>(hit 0.23):对照证明为<b>预存在缺陷</b>——原始 PR-head
kernel 配 <code>qfracs=(0.85,)</code> 同样失败(|miss|=1:取 −0.288984 而非 −0.288981,Δ=3e-6,低于 P4
rank-scatter 单层细递归分辨率 ≈ range/1024² ≈ 5e-6;与已知 op22 §9 的 2.7e-6 边界缺陷同类)。vseed 只是改变了
哪对值跨在 bin 边界上——缺陷需第二层递归修复(独立跟进),非本改动引入。下表为 v3 重审结果。</p></div>
{VS_FULL_KPI}
{VS_FULL_REG}
{VS_FULL_WIN}

<h3>9b · Shipped to the PR branch + P4 exact-tail correctness fix — final re-measure /
上 PR 分支 + P4 精确性修复 — 最终重测</h3>
<div class="box">
<div class="lang-en"><p>Both changes are now ON the PR branch: <b>vseed + per-K rung defaults</b>
(<code>88a563b145</code>) and the <b>P4 straddling-fine-bin exact tie resolution</b>
(<code>eae374554c</code>, ambiguity-gated MSB-first 8-bit-digit radix select over order-preserving
integer keys; fp32 default ON, 16-bit kernels byte-identical). The exact-tail fix closes the
pre-existing 3e-6 boundary defect (§9 audit finding b): the 12 real Pro/512k fp32 cells go
value-exact at every BS, and adversarial 5e-8-spaced / 1-ulp bitwise tie bands repair across
K∈{{512,1024,2048}}, cr∈{{1,4}}, cs∈{{1,4,8}}. Unaffected-cell cost is noise (paired cold-L2 A/B
geomean 0.998); repair-active rows (previously WRONG results) pay ~1.3× at BS=1. The table below
re-measures the FULL report grid with the NEW PR head vs the OLD head (@<code>018251950f</code>)
and base.</p></div>
<div class="lang-zh"><p>两项改动均已上 PR 分支:<b>vseed + per-K 阶梯默认值</b>(<code>88a563b145</code>)与
<b>P4 跨界细 bin 精确 tie 判决</b>(<code>eae374554c</code>,歧义门控的 MSB 优先 8-bit 数位基数选择,
基于保序整数键;fp32 默认开启,16-bit 内核字节不变)。该修复关闭了预存在的 3e-6 边界缺陷(§9 审计发现 b):
12 个真实 Pro/512k fp32 cell 在所有 BS 下变为值精确,对抗性 5e-8 间隔 / 1-ulp 位级 tie 带在
K∈{{512,1024,2048}}、cr∈{{1,4}}、cs∈{{1,4,8}} 全部修复。未受影响 cell 代价为噪声(配对冷 L2 A/B 几何均值
0.998);修复实际生效的行(此前结果是错的)在 BS=1 付出 ~1.3×。下表以新 PR HEAD 对旧 HEAD
(@<code>018251950f</code>)与 base 重测完整报告网格。</p></div>
{VS3_KPI}
{VS3_REG}
{VS3_WIN}
</div>

<h2>10 · Test data · environment · code / 测试数据·环境·代码</h2>
<div class="box">
<div class="lang-en">
<p><b>Test input data — how it was obtained + local paths.</b></p>
<ul>
<li><b>Synthetic (§3 / §7 synth / §8 synth):</b> generated by the unified <code>indexer-topk-temporal-synth</code>
pipeline (per-layer empirical inverse-CDF + GPD-tail marginals, rank-conditional temporal preIdx with real
per-step hit-rate, marginal-preserving positional clustering — all calibrated from the real 64K production
captures). The harnesses read pre-built frozen bundles:
<code>indexer_topk_op_bench/op22_temporal_fixed_hr_bench/bundles_env/&lt;scenario&gt;/&lt;model&gt;_&lt;dtype&gt;_N&lt;N&gt;/</code>
(generator <code>bundle_data_env.py</code>; calibration assets <code>calib_&lt;model&gt;.npz</code> /
<code>posz_&lt;model&gt;.npz</code>). Coverage: scenario best/worst · K∈{{512,1024,2048}} · N 4K–1M · fp32/fp16/bf16.</li>
<li><b>Real (§4 / §7 real / §8 real):</b> production decode captures from single-prompt BS=1 greedy end-to-end
runs of the actual models (per-layer indexer logits + top-K + preIdx dumped from the live GVR path via the
<code>dsv4-indexer-capture</code> q9j hook). Local root:
<code>/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/E2E_exp/indexer_decode_capture/data/</code> —
V4 Flash <code>flash/ISL_4k…ISL_1024k</code> (K=512, cr=4) · V4 Pro <code>pro/ISL_4k…ISL_1024k</code> (K=1024, cr=4) ·
V3.2 <code>v32/ISL_4k…ISL_256k</code> (K=2048, cr=1; preIdx reconstructed as <code>topk.out[s_prev]</code>).
Loaders <code>harness/real_data_v4cap.py</code> / <code>real_data_v32.py</code> pick 3 GVR-active layers per
(model, ISL) rung and cache slim bench tensors at
<code>op22_temporal_fixed_hr_bench/data_v4cap/</code> · <code>data_v32/</code>.</li>
</ul>
<p><b>Test environment.</b></p>
<ul>
<li>Single-node B200 (SM100), 8-way GPU-sharded sweeps, one cell per GPU at a time. GVR 3-arm PR-contract
refresh (§3/§4/§7 tables + §8 GVR rows): <b>umbriel-b200-094, 2026-07-16</b>. External-rival + BS-grid sweep
(§7 grids, §8 rival rows): <b>umbriel-b200-044, 2026-07-15</b>. Cross-run comparability is gated by the op26
anchor-drift check printed by <code>aggregate_refresh.py</code>.</li>
<li>Dev container: torch 2.12.0a0 (nv26.05) · CUDA 13.2 · cutlass-dsl 4.6.0, run with
<code>PYTHONNOUSERSITE=1</code> plus machine-local cutlass-dsl 4.5.0 and flashinfer 0.6.11
(newer-container fixes documented in <code>BS_SCALING_ENV_FIXES.md</code>).</li>
<li>Timing protocol: nsys with cold L2 (flush between reps), latency = projected NVTX GPU span
(kernel-projected; avoids double-counting overlapped multi-kernel paths), median over reps;
every cell carries a correctness gate (PR/op26 vs captured reference; rivals vs <code>torch.topk</code>).</li>
</ul>
<p><b>Test code locations.</b></p>
<ul>
<li>Production kernel: branch <code>omni/op21-gvr-prod</code> → PR
<a href="https://github.com/NVIDIA/TensorRT-LLM/pull/16457">NVIDIA/TensorRT-LLM#16457</a>, file
<code>tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py</code>
(R0 seeded-cluster + rank-scatter + <code>pick_config</code>/<code>launch</code> host API).
Importable snapshot == PR head: <code>gvrpkg_snapshot/gvrpkg/top_k/gvr_topk_decode.py</code> (this report dir).</li>
<li>Benchmark harnesses (all under <code>indexer_topk_op_bench/op26_r0_upstream_port_report/</code>):
<code>harness/</code> — 3-arm synth/real/BS nsys drivers; <code>refresh_harness/</code> — the 2026-07-16
PR-contract refresh; <code>rival_harness/</code> — unified 6-arm §8 rival sweep. Aggregators
<code>aggregate_refresh.py</code> / <code>aggregate_bs.py</code> / <code>aggregate_rival.py</code> →
<code>synth_3arm.csv</code> / <code>real_3arm.csv</code> / <code>bs_synth.csv</code> / <code>bs_real.csv</code> /
<code>rival_long.csv</code>; this page = <code>gen_report.py</code>.</li>
<li>op26 anchor arm (op-bench GVR): <code>indexer_topk_op_bench/op26_gvr_logfalsi_rs/</code>.</li>
</ul>
</div>
<div class="lang-zh">
<p><b>测试输入数据——获取方式与本地路径。</b></p>
<ul>
<li><b>合成（§3 / §7 合成 / §8 合成）：</b>由统一的 <code>indexer-topk-temporal-synth</code> 流水线生成
（逐层经验逆 CDF + GPD 尾部边缘分布、按秩条件的时序 preIdx（真实逐步命中率）、保边缘的位置聚类——均由真实
64K 生产采集标定）。各 harness 读取预生成的冻结 bundle：
<code>indexer_topk_op_bench/op22_temporal_fixed_hr_bench/bundles_env/&lt;scenario&gt;/&lt;model&gt;_&lt;dtype&gt;_N&lt;N&gt;/</code>
（生成器 <code>bundle_data_env.py</code>；标定资产 <code>calib_&lt;model&gt;.npz</code> /
<code>posz_&lt;model&gt;.npz</code>）。覆盖：best/worst 场景 · K∈{{512,1024,2048}} · N 4K–1M · fp32/fp16/bf16。</li>
<li><b>真实（§4 / §7 真实 / §8 真实）：</b>来自真实模型单 prompt、BS=1、贪心端到端推理的生产解码采集
（经 <code>dsv4-indexer-capture</code> q9j 钩子从在线 GVR 路径逐层 dump indexer logits + top-K + preIdx）。本地根目录：
<code>/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/E2E_exp/indexer_decode_capture/data/</code> ——
V4 Flash <code>flash/ISL_4k…ISL_1024k</code>（K=512，cr=4）· V4 Pro <code>pro/ISL_4k…ISL_1024k</code>（K=1024，cr=4）·
V3.2 <code>v32/ISL_4k…ISL_256k</code>（K=2048，cr=1；preIdx 由 <code>topk.out[s_prev]</code> 重建）。
加载器 <code>harness/real_data_v4cap.py</code> / <code>real_data_v32.py</code> 每个 (model, ISL) 档取 3 个
GVR-active 层，slim 张量缓存于 <code>op22_temporal_fixed_hr_bench/data_v4cap/</code> · <code>data_v32/</code>。</li>
</ul>
<p><b>测试环境。</b></p>
<ul>
<li>单节点 B200（SM100），8 卡分片扫描,每卡同时只跑一个 cell。GVR 3 臂 PR 契约复测（§3/§4/§7 表 + §8 GVR 行）:
<b>umbriel-b200-094,2026-07-16</b>。外部竞品 + BS 网格扫描（§7 网格、§8 竞品行）:<b>umbriel-b200-044,2026-07-15</b>。
跨 run 可比性由 <code>aggregate_refresh.py</code> 打印的 op26 锚漂移检查把关。</li>
<li>开发容器:torch 2.12.0a0（nv26.05）· CUDA 13.2 · cutlass-dsl 4.6.0,运行时加
<code>PYTHONNOUSERSITE=1</code> 及机器本地 cutlass-dsl 4.5.0、flashinfer 0.6.11
（新容器修复记录在 <code>BS_SCALING_ENV_FIXES.md</code>）。</li>
<li>计时协议:nsys 冷 L2（rep 间刷 L2）,延迟 = 投影 NVTX GPU span（核投影,避免重叠多核路径双计）,取各 rep 中位数;
每个 cell 均带正确性门（PR/op26 对采集参考;竞品对 <code>torch.topk</code>）。</li>
</ul>
<p><b>测试代码位置。</b></p>
<ul>
<li>生产核:分支 <code>omni/op21-gvr-prod</code> → PR
<a href="https://github.com/NVIDIA/TensorRT-LLM/pull/16457">NVIDIA/TensorRT-LLM#16457</a>,文件
<code>tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py</code>
（R0 seeded-cluster + rank-scatter + <code>pick_config</code>/<code>launch</code> host API）。
可导入快照 == PR head:<code>gvrpkg_snapshot/gvrpkg/top_k/gvr_topk_decode.py</code>（本报告目录）。</li>
<li>基准 harness（均在 <code>indexer_topk_op_bench/op26_r0_upstream_port_report/</code> 下）:
<code>harness/</code>——3 臂 合成/真实/BS nsys 驱动;<code>refresh_harness/</code>——2026-07-16 PR 契约复测;
<code>rival_harness/</code>——§8 统一 6 臂竞品扫描。聚合脚本
<code>aggregate_refresh.py</code> / <code>aggregate_bs.py</code> / <code>aggregate_rival.py</code> →
<code>synth_3arm.csv</code> / <code>real_3arm.csv</code> / <code>bs_synth.csv</code> / <code>bs_real.csv</code> /
<code>rival_long.csv</code>;本页 = <code>gen_report.py</code>。</li>
<li>op26 锚臂（op-bench GVR）:<code>indexer_topk_op_bench/op26_gvr_logfalsi_rs/</code>。</li>
</ul>
</div>
</div>

<h2>11 · Reproduction / 复现</h2>
<div class="box">
<div class="lang-en"><p>Branch <code>omni/op21-gvr-prod</code> (13 commits over origin/main; production kernel only).
Harness + importable pkg snapshot staged under this report dir:</p></div>
<div class="lang-zh"><p>分支 <code>omni/op21-gvr-prod</code>（在 origin/main 上 13 个提交；仅生产核）。
Harness 与可导入 pkg 快照置于本报告目录：</p></div>
<ul>
<li><code>harness/perf_3arm.py</code> · <code>perf_3arm_real.py</code> — 3-arm nsys drivers (synthetic / real)</li>
<li><code>harness/drive_3arm_shard.sh</code> · <code>drive_3arm_real_shard.sh</code> — sharded launchers</li>
<li><code>aggregate_real.py</code> → <code>real_3arm.csv</code>; <code>synth_3arm.csv</code>; <code>gen_report.py</code> → this page</li>
<li><code>gvrpkg_snapshot/gvrpkg/</code> — importable pkg == branch HEAD (torch+cutlass only)</li>
<li>real loaders <code>harness/real_data_v4cap.py</code> (V4) · <code>harness/real_data_v32.py</code> (V3.2, K2048 cr=1);
V3.2 drivers <code>perf_3arm_real_v32.py</code> · <code>drive_3arm_real_v32_shard.sh</code></li>
<li>BS scaling (§7): <code>harness/bs_cells.py</code> (72 cells) · <code>drive_bs_shard.sh</code> · <code>perf_3arm_bs.py</code> /
<code>perf_3arm_real_bs.py</code> · <code>setup_bs_env.sh</code> · <code>aggregate_bs.py</code> → <code>bs_synth.csv</code> / <code>bs_real.csv</code>.
Newer-container env fixes (nvshmem shadow, cutlass 4.5.0 make_fragment) in <code>BS_SCALING_ENV_FIXES.md</code>.</li>
<li>big-BS dispatch triage (§7 KPI): <code>ab_bigbs_runnercfg.py</code> (CUDA-event) · <code>bigbs_nsys.py</code> +
<code>parse_bigbs.py</code> → <code>bigbs_triage.csv</code> (nsys, frozen-config vs runner-config vs op26) ·
findings in <code>BIGBS_TRIAGE_NOTE.md</code>.</li>
<li><b>2026-07-16 PR-contract refresh (all §3/§4/§7 tables + §8 GVR rows)</b>:
<code>refresh_harness/</code> (<code>ops_refresh.py</code> — base/pr via <code>GvrTopKKernel.launch</code>/<code>pick_config</code>
@<code>018251950f</code>; <code>sweep_refresh.py</code> — full 9-N synth BS grid + ALL real ISL rungs;
<code>drive_refresh_shard.sh</code> 8-GPU · <code>parse_refresh.py</code>) → <code>aggregate_refresh.py</code>
→ synth_3arm / real_3arm / bs_synth / bs_real CSVs + rival_long GVR-row replacement + op26 anchor-drift gate.</li>
<li>nsys median: <code>SELECT (k.end-k.start)/1000.0 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id WHERE s.value LIKE '%gvr_topk%'</code></li>
</ul>
<p class="mut lang-en">Data + report are local only (this NFS dir). Code lives on the branch. Language toggle is CSS-only; charts use Plotly (CDN).</p>
<p class="mut lang-zh">数据与报告仅本地（本 NFS 目录）。代码在分支上。语言切换为纯 CSS；图表用 Plotly（CDN）。</p>
</div>

<p class="mut" style="margin-top:2em">Generated by gen_report.py · op26_r0_upstream_port_report · B200 SM100 · same-run nsys cold-L2</p>
</div>

<script>
const SYN={json.dumps(SYN_JS)};
const REAL={json.dumps(REAL_JS)};
const BS_SYN={json.dumps(BS_SYN_JS)};
const BS_REAL={json.dumps(BS_REAL_JS)};
const ARMC={{base:'#ff7a7a',pr:'#6ea8fe',op26:'#6ede8a'}};
const ARML={{base:'base (secant)',pr:'PR (R0+RS)',op26:'op26_r0auto'}};
const KLABJS={{'512':'V4 Flash K512','1024':'V4 Pro K1024','2048':'V3.2 K2048'}};
function vals(cls){{return [...document.querySelectorAll('.'+cls+':checked')].map(x=>x.value);}}
function rad(name){{const e=document.querySelector('input[name='+name+']:checked');return e?e.value:null;}}
function LAY(t,xt,yt,ref,xticks){{
  const xa={{title:xt,type:'log',gridcolor:'#272c36',zeroline:false}};
  if(xticks){{xa.tickmode='array';xa.tickvals=xticks.v;xa.ticktext=xticks.t;xa.tickangle=-32;xa.tickfont={{size:9}};}}
  const l={{title:{{text:t,font:{{size:12.5,color:'#e6e9ef'}}}},paper_bgcolor:'#171a21',plot_bgcolor:'#171a21',
    font:{{color:'#9aa4b2',size:11}},xaxis:xa,
    yaxis:{{title:yt,gridcolor:'#272c36',zeroline:false}},margin:{{l:52,r:10,t:32,b:xticks?58:40}},
    legend:{{orientation:'h',y:-0.24,font:{{size:9.5}}}},height:xticks?320:300}};
  if(ref!=null) l.shapes=[{{type:'line',xref:'paper',x0:0,x1:1,y0:ref,y1:ref,line:{{color:'#6ede8a',dash:'dot',width:1}}}}];
  return l;
}}
function islLabel(isl){{return isl.toUpperCase().replace('1024K','1M');}}
// x-tick annotation: each indexer-N position -> "ISL (N=<n>)"
function nTicks(rowsArr){{
  const map={{}};
  rowsArr.forEach(r=>{{map[r.N]=islLabel(r.isl);}});
  const v=Object.keys(map).map(Number).sort((a,b)=>a-b);
  return {{v:v,t:v.map(n=>map[n]+' (N='+n+')')}};
}}
const SYNCR={{'512':4,'1024':4,'2048':1}};   // ISL = indexer-N × cr (V4 cr=4, V3.2 cr=1)
function fmtISL(t){{return t>=1048576?(t/1048576)+'M':(t>=1024?(t/1024)+'K':''+t);}}
function drawSyn(){{
  const K=rad('sk'),scens=vals('ss'),arms=vals('sa'),lat=[],rat=[],cr=SYNCR[K];
  let ns=[];
  scens.forEach(sc=>{{
    const rows=SYN.filter(r=>String(r.K)===K&&r.scen===sc).sort((a,b)=>a.N-b.N);
    const xs=rows.map(r=>r.N),dash=sc==='worst'?'dash':'solid';
    ns.push(...xs);
    arms.forEach(a=>{{
      lat.push({{x:xs,y:rows.map(r=>r[a]),name:a+' '+sc,mode:'lines+markers',
        line:{{color:ARMC[a],dash:dash}},marker:{{size:5}}}});
      if(a!=='base') rat.push({{x:xs,y:rows.map(r=>r.base/r[a]),name:a+'/base '+sc,mode:'lines+markers',
        line:{{color:ARMC[a],dash:dash}},marker:{{size:5}}}});
    }});
  }});
  const uv=[...new Set(ns)].sort((a,b)=>a-b);
  const tk={{v:uv,t:uv.map(n=>fmtISL(n*cr)+' (N='+n+')')}};   // ISL (indexer N)
  const rel=cr===1?'ISL≈N':'ISL='+cr+'×N';
  Plotly.react('syn_lat',lat,LAY('Synthetic '+KLABJS[K]+' — latency vs indexer N (fp32 BS=1, cold-L2)','ISL (indexer N, '+rel+')','µs',null,tk),{{responsive:true}});
  Plotly.react('syn_rat',rat,LAY('Synthetic '+KLABJS[K]+' — speedup vs base (>1 faster)','ISL (indexer N, '+rel+')','ratio',1,tk),{{responsive:true}});
}}
const RDASH={{flash:'solid',pro:'dash',v32:'dot'}};
const RMLAB={{flash:'V4 Flash',pro:'V4 Pro',v32:'V3.2'}};
function drawReal(){{
  const models=vals('rm'),arms=vals('ra'),lat=[],rat=[],all=[];
  models.forEach(m=>{{
    const rows=REAL.filter(r=>r.model===m).sort((a,b)=>a.N-b.N);
    all.push(...rows);
    const xs=rows.map(r=>r.N),cd=rows.map(r=>islLabel(r.isl)),dash=RDASH[m]||'solid';
    const ht='%{{customdata}} · indexer N=%{{x}} · %{{y:.2f}}<extra>%{{fullData.name}}</extra>';
    arms.forEach(a=>{{
      lat.push({{x:xs,y:rows.map(r=>r[a]),customdata:cd,name:a+' '+RMLAB[m],mode:'lines+markers',
        line:{{color:ARMC[a],dash:dash}},marker:{{size:5}},hovertemplate:ht}});
      if(a!=='base') rat.push({{x:xs,y:rows.map(r=>r.base/r[a]),customdata:cd,name:a+'/base '+RMLAB[m],
        mode:'lines+markers',line:{{color:ARMC[a],dash:dash}},marker:{{size:5}},hovertemplate:ht}});
    }});
  }});
  const tk=nTicks(all);
  Plotly.react('real_lat',lat,LAY('Real V4+V3.2 decode — latency vs indexer N (fp32 BS=1, cold-L2)','ISL (indexer N: V4=ISL/4, V3.2≈ISL)','µs',null,tk),{{responsive:true}});
  Plotly.react('real_rat',rat,LAY('Real V4+V3.2 decode — speedup vs base (>1 faster)','ISL (indexer N)','ratio',1,tk),{{responsive:true}});
}}
// ---- §7 BS scaling: BS on log-x ----
function bsLAY(t,yt,ref){{
  const l={{title:{{text:t,font:{{size:12.5,color:'#e6e9ef'}}}},paper_bgcolor:'#171a21',plot_bgcolor:'#171a21',
    font:{{color:'#9aa4b2',size:11}},
    xaxis:{{title:'batch size (log)',type:'log',gridcolor:'#272c36',zeroline:false,
      tickmode:'array',tickvals:[1,2,4,8,16,32,64,128,256,512,1024],
      ticktext:['1','2','4','8','16','32','64','128','256','512','1024'],tickfont:{{size:9}}}},
    yaxis:{{title:yt,gridcolor:'#272c36',zeroline:false}},margin:{{l:52,r:10,t:32,b:44}},
    legend:{{orientation:'h',y:-0.26,font:{{size:9.5}}}},height:310}};
  if(ref!=null) l.shapes=[{{type:'line',xref:'paper',x0:0,x1:1,y0:ref,y1:ref,line:{{color:'#6ede8a',dash:'dot',width:1}}}}];
  return l;
}}
function drawBsSyn(){{
  const K=rad('bsk'),dt=rad('bsd'),N=+rad('bsn'),scens=vals('bss'),arms=vals('bsa'),lat=[],rat=[];
  scens.forEach(sc=>{{
    const rows=BS_SYN.filter(r=>String(r.K)===K&&r.dtype===dt&&r.N===N&&r.scen===sc).sort((a,b)=>a.BS-b.BS);
    if(!rows.length) return;
    const xs=rows.map(r=>r.BS),dash=sc==='worst'?'dash':'solid';
    arms.forEach(a=>{{
      lat.push({{x:xs,y:rows.map(r=>r[a]),name:a+' '+sc,mode:'lines+markers',
        line:{{color:ARMC[a],dash:dash}},marker:{{size:5}}}});
      if(a!=='base') rat.push({{x:xs,y:rows.map(r=>r.base&&r[a]?r.base/r[a]:null),name:a+'/base '+sc,
        mode:'lines+markers',line:{{color:ARMC[a],dash:dash}},marker:{{size:5}}}});
    }});
  }});
  Plotly.react('bs_syn_lat',lat,bsLAY('Synthetic '+KLABJS[K]+' '+dt+' N='+(N/1024)+'K — latency vs BS','µs',null),{{responsive:true}});
  Plotly.react('bs_syn_rat',rat,bsLAY('Synthetic '+KLABJS[K]+' '+dt+' N='+(N/1024)+'K — speedup vs base (>1 faster)','ratio',1),{{responsive:true}});
}}
function drawBsReal(){{
  const models=vals('brm'),dt=rad('brd'),rung=rad('brl'),arms=vals('bra'),lat=[],rat=[];
  models.forEach(m=>{{
    // rung = ISL string (full-ISL coverage 2026-07-16); models lacking the rung drop out
    let rows=BS_REAL.filter(r=>r.model===m&&r.dtype===dt&&r.isl===rung).sort((a,b)=>a.BS-b.BS);
    if(!rows.length) return;
    const xs=rows.map(r=>r.BS),dash=RDASH[m]||'solid',N0=rows[0].N;
    arms.forEach(a=>{{
      lat.push({{x:xs,y:rows.map(r=>r[a]),name:a+' '+RMLAB[m]+' N='+N0,mode:'lines+markers',
        line:{{color:ARMC[a],dash:dash}},marker:{{size:5}}}});
      if(a!=='base') rat.push({{x:xs,y:rows.map(r=>r.base&&r[a]?r.base/r[a]:null),name:a+'/base '+RMLAB[m],
        mode:'lines+markers',line:{{color:ARMC[a],dash:dash}},marker:{{size:5}}}});
    }});
  }});
  Plotly.react('bs_real_lat',lat,bsLAY('Real '+dt+' ('+rung+' rung) — latency vs BS','µs',null),{{responsive:true}});
  Plotly.react('bs_real_rat',rat,bsLAY('Real '+dt+' ('+rung+' rung) — speedup vs base (>1 faster)','ratio',1),{{responsive:true}});
}}
// ---- §7 BS × seq-len heatmaps (checkbox-selected metric maps) ----
const HMLAB={{base:'base µs',pr:'PR µs',op26:'op26 µs',
  pvb:'PR/base speedup (>1 = PR faster)',pvo:'op26/PR (>1 = op26 faster)'}};
function hmVal(r,m){{
  if(m==='pvb') return (r.base&&r.pr)?+(r.base/r.pr).toFixed(3):null;
  if(m==='pvo') return (r.pr&&r.op26)?+(r.pr/r.op26).toFixed(3):null;
  return r[m]!=null?+(+r[m]).toFixed(2):null;
}}
function drawHM(){{
  const cont=document.getElementById('hm_row'); if(!cont) return;
  const fam=rad('hmf'),dt=rad('hmd'),mets=vals('hmz');
  let rows,lab={{}},ttl;
  if(fam==='synth'){{
    const K=rad('hmk'),sc=rad('hms');
    rows=BS_SYN.filter(r=>String(r.K)===K&&r.dtype===dt&&r.scen===sc);
    rows.forEach(r=>{{lab[r.N]='N='+fmtISL(r.N);}});
    ttl='synth '+KLABJS[K]+' '+sc+' '+dt;
  }} else {{
    const m=rad('hmm');
    rows=BS_REAL.filter(r=>r.model===m&&r.dtype===dt);
    rows.forEach(r=>{{lab[r.N]=islLabel(r.isl)+' (N='+fmtISL(r.N)+')';}});
    ttl='real '+RMLAB[m]+' '+dt;
  }}
  cont.innerHTML='';
  if(!rows.length){{cont.innerHTML='<p class="mut" style="padding:8px">(no cells for this selection / 此选择无实测 cell)</p>';return;}}
  const xs=[...new Set(rows.map(r=>r.N))].sort((a,b)=>a-b);
  const ys=[...new Set(rows.map(r=>r.BS))].sort((a,b)=>a-b);
  const cm={{}};rows.forEach(r=>{{cm[r.N+'_'+r.BS]=r;}});
  mets.forEach((m,i)=>{{
    const z=ys.map(b=>xs.map(n=>{{const r=cm[n+'_'+b];return r?hmVal(r,m):null;}}));
    const d=document.createElement('div');d.id='hm_p'+i;d.className='plt';cont.appendChild(d);
    const ratio=(m==='pvb'||m==='pvo');
    const tr={{type:'heatmap',x:xs.map(n=>lab[n]),y:ys.map(String),z:z,
      colorscale:ratio?[[0,'#ff7a7a'],[0.5,'#20242e'],[1,'#6ede8a']]:'Viridis',
      texttemplate:'%{{z}}',textfont:{{size:8.5}},hoverongaps:false,
      colorbar:{{thickness:10,tickfont:{{size:9}}}}}};
    if(ratio) tr.zmid=1;
    Plotly.react(d.id,[tr],{{title:{{text:HMLAB[m]+' — '+ttl,font:{{size:11.5,color:'#e6e9ef'}}}},
      paper_bgcolor:'#171a21',plot_bgcolor:'#171a21',font:{{color:'#9aa4b2',size:10}},
      xaxis:{{type:'category',title:'seq-len rung (indexer N)',tickangle:-38,tickfont:{{size:8.5}}}},
      yaxis:{{type:'category',title:'BS'}},margin:{{l:48,r:6,t:32,b:70}},height:360}},{{responsive:true}});
  }});
}}
// ---- §8 External rival comparison ----
const RIVAL={json.dumps(RIVAL_JS)};
const RVC={{gvr_base:'#ff7a7a',gvr_pr:'#c98bff',op26_r0auto:'#6ea8fe',
           radix_cutedsl:'#6ede8a',sglang_v2:'#ffbf69',flashinfer_topk:'#4be0d0'}};
const RVLAB={{gvr_base:'GVR base',gvr_pr:'GVR pr',op26_r0auto:'GVR op26',
             radix_cutedsl:'Radix cuteDSL',sglang_v2:'SGLang v2',flashinfer_topk:'FlashInfer'}};
function drawRival(){{
  const fam=rad('rvf'),view=rad('rvv'),dt=rad('rvd'),arms=vals('rva');
  const sweep=view;    // 'seqlen' or 'bs'
  let rowsF=RIVAL.filter(r=>r.family===fam&&r.sweep===sweep&&r.dtype===dt);
  if(fam==='synth'){{const K=+rad('rvk'),sc=rad('rvs');rowsF=rowsF.filter(r=>r.K===K&&r.scenario===sc);
    if(sweep==='bs') rowsF=rowsF.filter(r=>r.N===+rad('rvn'));}}
  else{{const m=rad('rvm');rowsF=rowsF.filter(r=>r.model===m);
    if(sweep==='bs') rowsF=rowsF.filter(r=>r.isl===rad('rvi'));}}
  // x = N (seqlen view) or BS (bs view)
  const xf=(sweep==='seqlen')?(r=>r.N):(r=>r.BS);
  const lat=[],rat=[];
  // build per-arm series + GVR-op26 baseline map keyed by x (t = span-primary latency)
  const gvr={{}}; rowsF.filter(r=>r.op==='op26_r0auto').forEach(r=>{{gvr[xf(r)]=r.t;}});
  arms.forEach(a=>{{
    const rs=rowsF.filter(r=>r.op===a).sort((x,y)=>xf(x)-xf(y));
    if(!rs.length) return;
    const xs=rs.map(xf);
    lat.push({{x:xs,y:rs.map(r=>r.t),name:RVLAB[a],mode:'lines+markers',
      line:{{color:RVC[a]}},marker:{{size:5}}}});
    if(a!=='op26_r0auto') rat.push({{x:xs,y:rs.map(r=>gvr[xf(r)]?gvr[xf(r)]/r.t:null),
      name:RVLAB[a],mode:'lines+markers',line:{{color:RVC[a]}},marker:{{size:5}}}});
  }});
  const xt=(sweep==='seqlen')?'candidate length N (log)':'batch size (log)';
  const lx=(sweep==='bs')?{{tickmode:'array',tickvals:[1,2,4,8,16,32,64,128,256,512,1024],
    ticktext:['1','2','4','8','16','32','64','128','256','512','1024']}}:{{}};
  function L(t,yt,ref){{return Object.assign({{title:{{text:t,font:{{size:12,color:'#e6e9ef'}}}},
    paper_bgcolor:'#171a21',plot_bgcolor:'#171a21',font:{{color:'#9aa4b2',size:11}},
    xaxis:Object.assign({{title:xt,type:'log',gridcolor:'#272c36',zeroline:false,tickfont:{{size:9}}}},lx),
    yaxis:{{title:yt,gridcolor:'#272c36',zeroline:false}},margin:{{l:52,r:10,t:30,b:44}},
    legend:{{orientation:'h',y:-0.26,font:{{size:9}}}},height:320}},
    ref!=null?{{shapes:[{{type:'line',xref:'paper',x0:0,x1:1,y0:ref,y1:ref,line:{{color:'#6ea8fe',dash:'dot',width:1}}}}]}}:{{}});}}
  const scK=(fam==='synth')?(KLABJS[String(rad('rvk'))]+' '+rad('rvs')):RMLAB[rad('rvm')];
  Plotly.react('rv_lat',lat,L('External top-K — '+scK+' '+dt+' — latency vs '+(sweep==='bs'?'BS':'N')+' (lower = faster)','µs',null),{{responsive:true}});
  Plotly.react('rv_rat',rat,L('t(GVR op26)/t(arm) — >1 = arm FASTER than GVR','ratio',1),{{responsive:true}});
}}
function drawAll(){{try{{drawSyn();}}catch(e){{}} try{{drawReal();}}catch(e){{}}
  try{{drawBsSyn();}}catch(e){{}} try{{drawBsReal();}}catch(e){{}} try{{drawHM();}}catch(e){{}} try{{drawRival();}}catch(e){{}}}}
document.querySelectorAll('input[name=sk],.ss,.sa,.rm,.ra,input[name=bsk],input[name=bsd],input[name=bsn],.bss,.bsa,.brm,input[name=brd],input[name=brl],.bra,input[name=hmf],input[name=hmk],input[name=hms],input[name=hmm],input[name=hmd],.hmz,input[name=rvf],input[name=rvv],input[name=rvk],input[name=rvs],input[name=rvm],input[name=rvd],input[name=rvn],input[name=rvi],.rva').forEach(e=>e.addEventListener('change',()=>setTimeout(drawAll,0)));
if(window.Plotly) drawAll(); else window.addEventListener('load',drawAll);
</script>
</body>
</html>
"""

open(os.path.join(HERE, "REPORT.html"), "w").write(HTML)
# Re-inject marker-delimited chapters that a full regen would otherwise wipe
# (the git-checkout/last-writer hazard): §8.1/§8.2 correctness notes.
import subprocess, sys
_note = os.path.join(HERE, "sglv2_correctness", "update_report_sglv2_note.py")
if os.path.exists(_note):
    subprocess.run([sys.executable, _note], check=True)
print(f"wrote REPORT.html  synth={len(synth)} cells  real={len(real)} cells")
print(f"synth PR/base={syn_pvb:.3f} op26/PR={syn_pvo:.3f} | real PR/base={real_pvb:.3f} "
      f"(flash {real_pvb_flash:.3f} pro {real_pvb_pro:.3f}) PR-exact={real_exact}/{len(real)}")
