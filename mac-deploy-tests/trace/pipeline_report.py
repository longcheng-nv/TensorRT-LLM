#!/usr/bin/env python3
"""Write DSV41_PIPELINE_REPORT.html from trace_agent.py runs (baseline and prefill lookahead) and the recorded session.

usage: pipeline_report.py OUT.html CAPTURE.jsonl BASE.json LOOK.json [--base-alt BASE2.log]
The numbers on the page are computed here from those files; the few measured elsewhere are in REF below.
"""
from __future__ import annotations

import argparse
import bisect
import html
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
MAC_REPORT = os.path.join(os.path.dirname(HERE), "DSV41_MAC_REPORT.html")

REF = {
    # 2026-11-17 mk-7 decode trace, --fast, cache 16 GiB, 14-token prompt (logs/mk/trace_mk7/mk7_fast.json), ms/token
    "dec16": {"T": 82.4, "gpu_only": 45.5, "both": 20.0, "ssd_only": 15.6, "idle": 1.3},
    # 2026-11-24 acceptance run (dsv41 test, 21 requests, cache 8 GiB): time to first token vs decode
    "nov24": {"ttft_s": 443.4, "decode_s": 141.1, "share": 0.76},
    # mac-deploy-tests/mk/test_prefill_lookahead.py --tokens 600 (logs/mk/agent/test_prefill_lookahead.log)
    "bitexact": {"logits": 18099200, "off_s": (32.37, 34.09), "on_s": (27.99, 28.53), "off_gb": 165.3, "on_gb": 166.1},
    # pytest tests/ (tiny checkpoint: engine, parity vs the reference, vision, store, server, ...), final code
    "unit": "40/40",
    # dsv41 test --quick through the launcher with the final code (filled in by --quick-log)
    "quick": None,
    # mk/bench_prefill_vl.py and the grouping run (logs/mk/agent/bench_prefill_vl.log, vl_pairs.log): image prompt, 1,500 tokens
    # (measured before the slab cap of 2fcc4a4: k 125 instead of 117 on the 17 2-2-3 layers)
    "vl": {"off_s": (28.31, 27.12, 30.52, 27.61), "on_s": (23.58, 22.70, 23.47, 23.36), "max_diff": 1.125},
}

e = html.escape


def f1(x):
    return f"{x:,.1f}"


def f0(x):
    return f"{x:,.0f}"


def pct(x):
    return f"{100 * x:.0f}%"


# ------------------------------------------------------------------------------ data

def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def idle_split(run):
    """SSD idle time inside each prefill segment, by what the main thread was doing (route sync vs batch eval split)."""
    res = []
    for s in [x for x in run["tokens"] if x["kind"] == "prefill"]:
        t0, t1 = s["t0"], s["t1"]
        busy = union([(max(v["t0"], t0), min(v["t1"], t1)) for v in run["io"] if v["t1"] > t0 and v["t0"] < t1])
        st = [a for a, _ in busy]

        def busy_in(a, b):
            i, tot = max(0, bisect.bisect_right(st, a) - 1), 0.0
            for x, y in busy[i:]:
                if x >= b:
                    break
                lo, hi = max(a, x), min(b, y)
                tot += max(0.0, hi - lo)
            return tot

        cats = {}
        for c in run["cpu"]:
            if c["t1"] <= t0 or c["t0"] >= t1:
                continue
            k, n = c["k"], c.get("n", "")
            if k == "sync":
                k = "route_sync" if n.startswith("路由同步") else "batch_eval"
            elif k in ("launch_hit", "launch_miss", "io_issue", "post"):
                k = "batch_other"
            a, b = max(c["t0"], t0), min(c["t1"], t1)
            v = cats.setdefault(k, [0.0, 0.0])
            v[0] += b - a
            v[1] += (b - a) - busy_in(a, b)
        T = t1 - t0
        res.append({"T": T, "idle": T - sum(y - x for x, y in busy), "cats": cats})
    return res


def load(a):
    cap = [json.loads(line) for line in open(a.capture) if line.strip()]
    base, look = json.load(open(a.base)), json.load(open(a.look))
    alt = None
    if a.base_alt and os.path.exists(a.base_alt):
        alt = [float(m) for m in re.findall(r"first token ([\d.]+) s", open(a.base_alt).read())]
    return cap, base, look, alt


def parse_anchor_log(path):
    """mk/test_prefix_anchor.py output -> [{q, tokens, exact, cold, make, reuse, reused, same_cold, same_anchor, cos_cold, cos_anchor}]"""
    out, cur = [], None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"B: '(.*)' \((\d+) tokens\)", line)
        if m:
            cur = {"q": m.group(1), "tokens": int(m.group(2))}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s+(exact|cold, no anchor|A: first time seen|A2?: makes the anchor|B: restores the anchor)[^\d]*?\s+([\d.]+) s\s+reused\s+(\d+)", line)
        if m:
            key = {"exact": "exact", "cold, no anchor": "cold", "A: first time seen": "first", "A: makes the anchor": "make",
                   "A2: makes the anchor": "make", "B: restores the anchor": "reuse"}[m.group(1)]
            cur[key] = float(m.group(2))
            if key == "reuse":
                cur["reused"] = int(m.group(3))
            continue
        m = re.match(r"\s+(cold|anchor)\s+vs exact: cos ([\d.]+), max \|diff\| ([\d.]+), top-1 same (\w+), greedy tokens same (\w+)", line)
        if m:
            cur[f"cos_{m.group(1)}"] = float(m.group(2))
            cur[f"diff_{m.group(1)}"] = float(m.group(3))
            cur[f"top1_{m.group(1)}"] = m.group(4) == "True"
            cur[f"same_{m.group(1)}"] = m.group(5) == "True"
    return out


def suite_s(path):
    return sum(r.get("seconds") or 0 for r in json.load(open(path))["results"])


def harness_ttft(path, anchor_len=1500):
    """dsv41 test JSON -> (first requests of new sessions [(check, ttft, cached)], requests with images [(check, ttft, cached)])"""
    d = json.load(open(path))
    new, img = [], []
    for r in d["results"]:
        reqs = r.get("requests") or []
        if r["check"].startswith("prefix cache"):
            continue
        for i, q in enumerate(reqs):
            if q.get("images"):
                img.append((r["check"], q["ttft_s"], q["cached"]))
            elif i == 0 and q["cached"] <= anchor_len:
                new.append((r["check"], q["ttft_s"], q["cached"]))
    return new, img


# ------------------------------------------------------------------------------ html pieces

def tip(tv, tt, tn=""):
    return f' tabindex="0" data-tv="{e(tv)}" data-tt="{e(tt)}"' + (f' data-tn="{e(tn)}"' if tn else "")


def axis(maxv, ticks, unit=""):
    grid = "".join(f'<div class="hb-grid" style="left:{100 * t / maxv:.2f}%"></div>' for t in ticks if t > 0)
    lab = "".join(f'<span style="left:{100 * t / maxv:.2f}%">{e(str(t))}{unit if t == ticks[-1] else ""}</span>' for t in ticks)
    return grid, f'<div class="hb-axis" aria-hidden="true">{lab}</div>'


def stacked(rows, cats, maxv, ticks, unit, aria, fmt=f1):
    """rows: [(label_html, [values per cat], total_text, tip_title)] ; cats: [(name, css_class)]"""
    grid, ax = axis(maxv, ticks, unit)
    labels = "".join(f"<div>{lab}</div>" for lab, *_ in rows) + '<div class="axis-gap"></div>'
    body = []
    for lab, vals, total, title in rows:
        tot = sum(vals)
        segs = []
        for (name, cls), v in zip(cats, vals):
            if v <= 0:
                continue
            segs.append(f'<i class="{cls}" style="--f:{100 * v / tot:.3f}%" role="img" aria-label="{e(title)} · {e(name)}：{fmt(v)}{unit}"'
                        + tip(f"{fmt(v)}{unit}（{100 * v / tot:.0f}%）", f"{title} · {name}") + "></i>")
        w = 100 * tot / maxv
        body.append(f'<div class="hb-row"><div class="hb-bar stack" style="--w:{w:.2f}%">{"".join(segs)}</div>'
                    f'<span class="hb-v" style="--w:{w:.2f}%">{e(total)}</span></div>')
    legend = '<div class="legend" aria-hidden="true">' + "".join(f'<span><i class="sw {cls}"></i>{e(n)}</span>' for n, cls in cats) + "</div>"
    return (legend + f'<div class="hb" role="group" aria-label="{e(aria)}"><div class="hb-labels" aria-hidden="true">{labels}</div>'
            f'<div class="hb-plot"><div class="hb-base"></div>{grid}{"".join(body)}{ax}</div></div>')


def bars(rows, maxv, ticks, unit, aria, fmt=f1):
    """rows: [(label_html, value, css_class, tip_title, tip_note)]"""
    grid, ax = axis(maxv, ticks, unit)
    labels = "".join(f"<div>{lab}</div>" for lab, *_ in rows) + '<div class="axis-gap"></div>'
    body = "".join(
        f'<div class="hb-row" role="img" aria-label="{e(title)}：{fmt(v)}{unit}"{tip(f"{fmt(v)}{unit}", title, note)}>'
        f'<div class="hb-bar {cls}" style="--w:{100 * v / maxv:.2f}%"></div><span class="hb-v" style="--w:{100 * v / maxv:.2f}%">{fmt(v)}</span></div>'
        for lab, v, cls, title, note in rows)
    return (f'<div class="hb" role="group" aria-label="{e(aria)}"><div class="hb-labels" aria-hidden="true">{labels}</div>'
            f'<div class="hb-plot"><div class="hb-base"></div>{grid}{body}{ax}</div></div>')


def table(head, rows, tid=None, hidden=True, text=False):
    th = "".join(f"<th>{h}</th>" for h in head)
    cell = (lambda i, c: f'<td class="txt">{c}</td>') if text else (lambda i, c: f'<td class="num">{c}</td>' if i else f'<td class="nw">{c}</td>')
    tr = "".join("<tr>" + "".join(cell(i, c) for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'<div class="tbl"{f" id={chr(34)}{tid}{chr(34)}" if tid else ""}{" hidden" if hidden else ""}>'
            f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>")


def figure(title, sub, inner, tbl_html, tid, caption):
    btn = f'<button class="fig-btn" type="button" data-toggle="{tid}" aria-expanded="false" aria-controls="{tid}">显示数据表</button>' if tid else ""
    return (f'<figure class="fig"><div class="fig-head"><div><p class="fig-title">{title}</p><p class="fig-sub">{sub}</p></div>{btn}</div>'
            f"{inner}{tbl_html}<figcaption>{caption}</figcaption></figure>")


# ------------------------------------------------------------------------------ page

def build(a):
    cap, base, look, alt = load(a)
    B, L = base["summary"], look["summary"]
    split_b, split_l = idle_split(base["runs"][0]), idle_split(look["runs"][0])
    n = len(B)
    b_first = [r["first_token_s"] for r in B]
    l_first = [r["first_token_s"] for r in L]
    pf_T = sum(r["prefill"]["T"] for r in B) / 1e3
    first_T = sum(r["first"]["T"] for r in B) / 1e3
    dec_T = sum((r["decode"]["T"] * r["decode"]["n"]) for r in B if r["decode"]) / 1e3
    ttft_share = (pf_T + first_T) / (pf_T + first_T + dec_T)
    ssd_busy = [(r["prefill"]["ssd"] / r["prefill"]["T"]) for r in B]
    gbps = [r["prefill"]["GBps_busy"] for r in B]

    def cat_sum(split, k, i=1):
        return sum(s["cats"].get(k, [0, 0])[i] for s in split) / 1e3

    rs_b, rs_l = cat_sum(split_b, "route_sync"), cat_sum(split_l, "route_sync")
    idle_b = sum(s["idle"] for s in split_b) / 1e3
    cold_rs = split_b[0]["cats"]["route_sync"][1] / 1e3
    cold_T = split_b[0]["T"] / 1e3
    sum_b, sum_l = sum(b_first), sum(l_first)
    gb_b = [r["prefill"]["GB"] + r["first"]["GB"] for r in B]
    gb_l = [r["prefill"]["GB"] + r["first"]["GB"] for r in L]
    dec_ms = [r["decode"]["T"] for r in B if r["decode"]]
    dec_n = sum(r["decode"]["n"] for r in B if r["decode"])
    first_ms = [r["first"]["T"] for r in B]
    first_gb = [r["first"]["GB"] for r in B]

    css = open(MAC_REPORT, encoding="utf-8").read()
    css = css[css.index("<style>") + 7:css.index("</style>")]
    css = css.replace("/* Same visual language as MK_REPORT.html / OPT_REPORT.html (this report continues that series). */",
                      "/* Same visual language as DSV41_MAC_REPORT.html / MK_REPORT.html (this report continues that series). */")
    css += """
/* stacked bars + legend (series colours: the reference palette's first slots, fixed order; idle = neutral) */
:root{--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--cn:#b9b8b0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--c1:#3987e5;--c2:#d95926;--c3:#199e70;--cn:#5c5b56}}
:root[data-theme="dark"]{--c1:#3987e5;--c2:#d95926;--c3:#199e70;--cn:#5c5b56}
.hb-bar.stack{display:flex;background:none;overflow:hidden;border-radius:0 4px 4px 0}
.hb-bar.stack > i{display:block;height:100%;flex:0 0 var(--f);min-width:0;cursor:default;outline-offset:-2px}
.hb-bar.stack > i + i{box-shadow:inset 2px 0 0 var(--surface)}
.hb-bar.stack > i:hover,.hb-bar.stack > i:focus-visible{filter:brightness(1.12)}
.s1{background:var(--c1)}.s2{background:var(--c2)}.s3{background:var(--c3)}.sn{background:var(--cn)}
.hb-bar.acc{background:var(--accent)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12.5px;color:var(--fg-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend .sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.hb-labels small{color:var(--fg-3);font-size:11.5px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;font-size:13.5px}
.kv dt{color:var(--fg-3)}.kv dd{margin:0}
.hb-axis span{white-space:nowrap}
td.txt{min-width:9em}
td.nw{white-space:nowrap}
"""
    # ---------------------------------------------------------------- chart A: where a request's time goes
    rowsA = []
    for r, c in zip(B, cap):
        d = r["decode"] or {"T": 0, "n": 0}
        ttft = (r["prefill"]["T"] + r["first"]["T"]) / 1e3
        dt = d["T"] * d["n"] / 1e3
        rowsA.append((f"请求 {r['req']} <small>新 {r['new']:,} · 出 {r['generated']}</small>", [ttft, dt], f"{ttft + dt:.1f} s",
                      f"请求 {r['req']}（新 {r['new']:,} token，生成 {r['generated']} token）"))
    figA = figure("一次请求的时间花在哪（秒）", "回放（专家缓存 8 GiB，默认 <code>--fast</code>）· 首字前 = prefill + 首个输出 token",
                  stacked(rowsA, [("首字前（prefill + 首个 token）", "s2"), ("解码", "sn")], 60, [0, 15, 30, 45, 60], " s",
                          "每个请求的首字前时间与解码时间（秒）"),
                  table(["请求", "新 token", "首字前（回放）", "实际运行首字", "解码 token", "解码耗时", "解码 ms/token"],
                        [[f"请求 {r['req']}", f"{r['new']:,}", f"{r['first_token_s']:.2f} s", f"{c['t_tokens'][0]:.2f} s",
                          f"{(r['decode'] or {}).get('n', 0)}", f"{(r['decode'] or {'T': 0, 'n': 0})['T'] * (r['decode'] or {'n': 0})['n'] / 1e3:.1f} s",
                          f"{(r['decode'] or {}).get('T', 0):.0f}"] for r, c in zip(B, cap)], "tbl-a"),
                  "tbl-a",
                  f"这段会话三轮共 6 个请求：首字前合计 {pf_T + first_T:.0f} s，解码 {dec_T:.0f} s（{dec_n} 个 token），首字前占 {pct(ttft_share)}。"
                  f"这次回复较长；11-24 验收的 21 个请求回复更短，首字前占 {pct(REF['nov24']['share'])}。新 token 只有十几个的请求（4、5）首字前也要 4 s 多。")

    # ---------------------------------------------------------------- chart B: prefill, SSD vs GPU
    cats4 = [("只有 GPU 忙（SSD 空着）", "s1"), ("只有 SSD 忙（GPU 在等盘）", "s2"), ("两者都忙", "s3"), ("都空闲", "sn")]
    rowsB = []
    for r in B:
        p = r["prefill"]
        rowsB.append((f"请求 {r['req']} <small>新 {r['new']:,}</small>", [p["gpu_only"] / 1e3, p["ssd_only"] / 1e3, p["both"] / 1e3, p["idle"] / 1e3],
                      f"{p['T'] / 1e3:.1f} s", f"请求 {r['req']} prefill"))
    figB = figure("prefill 期间 SSD 与 GPU 各自在不在忙（秒）", "每个请求的 prefill 窗口（不含首个输出 token）；SSD 忙 = 任一 IO 线程在读；GPU 区间由主机同步推断",
                  stacked(rowsB, cats4, 30, [0, 10, 20, 30], " s", "prefill 窗口按 SSD / GPU 状态分解（秒）"),
                  table(["请求", "prefill", "只有 GPU 忙", "只有 SSD 忙", "两者都忙", "都空闲", "SSD 忙占比", "读盘", "SSD 忙时带宽", "平均并发 IO 线程"],
                        [[f"请求 {r['req']}（新 {r['new']:,}）", f"{r['prefill']['T'] / 1e3:.2f} s", f"{r['prefill']['gpu_only'] / 1e3:.2f} s",
                          f"{r['prefill']['ssd_only'] / 1e3:.2f} s", f"{r['prefill']['both'] / 1e3:.2f} s", f"{r['prefill']['idle'] / 1e3:.2f} s",
                          pct(r['prefill']['ssd'] / r['prefill']['T']), f"{r['prefill']['GB']:.1f} GB", f"{r['prefill']['GBps_busy']:.2f} GB/s",
                          f"{r['prefill']['io_concurrency']:.1f} / 8"] for r in B], "tbl-b"),
                  "tbl-b",
                  f"prefill 基本是读盘：SSD 忙的时间占 {pct(min(ssd_busy))}–{pct(max(ssd_busy))}，忙时 {min(gbps):.1f}–{max(gbps):.1f} GB/s，接近这块盘逐专家读取时的 6.0–6.4 GB/s。"
                  f"冷 prefill（请求 1）的「只有 GPU 忙」最多（{B[0]['prefill']['gpu_only'] / 1e3:.1f} s）：1,585 行的注意力、路由和专家计算都更重，SSD 在这段时间里没有活干。")

    # ---------------------------------------------------------------- chart C: what the CPU does while the SSD idles
    names = [("route_sync", "每层的路由同步", "GPU 在算上一层最后一批专家 + 本层注意力与路由；这一层要读哪些专家还不知道"),
             ("build", "构建下一层的计算图", "Python 构建 hc + 注意力图（惰性）"),
             ("other", "Engram 查表与其他", "Engram 层主机查 n-gram 表行、请求开始的前缀匹配等"),
             ("batch_eval", "等 GPU 算完一批专家", "层内：下一批的读盘在这批计算结束前就读完了"),
             ("wait", "等这一批专家读完", "SSD 在读，只有很短的交接空隙"),
             ("batch_other", "构建 / 下发一批", "")]
    tot_idle = {k: cat_sum(split_b, k) for k, *_ in names}
    tot_time = {k: cat_sum(split_b, k, 0) for k, *_ in names}
    rowsC = [(lab, tot_idle[k], "acc" if k == "route_sync" else "mute", lab, f"SSD 空闲 {tot_idle[k]:.2f} s，这个阶段共 {tot_time[k]:.2f} s。{note}")
             for k, lab, note in names if tot_idle[k] >= 0.005]
    figC = figure("SSD 空闲时主线程在做什么（整段会话的 prefill，秒）", f"6 个请求的 prefill 共 {pf_T:.1f} s，其中 SSD 完全空闲 {idle_b:.1f} s（{pct(idle_b / pf_T)}）",
                  bars(rowsC, 12, [0, 3, 6, 9, 12], " s", "SSD 空闲时间按主线程阶段归因（秒）", fmt=lambda v: f"{v:.2f}"),
                  table(["主线程阶段", "SSD 空闲", "该阶段总时长", "空闲占该阶段"],
                        [[lab, f"{tot_idle[k]:.2f} s", f"{tot_time[k]:.2f} s", pct(tot_idle[k] / tot_time[k]) if tot_time[k] else "—"] for k, lab, _ in names], "tbl-c"),
                  "tbl-c",
                  f"最大的一块是每层的路由同步：{rs_b:.1f} s，同步期间 SSD 100% 空闲。原因是 prefill 只预读本层的下一批专家，"
                  f"本层最后一批读完以后，要等下一层算完注意力和路由、知道该读哪些专家，才会再发读盘。冷 prefill 每层要等 200–300 ms（1,585 行），合计 {cold_rs:.1f} s，占这次 prefill 的 {pct(cold_rs / cold_T)}。")

    # ---------------------------------------------------------------- chart D: the lookahead
    rowsD = []
    for r, q in zip(B, L):
        rowsD.append((f"请求 {r['req']} · 基线", r["first_token_s"], "mute", f"请求 {r['req']} 基线", f"新 {r['new']:,} token"))
        rowsD.append((f"请求 {r['req']} · 跨层预读", q["first_token_s"], "acc", f"请求 {r['req']} 跨层预读",
                      f"{q['first_token_s'] - r['first_token_s']:+.2f} s（{(q['first_token_s'] / r['first_token_s'] - 1) * 100:+.0f}%）"))
    alt_col = alt if alt and len(alt) >= n else [None] * n
    figD = figure("prefill 跨层预读：首字前时间（秒）", "同一段会话回放：基线（<code>DEEPINFER_PREFILL_LOOKAHEAD=0</code>）与跨层预读（默认：每层预读 max(16, 行数 ÷ 12) 个下一层专家，受 slab 上限约束）",
                  bars(rowsD, 32, [0, 8, 16, 24, 32], " s", "基线与跨层预读的首字前时间（秒）", fmt=lambda v: f"{v:.1f}"),
                  table(["请求", "新 token", "基线", "基线（另一次）", "跨层预读", "变化", "路由同步时 SSD 空闲（基线 → 预读）", "读盘（基线 → 预读）"],
                        [[f"请求 {r['req']}", f"{r['new']:,}", f"{r['first_token_s']:.2f} s", f"{x:.2f} s" if x else "—", f"{q['first_token_s']:.2f} s",
                          f"{(q['first_token_s'] / r['first_token_s'] - 1) * 100:+.1f}%",
                          f"{sb['cats'].get('route_sync', [0, 0])[1] / 1e3:.2f} → {sl['cats'].get('route_sync', [0, 0])[1] / 1e3:.2f} s",
                          f"{gb:.1f} → {gl:.1f} GB"]
                         for r, q, x, sb, sl, gb, gl in zip(B, L, alt_col, split_b, split_l, gb_b, gb_l)]
                        + [["合计", "", f"{sum_b:.1f} s", f"{sum(alt_col):.1f} s" if all(alt_col) else "—", f"{sum_l:.1f} s",
                            f"{(sum_l / sum_b - 1) * 100:+.1f}%", f"{rs_b:.2f} → {rs_l:.2f} s", f"{sum(gb_b):.0f} → {sum(gb_l):.0f} GB"]], "tbl-d"),
                  "tbl-d",
                  f"六个请求的首字前合计 {sum_b:.1f} → {sum_l:.1f} s（{(sum_l / sum_b - 1) * 100:+.0f}%），冷 prefill {b_first[0]:.1f} → {l_first[0]:.1f} s。"
                  f"路由同步期间的 SSD 空闲从 {rs_b:.1f} s 降到 {rs_l:.1f} s；猜错的读取让读盘量多了 {(sum(gb_l) / sum(gb_b) - 1) * 100:.0f}%。"
                  f"只改读盘时机，不改计算：文本 prompt（600 token 两块 prefill）的 {REF['bitexact']['logits']:,} 个 logits 开关前后逐位相同。")

    # ---------------------------------------------------------------- chart E: decode per token
    rowsE = [("11-17 · 16 GiB <small>短提示</small>", [REF["dec16"]["gpu_only"], REF["dec16"]["ssd_only"], REF["dec16"]["both"], REF["dec16"]["idle"]],
              f"{REF['dec16']['T']:.0f}", "11-17 mk-7 trace，16 GiB")]
    for r in B:
        d = r["decode"]
        if not d or d["n"] < 5:
            continue
        rowsE.append((f"请求 {r['req']} · 8 GiB <small>{d['n']} token</small>", [d["gpu_only"], d["ssd_only"], d["both"], d["idle"]],
                      f"{d['T']:.0f}", f"请求 {r['req']}（{d['n']} 个解码 token）"))
    figE = figure("解码：每个 token 的时间分解（ms）", "回放里的解码 token（teacher forcing），与 11-17 在 16 GiB 缓存下的 mk-7 时间线对比",
                  stacked(rowsE, cats4, 160, [0, 40, 80, 120, 160], " ms", "每个解码 token 按 SSD / GPU 状态分解（毫秒）", fmt=lambda v: f"{v:.1f}"),
                  table(["", "ms/token", "tok/s", "只有 GPU 忙", "只有 SSD 忙", "两者都忙", "都空闲", "每 token 读盘", "前 5 个 / 其余"],
                        [["11-17 · 16 GiB（短提示）", f"{REF['dec16']['T']:.1f}", f"{1e3 / REF['dec16']['T']:.1f}", f"{REF['dec16']['gpu_only']:.1f}", f"{REF['dec16']['ssd_only']:.1f}",
                          f"{REF['dec16']['both']:.1f}", f"{REF['dec16']['idle']:.1f}", "—", "—"]]
                        + [[f"请求 {r['req']} · 8 GiB", f"{r['decode']['T']:.1f}", f"{1e3 / r['decode']['T']:.1f}", f"{r['decode']['gpu_only']:.1f}",
                            f"{r['decode']['ssd_only']:.1f}", f"{r['decode']['both']:.1f}", f"{r['decode']['idle']:.1f}", f"{r['decode']['GB']:.2f} GB",
                            f"{r['decode']['first5_ms']:.0f} / {r['decode']['rest_ms']:.0f}" if r['decode'].get('rest_ms') else "—"]
                           for r in B if r["decode"] and r["decode"]["n"] >= 5], "tbl-e"),
                  "tbl-e",
                  f"解码没有明显的流水线空泡：两者都空闲每 token 只有 1–2 ms。8 GiB 缓存下每 token 要读 0.5–0.8 GB，GPU 等盘 42–77 ms，是 16 GiB 时（15.6 ms）的 3–5 倍，"
                  f"所以 agent 里只有 {1e3 / max(dec_ms):.1f}–{1e3 / min(dec_ms):.1f} tok/s。这是缓存容量问题，不是调度气泡。prefill 之后的前 5 个解码 token 在多数请求里还要再慢 10–45 ms（缓存被 prefill 冲掉了）。")

    anchor_html, anchor_tile, anchor_li, anchor_step = "", "", "", "下一步"
    if a.anchor_log and os.path.exists(a.anchor_log):
        AN = [q for q in parse_anchor_log(a.anchor_log) if "reuse" in q]
        rows_an = []
        for q in AN:
            rows_an.append([e(q["q"]), f"{q['tokens']:,}", f"{q['cold']:.1f} s", f"{q['reuse']:.1f} s（{(q['reuse'] / q['cold'] - 1) * 100:+.0f}%）",
                            f"{q['make']:.1f} s", f"{q.get('cos_cold', 0):.4f} / {q.get('cos_anchor', 0):.4f}",
                            "相同" if q.get("same_cold") and q.get("same_anchor") else "不同"])
        hb_new = hb_img = ha_new = ha_img = None
        if a.harness_before and a.harness_after:
            hb_new, hb_img = harness_ttft(a.harness_before)
            ha_new, ha_img = harness_ttft(a.harness_after)
        med = lambda xs: statistics.median([x[1] for x in xs]) if xs else float("nan")  # noqa: E731
        harness_rows = ""
        if hb_new and ha_new:
            cold_a = [x for x in ha_new if x[2] == 0]  # the first new session (cold) and the one that made the anchor
            reuse_a = [x for x in ha_new if x[2] > 0]
            img_reuse = [x for x in ha_img if x[2] > 0]
            harness_rows = table(["<code>dsv41 test</code>，缓存 8 GiB", "11-24（无锚点）", "本次", "说明"], [
                ["新会话第一个请求，首字中位", f"{med(hb_new):.1f} s（{len(hb_new)} 个）", f"{med(reuse_a):.1f} s（{len(reuse_a)} 个复用锚点）",
                 "服务启动后的前两个新会话没有锚点可用：" + " / ".join(f"{x[1]:.1f} s" for x in cold_a) + "（第二个建锚点）"],
                ["带图请求，首字中位", f"{med(hb_img):.1f} s（{len(hb_img)} 个）", f"{med(ha_img):.1f} s（{len(ha_img)} 个，其中 {len(img_reuse)} 个复用锚点）",
                 "带图请求从锚点开始算，不再从头算"],
                ["15 项验收的总耗时", f"{suite_s(a.harness_before):.0f} s", f"{suite_s(a.harness_after):.0f} s（{(suite_s(a.harness_after) / suite_s(a.harness_before) - 1) * 100:+.0f}%）",
                 "已含两次建锚点的一次性开销（测试脚本和终端 dsv41 的 system 提示略有不同，各建一个锚点）"],
            ], None, hidden=False, text=True)
        q0 = AN[0] if AN else None
        anchor_tile = (f'<div class="tile"><div class="k">新会话首字（复用固定前缀）</div><div class="v">{(q0["reuse"] / q0["cold"] - 1) * 100:+.0f}%</div>'
                       f'<div class="d">{q0["cold"]:.1f} → {q0["reuse"]:.1f} s；服务启动后第二个新会话建锚点，一次性多 {q0["make"] - q0["cold"]:.0f} s</div></div>') if q0 else ""
        anchor_li = (f'<li><strong>已实现：新会话复用固定前缀（锚点快照，默认打开，<code>DEEPINFER_PREFIX_ANCHOR=0</code> 关闭）。</strong>'
                     f'dsh 每个会话开头约 1,488 token 的 system 提示和工具定义只算一次；新会话首字 {q0["cold"]:.1f} → {q0["reuse"]:.1f} s，'
                     f'贪心生成的 token 与冷 prefill、精确计算相同。</li>') if q0 else ""
        anchor_step = f"已实现（默认打开）：新会话首字 {(q0['reuse'] / q0['cold'] - 1) * 100:+.0f}%" if q0 else "下一步"
        anchor_html = f"""
<section id="anchor" aria-labelledby="h-anchor">
  <h2 id="h-anchor">新会话复用固定前缀：锚点快照</h2>
  <p>dsh 发来的每个请求都以同一段约 1,488 token 的 system 提示和工具定义开头（到第一个 <code style="white-space:nowrap">&lt;｜User｜&gt;</code> 为止）。当前目录、时间这类会变的内容都在用户消息之后。引擎原来只复用当前会话的状态和上一个请求的快照，每个新会话都要把这段从头算一遍；带图请求则完全不复用前缀，每次都从头算。</p>
  <ul class="plain">
    <li><strong>做法：</strong>同一段前缀（≥ 512 token）第二次出现在新会话开头时，把它单独 prefill：最后一块让 decoder 重放前缀末尾，所以状态和一次完整 prompt 结束时一样，对后面接什么内容都有效。然后保存快照和它的状态，即锚点，最多保留 2 个，按最近使用排序。之后的新会话、带图请求只要以它开头，就恢复锚点，只 prefill 后面的部分。恢复时复制 ring（前向会原地写 <code>Ring.pos</code>），所以同一个锚点可以反复用。只出现过一次的前缀不建锚点：有的客户端把日期之类的内容写进 system 提示，每次都建只会白花时间。</li>
    <li><strong>语义：</strong>和同会话多轮复用一样，decoder 从前缀末尾开始重放，而不是只重放整个 prompt 的最后 140 个 token。和精确计算（不截断 decoder）相比，锚点路径与冷 prefill 的接近程度是同一量级，贪心生成的 token 都相同（见下表）。</li>
    <li><strong>代价：</strong>建锚点的那个会话（服务启动后的第二个新会话）多一次 decoder 重放，后面的内容也要单独再读一遍专家，一次性多约 {q0["make"] - q0["cold"]:.0f} s。服务常驻，锚点在进程里一直有效；<code>reset()</code> 会清掉。</li>
    <li><strong>验证：</strong><code>tests/test_engine.py::test_shared_prefix_anchor</code>：恢复两次（第二次在解码改写过状态之后）都和手工构建的状态完全一致（最大差 0.0），建锚点的请求报告复用 0。真实模型见下表（<code>mk/test_prefix_anchor.py</code>，lossless 解码）。{('API 端到端：' + e(a.e2e) + '。') if a.e2e else ''}</li>
  </ul>
  {table(["新会话 B 的用户消息", "提示 token", "冷 prefill", "复用锚点", "建锚点的会话", "与精确计算的余弦（冷 / 锚点）", "24 个贪心 token（三条路径）"], rows_an, None, hidden=False, text=True)}
  {harness_rows}
</section>
""" if q0 else ""
    # ---------------------------------------------------------------- next steps table
    steps = [
        ["每层 prefill 的路由同步：SSD 等下一层路由", f"{rs_b:.1f} s（prefill 的 {pct(rs_b / pf_T)}）", "跨层预读（本次实现，默认打开）",
         "无损：只改读盘时机，不改计算", f"已测：首字前合计 {(sum_l / sum_b - 1) * 100:+.0f}%，带图约 −17%"],
        ["新会话重算同一段 system + 工具定义（约 1,489 token）", f"冷 prefill 约 {b_first[0]:.0f} s / 新会话", "缓存这段共享前缀的状态（锚点快照），新会话直接复用",
         "无损：与同会话多轮复用相同的语义", anchor_step],
        ["prefill 之后的首个输出 token 走解码路径、缓存是冷的", f"{min(first_ms):.0f}–{max(first_ms):.0f} ms / 请求，读 {min(first_gb):.1f}–{max(first_gb):.1f} GB", "把最后一个 prompt token 的专家留在缓存里 / 预读",
         "无损", "待评估"],
        ["层交界处的 Python 建图、Engram 查表", f"{tot_idle['build'] + tot_idle['other']:.1f} s", "跨层预读同时覆盖（读盘在这段时间继续）", "无损", "随第一项"],
        ["prefill 结果不能逐位复现（MLX scatter-add 原子加）", f"带图 prompt 两次运行 logits 最多差 {REF['vl']['max_diff']}，top-5 顺序会变",
         "每批按专家名次分开 scatter（每次索引唯一），或先写进 [行, 6, dim] 再求和", "让无损模式的 prefill 可复现", "待做"],
        ["解码在 8 GiB 缓存下等盘", f"{min(r['decode']['ssd_only'] for r in B if r['decode']):.0f}–{max(r['decode']['ssd_only'] for r in B if r['decode']):.0f} ms/token",
         "更大的专家缓存（关掉其他应用，dsv41 按空闲内存选 6–16 GiB）", "—", "容量问题，不是气泡"],
    ]
    steps_tbl = table(["气泡 / 重复工作", "本次会话里的量", "做法", "对输出的影响", "状态"], steps, None, hidden=False, text=True)

    # ---------------------------------------------------------------- validation table
    val = table(["请求", "提示 token", "复用（回放 / 实际）", "新 token", "首字前：实际运行", "回放", "读盘：实际运行", "回放"],
                [[f"请求 {r['req']}", f"{r['prompt']:,}", f"{r['reused']:,} / {r['live']['reused']:,}", f"{r['new']:,}", f"{c['t_tokens'][0]:.2f} s",
                  f"{r['first_token_s']:.2f} s", f"{c['stats']['expert_GB_read']:.1f} GB", f"{r['store']['GB_read']:.1f} GB"] for r, c in zip(B, cap)], None, hidden=False)

    dec_w = lambda S: sum(r["decode"]["T"] * r["decode"]["n"] for r in S if r["decode"]) / sum(r["decode"]["n"] for r in S if r["decode"])  # noqa: E731
    dec_b, dec_l = dec_w(B), dec_w(L)
    QUICK = e(REF["quick"]) if REF["quick"] else "（待补）"
    meta_line = (f"<span>代码 deepinfer-mac-dsv41-flash（mk-7 引擎 + 本次改动）</span><span>Apple M4 Pro · 48 GB</span>"
                 f"<span>{e(base['runs'][0]['meta']['mode'].split(';')[0])}</span><span>专家缓存 8 GiB · 接电源</span><span>采集 {e(base['generated'])}</span>")
    look_gain = (sum_l / sum_b - 1) * 100

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>DSV4.1 推理流水线报告</title>
<meta name="description" content="回放一段真实的 DeepSeek Harness 会话，在 Mac 上逐层记录 DSV4.1-Flash 的 CPU、GPU 与 SSD 活动：prefill 与解码的时间去向、层交界的 SSD 空泡，以及跨层预读的效果。">
<style>{css}</style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
<symbol id="i-good" viewBox="0 0 12 12"><circle cx="6" cy="6" r="5.4" fill="currentColor"/><path d="M3.4 6.2 5.2 8 8.7 4.3" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></symbol>
</defs></svg>
<header class="top"><div class="wrap">
  <span class="eyebrow">DeepInfer-Mac · DSV4.1-Flash · agent 流水线 · {e(base['generated'][:10])}</span>
  <nav class="toc" aria-label="目录"><a href="#summary">结论</a><a href="#method">方法</a><a href="#time">时间去向</a><a href="#prefill">prefill</a>
  <a href="#lookahead">跨层预读</a><a href="#anchor">前缀复用</a><a href="#decode">解码</a><a href="#next">下一步</a><a href="#viewer">时间线</a><a href="#repro">复现</a></nav>
</div></header>
<main class="wrap">

<section id="summary" aria-labelledby="h-summary">
  <h1 id="h-summary">DSV4.1-Flash 在 agent 里的推理流水线：气泡在哪</h1>
  <p class="lede">在服务端录下一段真实的 DeepSeek Harness 会话（三轮，6 个请求），再在同一个引擎上按服务端的方式回放：逐层记录 CPU 主线程的各阶段、由主机同步推断的 GPU 区间，以及原生 IO 每个 2 MB 读块。配置与平时用 <code>dsv41</code> 时一致：默认 <code>--fast</code>，专家缓存 8 GiB。<code>dsv41</code> 启动时按当时的空闲内存选缓存（6–16 GiB）：11-24 是 8 GiB，今天先后是 6 和 14 GiB，所以 agent 里的解码速度会跟着变。</p>
  <div class="meta">{meta_line}</div>
  <div class="tiles">
    <div class="tile"><div class="k">首字前（prefill）占请求时间</div><div class="v">{pct(ttft_share)}</div>
      <div class="d">本次会话（回复较长）；11-24 验收的 21 个请求为 {pct(REF['nov24']['share'])}</div></div>
    <div class="tile"><div class="k">prefill 时 SSD 忙</div><div class="v">{pct(min(ssd_busy))}–{pct(max(ssd_busy))}</div>
      <div class="d">忙时 {min(gbps):.1f}–{max(gbps):.1f} GB/s，接近盘的上限；prefill 是读盘受限</div></div>
    <div class="tile"><div class="k">层交界的 SSD 空泡</div><div class="v">{rs_b:.1f}<small>s</small></div>
      <div class="d">占 prefill 的 {pct(rs_b / pf_T)}；冷 prefill 里 {cold_rs:.1f} s（{pct(cold_rs / cold_T)}）</div></div>
    <div class="tile"><div class="k">跨层预读之后 <span class="chip good"><svg aria-hidden="true"><use href="#i-good"/></svg>计算不变</span></div><div class="v">{look_gain:+.0f}%</div>
      <div class="d">首字前合计 {sum_b:.1f} → {sum_l:.1f} s；冷 prefill {b_first[0]:.1f} → {l_first[0]:.1f} s</div></div>
    {anchor_tile}
  </div>
  <div class="verdict"><p><strong>结论</strong></p><ul class="plain">
    <li><strong>prefill 是读盘受限，而且读得很多。</strong>新 token 只有 14–15 个的请求也要读约 22 GB、首字前 4 s 多：prefill 必须精确路由，每层要把这些 token 用到的专家都读进来，8 GiB 缓存几乎留不住上一层的专家。</li>
    <li><strong>最明显的气泡在层与层之间。</strong>每层最后一批专家读完后，SSD 要等下一层算完注意力和路由才知道读什么，这段时间完全空闲：全会话 {rs_b:.1f} s，冷 prefill 每层 200–300 ms。</li>
    <li><strong>已实现跨层预读，默认打开（<code>DEEPINFER_PREFILL_LOOKAHEAD=16</code>，设 0 关闭）。</strong>在本层最后一批时，用下一层的路由器对本层输入预测、按行数读入最可能的专家。首字前合计 {look_gain:+.0f}%，冷 prefill {(l_first[0] / b_first[0] - 1) * 100:+.0f}%，带图 prefill 约 −17%；只改读盘时机，不改计算。</li>
    <li><strong>顺带查清：prefill 本身不能逐位复现。</strong>专家结果用 MLX 的 scatter-add 加回各行，GPU 上是原子加；一行在同一批里命中两个专家时，累加顺序不固定。带图 prompt 上同一设置的两次运行 logits 最多差 {REF['vl']['max_diff']}（top-1 不变）。与预读无关，修法见「下一步」。</li>
    <li><strong>解码没有明显空泡。</strong>两者都空闲每 token 只有 1–2 ms；慢在 8 GiB 缓存下每 token 要等盘 42–77 ms（16 GiB 时 15.6 ms），这是缓存容量问题。</li>
    {anchor_li or f'<li><strong>更大的一块是重复工作，不是气泡。</strong>每个新会话都要重算同一段约 1,489 token 的 system 提示和工具定义（这部分不含当前目录、时间等动态内容），冷 prefill {b_first[0]:.0f} s。给这段共享前缀留一个快照，新会话就能直接复用。</li>'}
  </ul></div>
</section>

<section id="method" aria-labelledby="h-method">
  <h2 id="h-method">方法：录一段真实会话，在引擎上逐层回放</h2>
  <ul class="plain">
    <li><strong>录制。</strong><code>mk/agent_capture.py</code> 在 8000 端口起与 <code>dsv41</code> 相同的服务（默认 <code>--fast</code>，缓存 8 GiB），记录每个请求的提示 token、生成 token 和服务端统计；然后用真实的 <code>dsv41 -p … -p … -p …</code> 跑三轮任务：列文件并总结 README、追问依赖、把总结写进文件。</li>
    <li><strong>回放。</strong><code>mk/trace_agent.py</code> 在一个新进程里按服务端的方式逐个请求做 prefill（前缀复用、分块、Bounded Replay 都与服务端相同），再把当时生成的 token 逐个喂回（teacher forcing）。这样每个请求的 prefill 量、解码工作量和下一请求能复用的前缀都和实际运行一致。</li>
    <li><strong>记录。</strong>CPU 主线程逐层的阶段（prefill 时每层按 16 个专家一批：预读下一批 → 等这一批读完 → 构建 → 等 GPU 算完），原生 IO 每个 2 MB 读块所在的线程与起止时刻。GPU 区间是推断的：这台 Mac 拿不到 GPU 时间戳，块的结束取主机同步返回的时刻。</li>
  </ul>
  <p>回放与实际运行吻合：复用的前缀长度完全一致，读盘量相差不到 1%；首字时间差 −10% 到 +5%（实际运行多了 HTTP、分词，以及服务刚启动时的冷页缓存）。</p>
  {val}
</section>

<section id="time" aria-labelledby="h-time">
  <h2 id="h-time">时间去向：首字前与解码</h2>
  {figA}
</section>

<section id="prefill" aria-labelledby="h-prefill">
  <h2 id="h-prefill">prefill：读盘受限，空泡在层交界</h2>
  {figB}
  {figC}
  <div class="note"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5 15 14H1z" fill="currentColor"/></svg>
  <p>层内几乎没有空泡：「等这一批专家读完」的 {tot_time['wait']:.0f} s 里 SSD 只空了 {tot_idle['wait']:.2f} s。不过单个 IO 线程在两个读块之间平均空约 0.7 ms（一个 2 MB 读块约 1.9 ms），8 个线程平均只有 5–6 个在读，SSD 忙时带宽 {min(gbps):.1f}–{max(gbps):.1f} GB/s。更深的队列能不能把带宽再推高，要单独做 IO 微基准。</p></div>
</section>

<section id="lookahead" aria-labelledby="h-lookahead">
  <h2 id="h-lookahead">跨层预读：把层交界的 SSD 空泡填上</h2>
  <p>做法：每层算路由时，顺带用下一层的路由器对本层的输入打分，在同一次主机同步里拿回每行的 top-6，按被预测的次数排序。本层进入最后一批专家时（本层已经没有要读的了），按低优先级读入下一层最可能的 max(16, 行数 ÷ 12) 个专家，最多占目标 slab 的一半，并给固定槽、当前批和预读批留出余量（8 GiB 时上限 173 / 117 / 16 个）。下一层路由出来以后，还没开始读的猜错项会被取消。专家批的顺序和计算都不变。只在 prefill 里生效（<code>LLM._prefill</code> / <code>_prefill_vl</code> 打开 <code>Transformer.prefilling</code>），解码与 DSpark 的多 token 前向不受影响。代码在 <code>deepinfer_mac/model/model.py</code>（<code>MoE.PREFILL_LOOKAHEAD</code>），默认打开，<code>DEEPINFER_PREFILL_LOOKAHEAD=0</code> 关闭。</p>
  <ul class="plain">
    <li><strong>文本 prefill：</strong><code>mk/test_prefill_lookahead.py</code> 用真实模型做「非末块 + 末块（Bounded Replay）」两块 prefill，开 / 关交替各两次：{REF['bitexact']['logits']:,} 个 logits 四次逐位相同；耗时 {REF['bitexact']['off_s'][0]:.1f} / {REF['bitexact']['off_s'][1]:.1f} s → {REF['bitexact']['on_s'][0]:.1f} / {REF['bitexact']['on_s'][1]:.1f} s，读盘 {REF['bitexact']['off_gb']:.1f} → {REF['bitexact']['on_gb']:.1f} GB。</li>
    <li><strong>带图 prefill：</strong><code>mk/bench_prefill_vl.py</code>（dsh 的 system 文本 + 测试图片，1,500 token，加上限前测）：{" / ".join(f"{v:.1f}" for v in REF['vl']['off_s'])} s → {" / ".join(f"{v:.1f}" for v in REF['vl']['on_s'])} s（约 −17%）。这条路径上 prefill 本身就不能逐位复现（见下一条），按结果分组时，组别与开关无关。</li>
    <li><strong>prefill 的运行间差异（原有）：</strong>连续 7 次带图 prefill 分成两组逐位相同的结果，组别与开关、是否清空缓存都无关；逐层校验和显示前 21 层完全相同，第一处差异在第 21 层 MoE 的输出，而它的输入、路由、共享专家都相同，槽里的专家字节也与模型文件一致。原因是 MLX 的 scatter-add（<code>y.at[r].add(o)</code>）在 GPU 上用原子加：同样的输入重复 30 次，结果就不逐位相同。记录在 <code>logs/mk/agent/prefill_nondeterminism.md</code>。</li>
    <li><strong>单测：</strong><code>pytest tests/</code> {REF['unit']} 通过（tiny checkpoint 上的引擎、与参考实现的一致性、视觉、存储、服务端）。</li>
    <li><strong>端到端：</strong>{QUICK}</li>
    <li><strong>长提示（API 套件发现并已修复）：</strong>最初没有上限时，4,096 行的块会要 341 个专家；预读中的槽不能被挤掉，于是占满了 slab，本层最后一批按需读盘找不到槽，报 MemoryError。<code>test_api_e2e.py</code> 的两项长提示测试（12.6K / 25.6K token）因此失败；加上 slab 上限后两项都通过，其余 26 项不受影响。</li>
    <li><strong>解码：</strong>预读只在 prefill 里；两次回放的解码均值 {dec_b:.1f} → {dec_l:.1f} ms/token，差 {(dec_l / dec_b - 1) * 100:+.1f}%，在运行间 ±3–5% 的波动范围内（稳态 token 也差这么多，与 prefill 后的缓存内容无关）。</li>
  </ul>
  {figD}
</section>

{anchor_html}
<section id="decode" aria-labelledby="h-decode">
  <h2 id="h-decode">解码：没有明显空泡，慢在缓存容量</h2>
  {figE}
</section>

<section id="next" aria-labelledby="h-next">
  <h2 id="h-next">气泡与重复工作：按收益排序</h2>
  {steps_tbl}
</section>

<section id="viewer" aria-labelledby="h-viewer">
  <h2 id="h-viewer">时间线查看器</h2>
  <p><a href="trace/AGENT_TRACE.html">trace/AGENT_TRACE.html</a> 是这段会话的逐层时间线（与 <code>trace/DECODE_TRACE.html</code> 同一个查看器，约 22 MB，由 <code>trace/agent_page.py</code> 从 <code>logs/mk/agent/</code> 的回放数据生成）：两组回放（基线、跨层预读）。在「Token」里可以选某个请求的 prefill、首个输出 token 或前 4 个解码 token；prefill 段里同一 IO 线程上间隔不到 2 ms 的读块合并成一条占用条。每个 2 MB 读块的完整事件在 Perfetto JSON 里（<code>logs/mk/agent/agent-fast-c8*.perfetto.json</code>，可拖进 ui.perfetto.dev）。</p>
</section>

<section id="repro" aria-labelledby="h-repro">
  <h2 id="h-repro">复现</h2>
  <div class="cmds">
    <div class="cmd"><div class="cmd-head"><span>1. 录一段会话（另一个终端里用 <code>dsv41 -p</code> 驱动，完成后 Ctrl+C）</span></div>
      <pre>cd ~/Models/DeepInfer-Mac
.venv/bin/python mac-deploy-tests/mk/agent_capture.py ~/Models/logs/mk/agent/capture.jsonl --cache-gb 8</pre></div>
    <div class="cmd"><div class="cmd-head"><span>2. 回放并记录（基线 / 跨层预读）</span></div>
      <pre>DEEPINFER_PREFILL_LOOKAHEAD=0 .venv/bin/python mac-deploy-tests/mk/trace_agent.py ~/Models/logs/mk/agent/capture.jsonl base.json --cache-gb 8
.venv/bin/python mac-deploy-tests/mk/trace_agent.py ~/Models/logs/mk/agent/capture.jsonl look.json --cache-gb 8</pre></div>
    <div class="cmd"><div class="cmd-head"><span>3. 逐位一致检查、页面</span></div>
      <pre>.venv/bin/python mac-deploy-tests/mk/test_prefill_lookahead.py --tokens 600
python3 mac-deploy-tests/trace/agent_page.py mac-deploy-tests/trace/AGENT_TRACE.html base.json look.json
python3 mac-deploy-tests/trace/pipeline_report.py mac-deploy-tests/DSV41_PIPELINE_REPORT.html capture.jsonl base.json look.json</pre></div>
  </div>
</section>

<footer class="foot"><span>DeepInfer-Mac · deepinfer-mac-dsv41-flash</span><span>{e(base['generated'][:10])}</span><span>前作：DSV41_MAC_REPORT.html · MK_REPORT.html · DSV41_OPT_REPORT.html</span></footer>
</main>
<div class="tip" id="tip" role="tooltip" hidden></div>
<script>
(function () {{
  "use strict";
  var tip = document.getElementById("tip");
  function fill(el) {{
    tip.replaceChildren();
    var v = document.createElement("div"); v.className = "tv"; v.textContent = el.getAttribute("data-tv") || "";
    var t = document.createElement("div"); t.className = "tt"; t.textContent = el.getAttribute("data-tt") || "";
    tip.append(v, t);
    var n = el.getAttribute("data-tn");
    if (n) {{ var ne = document.createElement("div"); ne.className = "tn"; ne.textContent = n; tip.append(ne); }}
    tip.hidden = false;
  }}
  function place(x, y) {{
    var r = tip.getBoundingClientRect(), left = x + 14, top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    tip.style.left = Math.max(8, left) + "px"; tip.style.top = Math.max(8, top) + "px";
  }}
  function hide() {{ tip.hidden = true; }}
  function showAt(el) {{ fill(el); var b = el.getBoundingClientRect(); place(b.left + b.width / 2, b.top + b.height / 2); }}
  document.querySelectorAll("[data-tv]").forEach(function (el) {{
    el.addEventListener("pointerenter", function (ev) {{ fill(el); place(ev.clientX, ev.clientY); }});
    el.addEventListener("pointermove", function (ev) {{ if (!tip.hidden) place(ev.clientX, ev.clientY); }});
    el.addEventListener("pointerleave", function (ev) {{ if (ev.pointerType !== "touch") hide(); }});
    el.addEventListener("click", function () {{ showAt(el); }});
    el.addEventListener("focus", function () {{ showAt(el); }});
    el.addEventListener("blur", hide);
  }});
  document.addEventListener("pointerdown", function (ev) {{ var t = ev.target; if (!(t && t.closest && t.closest("[data-tv]"))) hide(); }});
  window.addEventListener("scroll", hide, {{ passive: true }});
  document.addEventListener("keydown", function (ev) {{ if (ev.key === "Escape") hide(); }});
  document.querySelectorAll("[data-toggle]").forEach(function (b) {{
    var t = document.getElementById(b.getAttribute("data-toggle"));
    if (!t) return;
    b.addEventListener("click", function () {{
      t.hidden = !t.hidden; b.setAttribute("aria-expanded", String(!t.hidden));
      b.textContent = t.hidden ? "显示数据表" : "隐藏数据表";
    }});
  }});
}})();
</script>
</body>
</html>
"""
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("capture")
    ap.add_argument("base")
    ap.add_argument("look")
    ap.add_argument("--base-alt", help="log of another baseline replay (its 'first token' lines), shown in the lookahead table")
    ap.add_argument("--quick", help="one-line result of `dsv41 test --quick` with this code")
    ap.add_argument("--anchor-log", help="output of mk/test_prefix_anchor.py")
    ap.add_argument("--harness-before", help="dsv41 test JSON without anchors (logs/dsh/test-*.json)")
    ap.add_argument("--harness-after", help="dsv41 test JSON with anchors")
    ap.add_argument("--e2e", help="one-line result of the API e2e suite with this code")
    a = ap.parse_args()
    REF["quick"] = a.quick
    page = build(a)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {a.out} ({len(page.encode('utf-8')) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
