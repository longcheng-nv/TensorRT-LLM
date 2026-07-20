#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
Idempotent replacer: rebuild the §9b "Canonical shipped-head tables" details
block from a FRESH same-run 3-arm re-measure at the CURRENT PR#16457 head
@e6fdbfac3d (drive_canonical.sh, 9 fp32 seqlen batches, umbriel-b200-027,
nsys cold-L2), replacing the eae374554c numbers.

Data: headfull_harness/results_canonical/results_canonical_seqlen_fp32.jsonl
(copied from /tmp/gvrcanon_e6fdbfac/refresh_results/results.jsonl by the
runner). Anchor gate vs the 07-20 headfull sweep is computed and embedded.

First run replaces the legacy un-marked block; wraps the new one in
CANONHEAD:BEGIN/END markers, safe to re-run.
"""
import gzip
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "REPORT.html")
DATA = os.path.join(HERE, "headfull_harness", "results_canonical",
                    "results_canonical_seqlen_fp32.jsonl")
DATA_BS = os.path.join(HERE, "headfull_harness", "results_canonical",
                       "results_canonical_bs_real.jsonl")
HEADFULL = os.path.join(HERE, "headfull_harness", "results_headfull",
                        "results_54batch_027plus019.jsonl.gz")
BEGIN = "<!-- CANONHEAD:BEGIN (update_report_canonical.py) -->"
END = "<!-- CANONHEAD:END -->"
LEGACY_RE = (r"<details open><summary class='mut'><b>Canonical shipped-head tables"
             r".*?</details>")
HEAD = "e6fdbfac3d"


def geo(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def _fx(v):
    return f"{v:.2f}×" if v >= 1 else f"{v:.3f}×"


def load_rows():
    rows = [json.loads(l) for l in open(DATA)]
    rows = [r for r in rows if r.get("sweep") == "seqlen" and r.get("dtype") == "fp32"
            and r.get("BS") == 1 and "us" in r]
    key = {}
    for r in rows:
        k = (r["family"], r.get("scenario") or r.get("model"), r["K"], r["N"])
        key.setdefault(k, {})[r["op"]] = r
    return rows, key


def anchor_gate(rows):
    """Fresh op26 rows vs the 07-20 headfull sweep (same node family)."""
    hf = [json.loads(l) for l in gzip.open(HEADFULL, "rt")]
    ref = {(r["family"], r.get("scenario") or r.get("model"), r["K"], r["N"]): r["us"]
           for r in hf if r["op"] == "op26_r0auto" and r.get("sweep") == "seqlen"
           and r.get("dtype") == "fp32" and r.get("BS") == 1 and "us" in r}
    ratios = sorted(ref[(r["family"], r.get("scenario") or r.get("model"), r["K"], r["N"])] / r["us"]
                    for r in rows if r["op"] == "op26_r0auto"
                    and (r["family"], r.get("scenario") or r.get("model"), r["K"], r["N"]) in ref)
    if not ratios:
        return None, None, 0
    med = ratios[len(ratios) // 2]
    p95 = ratios[min(len(ratios) - 1, int(0.95 * len(ratios)))]
    return med, p95, len(ratios)


BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def build_bs():
    """Real-capture BS-scaling table (3 dtypes x all ISL, PR/base geomean per BS)."""
    if not os.path.exists(DATA_BS):
        return ""
    rows = [json.loads(l) for l in open(DATA_BS)]
    rows = [r for r in rows if r.get("sweep") == "bs" and r.get("family") == "real"
            and "us" in r]
    key = {}
    for r in rows:
        key.setdefault((r["model"], r["dtype"], r["N"], r["BS"]), {})[r["op"]] = r
    pairs = {k: d["gvr_base"]["us"] / d["gvr_pr"]["us"] for k, d in key.items()
             if "gvr_pr" in d and "gvr_base" in d}
    pr_exact = sum(d["gvr_pr"]["exact"] for d in key.values() if "gvr_pr" in d)
    n_pr = sum(1 for d in key.values() if "gvr_pr" in d)
    h = ("<tr><th>PR/base geomean</th>" +
         "".join(f"<th>BS {b}</th>" for b in BS_GRID) + "</tr>")
    body = ""
    for m, lab in (("flash", "Flash (K512)"), ("pro", "Pro (K1024)"),
                   ("v32", "V3.2 (K2048)"), (None, "all models")):
        cells = []
        for b in BS_GRID:
            g = [r for k, r in pairs.items() if k[3] == b and (m is None or k[0] == m)]
            cells.append(f"<td>{geo(g):.3f}</td>" if g else "<td>—</td>")
        name = lab if m else f"<b>{lab}</b>"
        body += f"<tr><td>{name}</td>{''.join(cells)}</tr>"
    fp32 = [r for k, r in pairs.items() if k[1] == "fp32"]
    b16 = [r for k, r in pairs.items() if k[1] in ("bf16", "fp16")]
    return (
        "<p><b>Real BS scaling @ current head</b> — all captured ISL rungs × 3 dtypes, "
        f"same-run 3-arm; PR exact {pr_exact}/{n_pr}; overall PR/base geomean "
        f"{geo(list(pairs.values())):.3f}× (fp32 {geo(fp32):.3f}× / 16-bit {geo(b16):.3f}×)</p>"
        f"<table>{h}{body}</table>")


def build():
    rows, key = load_rows()
    med, p95, n_anchor = anchor_gate(rows)
    out = []
    for K, lab in ((512, "K512 · V4 Flash"), (1024, "K1024 · V4 Pro"), (2048, "K2048 · V3.2")):
        ns = sorted({k[3] for k in key if k[0] == "synth" and k[2] == K})
        if not ns:
            continue
        h = ("<tr><th>N</th><th>base best µs</th><th>NEW best µs</th><th>best ↑</th>"
             "<th>base worst µs</th><th>NEW worst µs</th><th>worst ↑</th></tr>")
        b, gb, gw = "", [], []
        for N in ns:
            be = key.get(("synth", "best", K, N), {})
            wo = key.get(("synth", "worst", K, N), {})
            rb = be["gvr_base"]["us"] / be["gvr_pr"]["us"] if "gvr_pr" in be else None
            rw = wo["gvr_base"]["us"] / wo["gvr_pr"]["us"] if "gvr_pr" in wo else None
            if rb:
                gb.append(rb)
            if rw:
                gw.append(rw)
            fmt = lambda d, a: f"{d[a]['us']:.3f}" if a in d else "—"
            b += (f"<tr><td>{N:,}</td><td>{fmt(be,'gvr_base')}</td><td>{fmt(be,'gvr_pr')}</td>"
                  f"<td>{_fx(rb) if rb else '—'}</td><td>{fmt(wo,'gvr_base')}</td>"
                  f"<td>{fmt(wo,'gvr_pr')}</td><td>{_fx(rw) if rw else '—'}</td></tr>")
        out.append(f"<p><b>{lab}</b> — geomean best {geo(gb):.3f}× / worst {geo(gw):.3f}×</p>"
                   f"<table>{h}{b}</table>")
    for m, lab in (("flash", "V4 Flash (K512)"), ("pro", "V4 Pro (K1024)"), ("v32", "V3.2 (K2048)")):
        cells = sorted([k for k in key if k[0] == "real" and k[1] == m], key=lambda k: k[3])
        if not cells:
            continue
        h = ("<tr><th>ISL</th><th>N</th><th>hit</th><th>base µs</th><th>NEW µs</th>"
             "<th>speedup</th><th>exact</th></tr>")
        b, g = "", []
        for k in cells:
            d = key[k]
            pr, ba = d["gvr_pr"], d["gvr_base"]
            sp = ba["us"] / pr["us"]
            g.append(sp)
            b += (f"<tr><td>{pr['isl']}</td><td>{k[3]:,}</td><td>{pr['hit']:.2f}</td>"
                  f"<td>{ba['us']:.3f}</td><td>{pr['us']:.3f}</td><td>{_fx(sp)}</td>"
                  f"<td>{'✓' if pr['exact'] else '✗'}</td></tr>")
        out.append(f"<p><b>Real {lab}</b> — geomean {geo(g):.3f}×</p><table>{h}{b}</table>")

    out.append(build_bs())

    anchor_txt = (f"anchor op26_r0auto vs 07-20 headfull run: median {med:.3f} / p95 {p95:.3f} "
                  f"(n={n_anchor})" if med else "anchor gate unavailable")
    block = (
        "<details open><summary class='mut'><b>Canonical shipped-head tables (BS=1 fp32, per seq-len) "
        "/ 当前 HEAD canonical 表（BS=1 fp32，按序列长）</b></summary>"
        "<div class='lang-en'><p class='mut'>Fresh same-run 3-arm re-measure at the CURRENT PR head "
        f"<code>{HEAD}</code> (base vs NEW; identical grid to §3/§4; 2026-07-20, umbriel-b200-027, "
        "single node, nsys cold-L2 — <code>headfull_harness/drive_canonical.sh</code>). These numbers "
        "supersede the <code>eae374554c</code> tables previously quoted in the PR#16457 body. "
        f"Sanity: {anchor_txt}.</p></div>"
        "<div class='lang-zh'><p class='mut'>在<b>当前</b> PR HEAD "
        f"<code>{HEAD}</code> 下的全新同轮 3 臂重测（base vs NEW；网格与 §3/§4 完全一致;"
        "2026-07-20,umbriel-b200-027 单节点,nsys cold-L2——"
        "<code>headfull_harness/drive_canonical.sh</code>）。本组数字取代此前 PR#16457 body 引用的 "
        f"<code>eae374554c</code> 表。一致性:{anchor_txt}。</p></div>"
        + "".join(out) + "</details>")
    return BEGIN + block + END


def main():
    html = open(REPORT, encoding="utf-8").read()
    block = build()
    if BEGIN in html:
        html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block,
                      html, flags=re.S)
        action = "replaced (marker)"
    else:
        new, n = re.subn(LEGACY_RE, lambda _: block, html, count=1, flags=re.S)
        if n != 1:
            raise SystemExit("legacy canonical block not found")
        html, action = new, "replaced (legacy block)"
    open(REPORT, "w", encoding="utf-8").write(html)
    print(f"canonical block {action} ({len(block)} chars)")


if __name__ == "__main__":
    main()
