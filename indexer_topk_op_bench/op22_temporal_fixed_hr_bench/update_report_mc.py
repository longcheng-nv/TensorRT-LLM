# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr+mc — add the "GVR multi-CTA (cuteDSL, PR#15198)" arm to the
re-tested REPORT.html sections 1/2 (interactive D blob + CSVs).

The mc arm (gvr_multicta_cutedsl) was measured on umbriel-b200-074
TOGETHER with a co-located gvr_cutedsl anchor re-run (same bundles, same
nsys cold-L2 protocol, OP22RR_ARMS filter). Because the original rr
dataset was collected on other B200 nodes (047/037/044), mc times are
ANCHOR-TRANSFERRED onto the report's scale per cell:

    us_mc_adj = us_mc(074) * us_base(orig) / us_base(074)     (cold & warm)

so the per-cell speedup t(GVR-1CTA)/t(mc) shown in the charts is the
SAME-NODE 074 ratio, re-expressed on the original baseline µs scale
(merge_anchored_ops convention from report/gen_report.py).

Patches: D blob (append mc rows), COL/SHORT JS consts, ock1/ock2
checkboxes, a bilingual method note, CSV re-write with mc columns.
Idempotent: re-running replaces the previously injected mc rows/labels.

Usage: python3 update_report_mc.py [<mc_root>] [<orig_root>]
       defaults ../results_b200_op22rr_mc074  ../results_b200_op22rr
"""
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MC_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op22rr_mc074"
ORIG_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    HERE.parents[0] / "results_b200_op22rr"
REPORT = HERE / "REPORT.html"

SCENS = ["real", "best", "worst"]
SUBS = [("seqlen", "seqlen_sweep", "seq"), ("bs", "bs_scaling", "bs"),
        ("bs_hugeN", "bs_hugeN", "bs")]
OPS = ["gvr_cutedsl", "op21_legacy", "op21_hls", "radix_cutedsl",
       "sglang_streaming"]
MC = "gvr_multicta_cutedsl"
BASE = "gvr_cutedsl"
MC_COL = "#2ec4b6"
MC_SHORT = "GVR-mCTA"


def _detect_nodes():
    """Union of hosts from all mc driver logs (A + node-B helpers)."""
    import re
    nodes = []
    for log in sorted(HERE.glob("mc074*gpu*.log")):
        for m in re.finditer(r"host=([\w.-]+)", log.read_text()):
            if m.group(1) not in nodes:
                nodes.append(m.group(1))
    return " + ".join(nodes) if nodes else "umbriel-b200-074"


MC_NODE = _detect_nodes()


def load(root, want_ops):
    rows = []
    for scen in SCENS:
        for _sw, sub, w in SUBS:
            f = root / scen / sub / "results.jsonl"
            if not f.exists():
                print(f"MISSING {f}")
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["op"] not in want_ops:
                    continue
                if "error" in r or "us_cold" not in r:
                    continue
                row = {"s": scen, "w": w, "K": r["K"], "d": r["dtype"],
                       "N": r["N"], "B": r["BS"], "o": r["op"],
                       "c": round(r["us_cold"], 3),
                       "h": round(r.get("us_warm", r["us_cold"]), 3)}
                if "cluster_size" in r:
                    row["cs"] = r["cluster_size"]
                rows.append(row)
    return rows


def key(r):
    return (r["s"], r["w"], r["K"], r["d"], r["N"], r["B"])


def build_mc_rows(orig_rows, mc_rows_074, base_rows_074):
    base074 = {key(r): r for r in base_rows_074}
    baseorig = {key(r): r for r in orig_rows if r["o"] == BASE}
    out, drift_c, missing = [], [], 0
    for r in mc_rows_074:
        k = key(r)
        b074, borig = base074.get(k), baseorig.get(k)
        if b074 is None or borig is None:
            missing += 1
            continue
        adj = dict(r)
        adj["c"] = round(r["c"] * borig["c"] / b074["c"], 3)
        adj["h"] = round(r["h"] * borig["h"] / b074["h"], 3)
        out.append(adj)
        drift_c.append(b074["c"] / borig["c"])
    if missing:
        print(f"WARNING: {missing} mc cells lacked an anchor pair")
    if drift_c:
        qs = statistics.quantiles(drift_c, n=10)
        print(f"anchor drift base074/baseorig (cold): median="
              f"{statistics.median(drift_c):.4f} p10={qs[0]:.4f} "
              f"p90={qs[-1]:.4f} n={len(drift_c)}")
    return out


def sub1(t, old, new, label):
    assert old in t, f"anchor missing: {label}"
    assert t.count(old) == 1, f"anchor not unique: {label}"
    return t.replace(old, new, 1)


def patch_report(all_rows):
    t = REPORT.read_text()

    # ---- 1. D blob (idempotent: full replace) ----
    i = t.find("const D=[")
    j = t.find("];", i)
    assert i > 0 and j > i
    t = t[:i] + "const D=" + json.dumps(all_rows, separators=(",", ":")) \
        + ";" + t[j + 2:]

    # ---- 2. COL / SHORT consts (idempotent via full-const rewrite) ----
    i = t.find("const COL=")
    j = t.find(";", i)
    assert i > 0 and j > i
    consts = (
        'const COL={"gvr_cutedsl":"#b3e05a","op21_legacy":"#c9a227",'
        '"op21_hls":"#ffd700","gvr_multicta_cutedsl":"' + MC_COL + '",'
        '"radix_cutedsl":"#4ea8de","sglang_streaming":"#d62728"},'
        'DASH={"real":"solid","best":"dash","worst":"dot"},'
        'SHORT={"gvr_cutedsl":"GVR-1CTA","op21_legacy":"op21-legacy",'
        '"op21_hls":"HLS","gvr_multicta_cutedsl":"' + MC_SHORT + '",'
        '"radix_cutedsl":"Radix",'
        '"sglang_streaming":"SGLang"},MAIN="gvr_cutedsl"')
    t = t[:i] + consts + t[j:]

    # ---- 3. checkboxes (skip if already present) ----
    for cls in ("ock1", "ock2"):
        mc_label = (f'<label class="ck"><input type="checkbox" class="{cls}" '
                    f'value="{MC}" checked>{MC_SHORT}</label>')
        if mc_label in t:
            continue
        anchor = (f'<label class="ck"><input type="checkbox" class="{cls}" '
                  f'value="sglang_streaming" checked>SGLang</label>')
        t = sub1(t, anchor, anchor + " " + mc_label, f"{cls} sglang anchor")

    # ---- 4. bilingual method note (skip if already present) ----
    if "mc-note-2026-07-08" not in t:
        note = (
            '<div class="card" style="border-color:#2ec4b6" '
            'id="mc-note-2026-07-08">'
            '<div class="i18n-en"><p><b>GVR-mCTA arm added 2026-07-09</b> '
            '(<code>gvr_multicta_cutedsl</code> — the multi-CTA/cluster GVR '
            'top-K from PR#15198 incl. its host-side cluster_size '
            'auto-dispatch {N&lt;65536→1; BS≤16 &amp; N≥65536→4; '
            'rows×2≤SMs→2; else 1}, i.e. the single-/multi-CTA MIXED '
            'dispatch as it would ship). Measured on ' + MC_NODE + ' (same '
            'B200 SKU, same bundles byte-for-byte, same nsys cold-L2 '
            'protocol, 20 cold / 50 warm reps, kernel HEAD unchanged) '
            'TOGETHER with a co-located GVR-1CTA anchor re-run; plotted µs '
            'are anchor-transferred onto the original baseline scale '
            '(us_mc × us_base(orig)/us_base(074) per cell), so its speedup '
            'curve is the same-node ratio. Per-cell exactness validated at '
            'BS=1. Raw 074 values in '
            '<a href="op22rr_mc_raw074.csv">op22rr_mc_raw074.csv</a>.</p>'
            '</div>'
            '<div class="i18n-zh"><p><b>GVR-mCTA 臂已于 2026-07-09 补充</b>'
            '（<code>gvr_multicta_cutedsl</code> —— PR#15198 的多 CTA/'
            'cluster GVR top-K，含其 host 侧 cluster_size 自动调度 '
            '{N&lt;65536→1；BS≤16 且 N≥65536→4；rows×2≤SM 数→2；否则 1}，'
            '即上线形态的单/多 CTA 混合调度）。采集于 ' + MC_NODE + '（同 '
            'B200 SKU、bundle 逐字节相同、nsys cold-L2 协议不变、20 cold / '
            '50 warm reps、kernel HEAD 不变），并与 GVR-1CTA 锚点臂同机复测；'
            '图中 µs 已按 cell 锚点迁移到原基线刻度（us_mc × us_base(orig)/'
            'us_base(074)），因此其加速比曲线即同机比值。BS=1 处逐格精确性'
            '校验。074 原始值见 <a href="op22rr_mc_raw074.csv">'
            'op22rr_mc_raw074.csv</a>。</p></div></div>\n')
        anchor1 = ('<h2><span class="i18n-en">1. Seq-len sweep (BS=1)</span>'
                   '<span class="i18n-zh">1. 序列长度扫描（BS=1）</span></h2>')
        t = sub1(t, anchor1, note + anchor1, "mc note before s1")

    REPORT.write_text(t)
    print(f"REPORT.html patched: D={len(all_rows)} rows "
          f"({sum(1 for r in all_rows if r['o'] == MC)} mc)")


def write_csvs(all_rows, mc_raw_074, base_074):
    ops_all = OPS[:4] + [MC] + OPS[4:]
    by = {}
    for r in all_rows:
        by.setdefault((r["w"], r["s"], r["K"], r["d"], r["N"], r["B"]),
                      {})[r["o"]] = r
    for w, name in (("seq", "op22rr_seqlen_data.csv"),
                    ("bs", "op22rr_bs_data.csv")):
        head = ["scenario", "K", "dtype", "N", "BS"]
        for o in ops_all:
            head += [f"{o}_cold_us", f"{o}_warm_us"]
        head += [f"speedup_vs_base_{o}" for o in ops_all if o != BASE]
        head += ["mc_cluster_size"]
        out = [head]
        for k in sorted(kk for kk in by if kk[0] == w):
            _w, s, K, d, N, B = k
            ops = by[k]
            row = [s, K, d, N, B]
            for o in ops_all:
                r = ops.get(o)
                row += [r["c"] if r else "", r["h"] if r else ""]
            base = ops.get(BASE)
            for o in ops_all:
                if o == BASE:
                    continue
                r = ops.get(o)
                row.append(round(base["c"] / r["c"], 4)
                           if (base and r and r["c"]) else "")
            mc = ops.get(MC)
            row.append(mc.get("cs", "") if mc else "")
            out.append(row)
        with open(HERE / name, "w", newline="") as f:
            csv.writer(f).writerows(out)
        print(f"wrote {name} ({len(out) - 1} rows)")

    # raw 074 side-file: mc + co-located anchor, unadjusted
    b074 = {key(r): r for r in base_074}
    head = ["scenario", "sweep", "K", "dtype", "N", "BS", "cluster_size",
            "mc_cold_us_074", "mc_warm_us_074",
            "base_cold_us_074", "base_warm_us_074",
            "speedup_same_node_cold"]
    out = [head]
    for r in sorted(mc_raw_074, key=key):
        b = b074.get(key(r))
        out.append([r["s"], r["w"], r["K"], r["d"], r["N"], r["B"],
                    r.get("cs", ""), r["c"], r["h"],
                    b["c"] if b else "", b["h"] if b else "",
                    round(b["c"] / r["c"], 4) if b else ""])
    with open(HERE / "op22rr_mc_raw074.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"wrote op22rr_mc_raw074.csv ({len(out) - 1} rows)")


def main():
    orig_rows = load(ORIG_ROOT, set(OPS))
    mc_raw = load(MC_ROOT, {MC})
    base074 = load(MC_ROOT, {BASE})
    print(f"orig rows={len(orig_rows)} mc074={len(mc_raw)} "
          f"base074={len(base074)}")
    assert orig_rows and mc_raw and base074
    mc_adj = build_mc_rows(orig_rows, mc_raw, base074)
    # D rows carry only s/w/K/d/N/B/o/c/h (+cs harmless but keep blob lean)
    d_rows = orig_rows + [{k: v for k, v in r.items() if k != "cs"}
                          for r in mc_adj]
    patch_report(d_rows)
    write_csvs(orig_rows + mc_adj, mc_adj, base074)


if __name__ == "__main__":
    main()
