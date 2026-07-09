# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr+op25 — add the "HLS-op25" arm (gvr_ms_auto at the op25 ship
HEAD: w3a ladder + N-gated slot_scale=2 + fp32 C=8 dispatch) to the
re-tested REPORT.html sections 1/2 (interactive D blob + CSVs).

Clone of update_report_mc.py. The op25 arm was measured on a different
node than the original rr dataset (047/037/044) TOGETHER with a
co-located gvr_cutedsl anchor re-run (same bundles byte-for-byte, same
nsys cold-L2 protocol, OP22RR_ARMS filter), so op25 times are
ANCHOR-TRANSFERRED onto the report's scale per cell:

    us_adj = us_op25(local) * us_base(orig) / us_base(local)   (cold & warm)

The mc (GVR-mCTA) rows are re-derived from their own root the same way,
so the D blob full-replace stays idempotent across both backfills.

Patches: D blob, COL/SHORT JS consts, ock1/ock2 checkboxes, a bilingual
method note, CSV re-write with mc + op25 columns.

Usage: python3 update_report_op25.py [<op25_root>] [<orig_root>] [<mc_root>]
       defaults ../results_b200_op25hls028  ../results_b200_op22rr
                ../results_b200_op22rr_mc074
"""
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP25_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op25hls028"
ORIG_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    HERE.parents[0] / "results_b200_op22rr"
MC_ROOT = Path(sys.argv[3]) if len(sys.argv) > 3 else \
    HERE.parents[0] / "results_b200_op22rr_mc074"
REPORT = HERE / "REPORT.html"

SCENS = ["real", "best", "worst"]
SUBS = [("seqlen", "seqlen_sweep", "seq"), ("bs", "bs_scaling", "bs"),
        ("bs_hugeN", "bs_hugeN", "bs")]
OPS = ["gvr_cutedsl", "op21_legacy", "op21_hls", "radix_cutedsl",
       "sglang_streaming"]
MC = "gvr_multicta_cutedsl"
OP25 = "op25_hls"
BASE = "gvr_cutedsl"
OP25_COL = "#ff7f0e"
OP25_SHORT = "HLS-op25"


def _detect_nodes(glob_pat, default):
    import re
    nodes = []
    for log in sorted(HERE.glob(glob_pat)):
        for m in re.finditer(r"host=([\w.-]+)", log.read_text()):
            if m.group(1) not in nodes:
                nodes.append(m.group(1))
    return " + ".join(nodes) if nodes else default


OP25_NODE = _detect_nodes("op25hls028_gpu*.log", "umbriel-b200-028")


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
                if "ms_path" in r:
                    row["mp"] = r["ms_path"]
                rows.append(row)
    return rows


def key(r):
    return (r["s"], r["w"], r["K"], r["d"], r["N"], r["B"])


def gate_exactness(root, arm):
    """Every BS=1 exactness record for `arm` in the raw batch jsonls must
    be ok. The merged results.jsonl drops the field, so scan the raws."""
    n_ok = n_fail = 0
    for f in sorted(root.glob("*/*/results_K*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("op") == arm and "exact" in r:
                if r["exact"] == "ok":
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"EXACT FAIL: {f} {r['K']} {r['dtype']} N={r['N']}")
    print(f"exactness[{arm}]: ok={n_ok} FAIL={n_fail}")
    assert n_fail == 0 and n_ok > 0, "exactness gate failed"


def transfer(orig_rows, new_rows, base_rows_local, tag):
    """Anchor-transfer new_rows onto the original baseline µs scale."""
    blocal = {key(r): r for r in base_rows_local}
    borig = {key(r): r for r in orig_rows if r["o"] == BASE}
    out, drift_c, missing = [], [], 0
    for r in new_rows:
        k = key(r)
        bl, bo = blocal.get(k), borig.get(k)
        if bl is None or bo is None:
            missing += 1
            continue
        adj = dict(r)
        adj["c"] = round(r["c"] * bo["c"] / bl["c"], 3)
        adj["h"] = round(r["h"] * bo["h"] / bl["h"], 3)
        out.append(adj)
        drift_c.append(bl["c"] / bo["c"])
    if missing:
        print(f"WARNING[{tag}]: {missing} cells lacked an anchor pair")
    if drift_c:
        qs = statistics.quantiles(drift_c, n=10)
        print(f"anchor drift base({tag})/baseorig (cold): median="
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
        '"op21_hls":"#ffd700","gvr_multicta_cutedsl":"#2ec4b6",'
        '"op25_hls":"' + OP25_COL + '",'
        '"radix_cutedsl":"#4ea8de","sglang_streaming":"#d62728"},'
        'DASH={"real":"solid","best":"dash","worst":"dot"},'
        'SHORT={"gvr_cutedsl":"GVR-1CTA","op21_legacy":"op21-legacy",'
        '"op21_hls":"HLS","gvr_multicta_cutedsl":"GVR-mCTA",'
        '"op25_hls":"' + OP25_SHORT + '",'
        '"radix_cutedsl":"Radix",'
        '"sglang_streaming":"SGLang"},MAIN="gvr_cutedsl"')
    t = t[:i] + consts + t[j:]

    # ---- 3. checkboxes: insert after the mc checkbox ----
    for cls in ("ock1", "ock2"):
        op25_label = (f'<label class="ck"><input type="checkbox" '
                      f'class="{cls}" value="{OP25}" checked>'
                      f'{OP25_SHORT}</label>')
        if op25_label in t:
            continue
        anchor = (f'<label class="ck"><input type="checkbox" class="{cls}" '
                  f'value="{MC}" checked>GVR-mCTA</label>')
        t = sub1(t, anchor, anchor + " " + op25_label, f"{cls} mc anchor")

    # ---- 4. bilingual method note (skip if already present) ----
    if "op25-note-2026-07-09" not in t:
        note = (
            '<div class="card" style="border-color:' + OP25_COL + '" '
            'id="op25-note-2026-07-09">'
            '<div class="i18n-en"><p><b>HLS-op25 arm added 2026-07-09</b> '
            '(<code>op25_hls</code> — <code>gvr_ms_auto</code> at the op25 '
            'ship HEAD: the w3a threshold ladder (0.92, 0.45, 0.048) for '
            'K512/K1024, slot capacity ×2 gated to N&lt;65536, and the fp32 '
            'C=8 cluster dispatch rule (bs≤8, n≥131072 or n≥65536 &amp; '
            'K≥1024). The existing HLS arm was measured PRE-op25, so the '
            'gap between the two curves is the op25 delta). Measured on '
            + OP25_NODE + ' (same B200 SKU, same bundles byte-for-byte, '
            'same nsys cold-L2 protocol, 20 cold / 50 warm reps) TOGETHER '
            'with a co-located GVR-1CTA anchor re-run; plotted µs are '
            'anchor-transferred onto the original baseline scale '
            '(us_op25 × us_base(orig)/us_base(local) per cell), so its '
            'speedup curve is the same-node ratio. Per-cell exactness '
            'validated at BS=1. Raw local values in '
            '<a href="op22rr_op25_raw028.csv">op22rr_op25_raw028.csv</a>.'
            '</p></div>'
            '<div class="i18n-zh"><p><b>HLS-op25 臂已于 2026-07-09 补充</b>'
            '（<code>op25_hls</code> —— op25 ship HEAD 上的 '
            '<code>gvr_ms_auto</code>：K512/K1024 的 w3a 阈值梯 (0.92, '
            '0.45, 0.048)、slot 容量×2（N&lt;65536 门控）、fp32 C=8 集群'
            '调度规则（bs≤8，n≥131072 或 n≥65536 且 K≥1024）。原 HLS 臂'
            '测于 op25 之前，两条曲线之差即 op25 增量）。采集于 '
            + OP25_NODE + '（同 B200 SKU、bundle 逐字节相同、nsys '
            'cold-L2 协议不变、20 cold / 50 warm reps），并与 GVR-1CTA '
            '锚点臂同机复测；图中 µs 已按 cell 锚点迁移到原基线刻度'
            '（us_op25 × us_base(orig)/us_base(local)），因此其加速比曲线'
            '即同机比值。BS=1 处逐格精确性校验。本机原始值见 '
            '<a href="op22rr_op25_raw028.csv">op22rr_op25_raw028.csv</a>。'
            '</p></div></div>\n')
        anchor1 = ('<h2><span class="i18n-en">1. Seq-len sweep (BS=1)</span>'
                   '<span class="i18n-zh">1. 序列长度扫描（BS=1）</span></h2>')
        t = sub1(t, anchor1, note + anchor1, "op25 note before s1")

    REPORT.write_text(t)
    print(f"REPORT.html patched: D={len(all_rows)} rows "
          f"({sum(1 for r in all_rows if r['o'] == OP25)} op25, "
          f"{sum(1 for r in all_rows if r['o'] == MC)} mc)")


def write_csvs(all_rows, op25_raw_local, base_local):
    ops_all = OPS[:4] + [MC, OP25] + OPS[4:]
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
        head += ["mc_cluster_size", "op25_ms_path"]
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
            o25 = ops.get(OP25)
            row.append(o25.get("mp", "") if o25 else "")
            out.append(row)
        with open(HERE / name, "w", newline="") as f:
            csv.writer(f).writerows(out)
        print(f"wrote {name} ({len(out) - 1} rows)")

    # raw local side-file: op25 + co-located anchor, unadjusted
    bl = {key(r): r for r in base_local}
    head = ["scenario", "sweep", "K", "dtype", "N", "BS", "ms_path",
            "op25_cold_us_local", "op25_warm_us_local",
            "base_cold_us_local", "base_warm_us_local",
            "speedup_same_node_cold"]
    out = [head]
    for r in sorted(op25_raw_local, key=key):
        b = bl.get(key(r))
        out.append([r["s"], r["w"], r["K"], r["d"], r["N"], r["B"],
                    r.get("mp", ""), r["c"], r["h"],
                    b["c"] if b else "", b["h"] if b else "",
                    round(b["c"] / r["c"], 4) if b else ""])
    with open(HERE / "op22rr_op25_raw028.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"wrote op22rr_op25_raw028.csv ({len(out) - 1} rows)")


def main():
    gate_exactness(OP25_ROOT, OP25)
    orig_rows = load(ORIG_ROOT, set(OPS))
    mc_raw = load(MC_ROOT, {MC})
    base_mc = load(MC_ROOT, {BASE})
    op25_raw = load(OP25_ROOT, {OP25})
    base_25 = load(OP25_ROOT, {BASE})
    print(f"orig rows={len(orig_rows)} mc={len(mc_raw)}/{len(base_mc)} "
          f"op25={len(op25_raw)}/{len(base_25)}")
    assert orig_rows and mc_raw and base_mc and op25_raw and base_25
    mc_adj = transfer(orig_rows, mc_raw, base_mc, "mc074")
    op25_adj = transfer(orig_rows, op25_raw, base_25, "op25local")
    # D rows carry only s/w/K/d/N/B/o/c/h (strip cs/mp to keep blob lean)
    strip = lambda rows: [{k: v for k, v in r.items()  # noqa: E731
                           if k not in ("cs", "mp")} for r in rows]
    d_rows = orig_rows + strip(mc_adj) + strip(op25_adj)
    patch_report(d_rows)
    write_csvs(orig_rows + mc_adj + op25_adj, op25_adj, base_25)


if __name__ == "__main__":
    main()
