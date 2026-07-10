# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr+op27 — add the op27_hls arm (gvr_ms_auto at the op27 HEAD:
op25 ship config + the K2048 TAIL ladder 0.75/0.45/0.048) to the
re-tested REPORT.html sections 1/2.

SELF-CONTAINED LAST-WRITER (extends update_report_op26.py): re-derives
the mc, op25, radix AND op26 backfill arms from their own roots, then
adds op27_hls, rewriting the full D blob / COL / SHORT consts — so the
final REPORT state is identical regardless of which updater ran before.
Any updater run AFTER this one must itself re-derive op27 or it will
erase the arm.

Also (a) restores the CANONICAL operator names in the JS SHORT legend
(report/report.html OP_LABEL convention, per the 2026-07-10 report-wide
rename; update_report_op26.py had regressed them to terse forms) and
(b) inserts the missing op26/op27 rows into the bilingual
"Operators & methodology" tables.

Anchor transfer: op27_hls is paired with a co-located gvr_cutedsl
anchor per nsys batch (8-GPU shard on umbriel-b200-027) —
    us_adj = us * us_base(orig) / us_base(local)      (cold & warm)

Usage: python3 update_report_op27.py [<op27_root>]
       default ../results_b200_op22rr_op27027
       (orig/mc/op25/radix/op26 roots fixed as in update_report_op26.py)
"""
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP27_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op22rr_op27027"
OP26A_ROOT = HERE.parents[0] / "results_b200_op26a"
OP26B_ROOT = HERE.parents[0] / "results_b200_op26b"
RADIX_ROOT = HERE.parents[0] / "results_b200_op22rr_radix027"
ORIG_ROOT = HERE.parents[0] / "results_b200_op22rr"
MC_ROOT = HERE.parents[0] / "results_b200_op22rr_mc074"
OP25_ROOT = HERE.parents[0] / "results_b200_op25hls028"
REPORT = HERE / "REPORT.html"

SCENS = ["real", "best", "worst"]
SUBS = [("seqlen", "seqlen_sweep", "seq"), ("bs", "bs_scaling", "bs"),
        ("bs_hugeN", "bs_hugeN", "bs")]
OPS = ["gvr_cutedsl", "op21_legacy", "op21_hls", "radix_cutedsl",
       "sglang_streaming"]
MC = "gvr_multicta_cutedsl"
OP25 = "op25_hls"
RD1 = "radix_single_cuda"
RDM = "radix_multi_cuda"
O26A = "op26_1cta"
O26M = "op26_mc"
O27 = "op27_hls"
BASE = "gvr_cutedsl"
COL_MAP = {MC: "#2ec4b6", OP25: "#ff7f0e", RD1: "#ff8c42", RDM: "#e84855",
           O26A: "#9b5de5", O26M: "#f15bb5", O27: "#c77dff"}
# canonical display names (report/report.html OP_LABEL convention) — used
# for BOTH the plot legend (SHORT) and checkbox labels, per the 2026-07-10
# report-wide operator rename.
CANON = {
    "gvr_cutedsl": "GVR (cuteDSL)",
    "op21_legacy": "GVR op#21 ms_auto (pre-HLS)",
    "op21_hls": "GVR op#21 ms_auto (HLS)",
    MC: "GVR multi-CTA (cuteDSL, PR#15198)",
    OP25: "GVR op#21 ms_auto (HLS-op25)",
    RD1: "Radix single-CTA (CUDA)",
    RDM: "Radix multi-CTA (CUDA)",
    O26A: "GVR op#26 logP2+RS (single-CTA)",
    O26M: "GVR op#26 logP2 (multi-CTA, PR#15198)",
    O27: "GVR op#21 ms_auto (HLS-op27)",
    "radix_cutedsl": "Radix (cuteDSL)",
    "sglang_streaming": "SGLang StreamingTopK",
}


def _detect_nodes(glob_pat, default):
    import re
    nodes = []
    for log in sorted(HERE.glob(glob_pat)):
        for m in re.finditer(r"host=([\w.-]+)", log.read_text()):
            if m.group(1) not in nodes:
                nodes.append(m.group(1))
    return " + ".join(nodes) if nodes else default


OP27_NODE = _detect_nodes("op27_027_gpu*.log", "umbriel-b200-027")


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
    assert n_fail == 0 and n_ok > 0, f"exactness gate failed: {arm}"


def transfer_ref(ref_rows, new_rows, anchor_rows_local, tag):
    """Anchor-transfer new_rows onto ref_rows' µs scale."""
    alocal = {key(r): r for r in anchor_rows_local}
    aref = {key(r): r for r in ref_rows}
    out, drift_c, missing = [], [], 0
    for r in new_rows:
        k = key(r)
        al, ar = alocal.get(k), aref.get(k)
        if al is None or ar is None:
            missing += 1
            continue
        adj = dict(r)
        adj["c"] = round(r["c"] * ar["c"] / al["c"], 3)
        adj["h"] = round(r["h"] * ar["h"] / al["h"], 3)
        out.append(adj)
        drift_c.append(al["c"] / ar["c"])
    if missing:
        print(f"WARNING[{tag}]: {missing} cells lacked an anchor pair")
    if drift_c:
        qs = statistics.quantiles(drift_c, n=10)
        print(f"anchor drift local/ref [{tag}] (cold): median="
              f"{statistics.median(drift_c):.4f} p10={qs[0]:.4f} "
              f"p90={qs[-1]:.4f} n={len(drift_c)}")
    return out


def sub1(t, old, new, label):
    assert old in t, f"anchor missing: {label}"
    assert t.count(old) == 1, f"anchor not unique: {label}"
    return t.replace(old, new, 1)


def subn(t, old, new, n, label):
    assert t.count(old) == n, f"anchor count != {n}: {label}"
    return t.replace(old, new)


OP25_NOTE_ID = "op25-note-2026-07-09"
RADIX_NOTE_ID = "radix-note-2026-07-09"
OP26_NOTE_ID = "op26-note-2026-07-09"
OP27_NOTE_ID = "op27-note-2026-07-10"

# reuse prior updaters' note builders so a missing note is re-inserted
import importlib.util as _ilu  # noqa: E402


def _load_mod(name):
    spec = _ilu.spec_from_file_location(name, HERE / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    mod.__name__ = name
    spec.loader.exec_module(mod)
    return mod


_urx = _load_mod("update_report_radix")
_u26 = _load_mod("update_report_op26")


def _op27_note():
    return (
        '<div class="card" style="border-color:' + COL_MAP[O27] + '" '
        'id="' + OP27_NOTE_ID + '">'
        '<div class="i18n-en"><p><b>GVR op#21 ms_auto (HLS-op27) arm '
        'added 2026-07-10</b> (<code>op27_hls</code> — '
        '<code>gvr_ms_auto</code> at the op27 HEAD: the op25 ship config '
        'plus a K2048 TAIL ladder (0.75, 0.45, 0.048; '
        '<code>OP27_K2048_TAIL</code> default-ON). K512/K1024 binaries '
        'are bit-identical to the HLS-op25 arm; the delta is K2048 only, '
        'where every op22rr WORST loss cell was mode all_ge (the stock '
        'ladder has no tail column). Same-node paired decomposition '
        '(op27_hls_allge_probe/ab_decomp.py): worst gm 1.96&times; vs '
        'the stock ladder (per-cell 1.55&ndash;2.39), best &minus;1.2% / '
        'real &minus;0.7% (noise-level); full-grid host replay shows an '
        'unchanged best/real mode mix. Measured on ' + OP27_NODE + ' '
        '(same B200 SKU, same bundles byte-for-byte, same nsys cold-L2 '
        'protocol, 20 cold / 50 warm reps, 8-GPU shard) TOGETHER with a '
        'co-located GVR (cuteDSL) anchor re-run; plotted µs are '
        'anchor-transferred onto the original baseline scale '
        '(us_op27 &times; us_base(orig)/us_base(local) per cell), so its '
        'speedup curve is the same-node ratio. Per-cell exactness '
        'validated at BS=1. Raw local values in '
        '<a href="op22rr_op27_raw027.csv">op22rr_op27_raw027.csv</a>.'
        '</p></div>'
        '<div class="i18n-zh"><p><b>GVR op#21 ms_auto (HLS-op27) 臂已于 '
        '2026-07-10 补充</b>（<code>op27_hls</code> —— op27 HEAD 上的 '
        '<code>gvr_ms_auto</code>：op25 ship 配置 + K2048 尾梯 (0.75, '
        '0.45, 0.048；<code>OP27_K2048_TAIL</code> 默认开)。K512/K1024 '
        '二进制与 HLS-op25 臂逐位一致；增量仅在 K2048 —— op22rr WORST '
        '的全部失利 cell 均为 all_ge 模式（原梯无尾列）。同机配对分解'
        '（op27_hls_allge_probe/ab_decomp.py）：worst 几何均值对原梯 '
        '1.96&times;（逐 cell 1.55&ndash;2.39），best &minus;1.2% / '
        'real &minus;0.7%（噪声级）；全网格 host 重放显示 best/real '
        '模式分布不变。采集于 ' + OP27_NODE + '（同 B200 SKU、bundle '
        '逐字节相同、nsys cold-L2 协议不变、20 cold / 50 warm reps、'
        '8 卡分片），并与 GVR (cuteDSL) 锚点臂同机复测；图中 µs 已按 '
        'cell 锚点迁移到原基线刻度（us_op27 &times; '
        'us_base(orig)/us_base(local)），因此其加速比曲线即同机比值。'
        'BS=1 处逐格精确性校验。本机原始值见 '
        '<a href="op22rr_op27_raw027.csv">op22rr_op27_raw027.csv</a>。'
        '</p></div></div>\n')


# methodology-table rows (en == zh convention, single string inserted in
# both tables); skip-if-present keyed on the row's color/style prefix.
def _method_row(op, desc, kind, exact, cov):
    return (f"<tr><td style='color:{COL_MAP[op]}'>{desc}</td>"
            f"<td>{kind}</td><td>{exact}</td><td>{cov}</td></tr>")


M_ROW_26A = _method_row(
    O26A, "GVR op#26 logP2+RS (single-CTA) — classic GVR + op13 P2 dispatch "
    "+ corrected P3 fallback + op#7 exact rank-scatter P4 (gated)",
    "CuTe DSL", "exact (heuristic seed)",
    "full — backfill 2026-07-09/10, anchor-transferred (036/027)")
M_ROW_26M = _method_row(
    O26M, "GVR op#26 logP2 (multi-CTA, PR#15198) — cluster kernel + "
    "log-count P2 interpolation (K1024/K2048)",
    "CuTe DSL (DSMEM cluster)", "exact (heuristic seed)",
    "full — backfill 2026-07-09/10, anchor-transferred (036/027, chained)")
M_ROW_27 = _method_row(
    O27, "GVR op#21 ms_auto (HLS-op27) — op25 ship config + K2048 tail "
    "ladder (0.75, 0.45, 0.048); K512/K1024 bit-identical to HLS-op25",
    "CUDA C++ (msc dispatch)", "exact (heuristic pre_idx threshold-seed)",
    "full — backfill 2026-07-10, anchor-transferred (node 027)")


def patch_report(all_rows):
    t = REPORT.read_text()

    # ---- 1. D blob (idempotent: full replace) ----
    i = t.find("const D=[")
    j = t.find("];", i)
    assert i > 0 and j > i
    t = t[:i] + "const D=" + json.dumps(all_rows, separators=(",", ":")) \
        + ";" + t[j + 2:]

    # ---- 2. COL / SHORT consts (full rewrite; SHORT = canonical names) --
    i = t.find("const COL=")
    j = t.find(";", i)
    assert i > 0 and j > i
    order = ["gvr_cutedsl", "op21_legacy", "op21_hls", MC, OP25, RD1, RDM,
             O26A, O26M, O27, "radix_cutedsl", "sglang_streaming"]
    col_all = {"gvr_cutedsl": "#b3e05a", "op21_legacy": "#c9a227",
               "op21_hls": "#ffd700", "radix_cutedsl": "#4ea8de",
               "sglang_streaming": "#d62728", **COL_MAP}
    consts = (
        "const COL=" + json.dumps({o: col_all[o] for o in order},
                                  separators=(",", ":")) + ","
        'DASH={"real":"solid","best":"dash","worst":"dot"},'
        "SHORT=" + json.dumps({o: CANON[o] for o in order},
                              separators=(",", ":")) + ',MAIN="gvr_cutedsl"')
    t = t[:i] + consts + t[j:]

    # ---- 3. checkboxes: append op27 after op26_mc (value-keyed) ----
    import re as _re
    for cls in ("ock1", "ock2"):
        for op in (MC, OP25, RD1, RDM, O26A, O26M):
            assert f'class="{cls}" value="{op}"' in t, \
                f"expected existing {cls} checkbox for {op}"
        if f'class="{cls}" value="{O27}"' not in t:
            m = _re.search(
                f'<label class="ck"><input type="checkbox" class="{cls}" '
                f'value="{O26M}" checked>[^<]*</label>', t)
            assert m, f"{cls} op26_mc label not found"
            anchor = m.group(0)
            assert t.count(anchor) == 1, f"{cls} anchor not unique"
            lbl = (f'<label class="ck"><input type="checkbox" '
                   f'class="{cls}" value="{O27}" checked>'
                   f'{CANON[O27]}</label>')
            t = sub1(t, anchor, anchor + " " + lbl, f"{cls} op27 anchor")

    # ---- 4. bilingual method notes (skip if already present) ----
    anchor1 = ('<h2><span class="i18n-en">1. Seq-len sweep (BS=1)</span>'
               '<span class="i18n-zh">1. 序列长度扫描（BS=1）</span></h2>')
    if OP25_NOTE_ID not in t:
        t = sub1(t, anchor1, _urx._op25_note() + anchor1, "op25 note")
    if RADIX_NOTE_ID not in t:
        t = sub1(t, anchor1, _urx._radix_note() + anchor1, "radix note")
    if OP26_NOTE_ID not in t:
        t = sub1(t, anchor1, _u26._op26_note() + anchor1, "op26 note")
    if OP27_NOTE_ID not in t:
        t = sub1(t, anchor1, _op27_note() + anchor1, "op27 note before s1")

    # ---- 5. methodology-table rows (en+zh tables; skip-if-present) ----
    # insertion anchor = the existing op25 row (unique per table by its
    # style prefix + trailing cell), located as the 2-occurrence literal.
    op25_row_prefix = f"<tr><td style='color:{COL_MAP[OP25]}'>"
    assert t.count(op25_row_prefix) == 2, "op25 methodology rows not found"
    for row, tag in ((M_ROW_26A, O26A), (M_ROW_26M, O26M), (M_ROW_27, O27)):
        if f"<tr><td style='color:{COL_MAP[tag]}'>" in t:
            continue
        # append right before the Radix (cuteDSL) row in both tables
        rad_row_prefix = "<tr><td style='color:#4ea8de'>"
        assert t.count(rad_row_prefix) == 2, "radix cuteDSL rows not found"
        t = subn(t, rad_row_prefix, row + rad_row_prefix, 2,
                 f"methodology row {tag}")

    REPORT.write_text(t)
    counts = {o: sum(1 for r in all_rows if r["o"] == o)
              for o in (MC, OP25, RD1, RDM, O26A, O26M, O27)}
    print(f"REPORT.html patched: D={len(all_rows)} rows  backfill={counts}")


def write_csvs(all_rows, op27_raw_local, base_27_local):
    ops_all = OPS[:4] + [MC, OP25, RD1, RDM, O26A, O26M, O27] + OPS[4:]
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
        head += ["mc_cluster_size", "op25_ms_path", "op26_mc_cluster_size",
                 "op27_ms_path"]
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
            o26m = ops.get(O26M)
            row.append(o26m.get("cs", "") if o26m else "")
            o27 = ops.get(O27)
            row.append(o27.get("mp", "") if o27 else "")
            out.append(row)
        with open(HERE / name, "w", newline="") as f:
            csv.writer(f).writerows(out)
        print(f"wrote {name} ({len(out) - 1} rows)")

    # raw local side-file: op27 arm + co-located anchor, unadjusted
    bl = {key(r): r for r in base_27_local}
    head = ["scenario", "sweep", "K", "dtype", "N", "BS", "op",
            "op27_cold_us_local", "op27_warm_us_local",
            "base_cold_us_local", "base_warm_us_local",
            "speedup_same_node_cold"]
    out = [head]
    for r in sorted(op27_raw_local, key=lambda r: (r["o"],) + key(r)):
        b = bl.get(key(r))
        out.append([r["s"], r["w"], r["K"], r["d"], r["N"], r["B"],
                    r["o"], r["c"], r["h"],
                    b["c"] if b else "", b["h"] if b else "",
                    round(b["c"] / r["c"], 4) if b else ""])
    with open(HERE / "op22rr_op27_raw027.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"wrote op22rr_op27_raw027.csv ({len(out) - 1} rows)")


def main():
    for root, arms in ((RADIX_ROOT, (RD1, RDM)), (OP26A_ROOT, (O26A,)),
                       (OP26B_ROOT, (O26M,)), (OP27_ROOT, (O27,))):
        for arm in arms:
            gate_exactness(root, arm)

    orig_rows = load(ORIG_ROOT, set(OPS))
    mc_raw = load(MC_ROOT, {MC})
    base_mc = load(MC_ROOT, {BASE})
    op25_raw = load(OP25_ROOT, {OP25})
    base_25 = load(OP25_ROOT, {BASE})
    radix_raw = load(RADIX_ROOT, {RD1, RDM})
    base_rd = load(RADIX_ROOT, {BASE})
    o26a_raw = load(OP26A_ROOT, {O26A})
    base_26a = load(OP26A_ROOT, {BASE})
    o26m_raw = load(OP26B_ROOT, {O26M})
    mc_26b = load(OP26B_ROOT, {MC})
    o27_raw = load(OP27_ROOT, {O27})
    base_27 = load(OP27_ROOT, {BASE})
    print(f"orig={len(orig_rows)} mc={len(mc_raw)}/{len(base_mc)} "
          f"op25={len(op25_raw)}/{len(base_25)} "
          f"radix={len(radix_raw)}/{len(base_rd)} "
          f"op26a={len(o26a_raw)}/{len(base_26a)} "
          f"op26b={len(o26m_raw)}/{len(mc_26b)} "
          f"op27={len(o27_raw)}/{len(base_27)}")
    assert all((orig_rows, mc_raw, base_mc, op25_raw, base_25, radix_raw,
                base_rd, o26a_raw, base_26a, o26m_raw, mc_26b,
                o27_raw, base_27))

    borig = [r for r in orig_rows if r["o"] == BASE]
    mc_adj = transfer_ref(borig, mc_raw, base_mc, "mc074")
    op25_adj = transfer_ref(borig, op25_raw, base_25, "op25local")
    rd_adj = transfer_ref(borig, radix_raw, base_rd, "radix027")
    o26a_adj = transfer_ref(borig, o26a_raw, base_26a, "op26a")
    o26m_adj = transfer_ref(mc_adj, o26m_raw, mc_26b, "op26b-chained")
    o27_adj = transfer_ref(borig, o27_raw, base_27, "op27-027")

    strip = lambda rows: [{k: v for k, v in r.items()  # noqa: E731
                           if k not in ("cs", "mp")} for r in rows]
    d_rows = (orig_rows + strip(mc_adj) + strip(op25_adj) + strip(rd_adj)
              + strip(o26a_adj) + strip(o26m_adj) + strip(o27_adj))
    patch_report(d_rows)
    write_csvs(orig_rows + mc_adj + op25_adj + rd_adj + o26a_adj
               + o26m_adj + o27_adj, o27_raw, base_27)


if __name__ == "__main__":
    main()
