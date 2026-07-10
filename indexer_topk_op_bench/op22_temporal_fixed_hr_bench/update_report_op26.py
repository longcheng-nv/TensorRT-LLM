# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr+op26 — add the op26 arms (op26_1cta / op26_mc: HLS-op25's
log-falsi + exact rank-scatter ideas ported back to the classic GVR
kernels, bucket op26_gvr_logfalsi_rs/) to the re-tested REPORT.html
sections 1/2.

SELF-CONTAINED LAST-WRITER (extends update_report_radix.py): re-derives
the mc, op25 AND radix backfill arms from their own roots, then adds the
two op26 arms, rewriting the full D blob / COL / SHORT consts — so the
final REPORT state is identical regardless of which updater ran before.
Any updater run AFTER this one must itself re-derive op26 or it will
erase these arms (same gotcha as update_report_radix.py).

Anchor transfer (anchors are co-located PER BATCH — each nsys batch runs
the op26 arm and its anchor arm back-to-back on the same GPU, so the
transfer is node-agnostic even though the campaign spanned b200-036 (4
early op26a batches) and b200-027 (the rest, 6-GPU dtype shards)):
  op26_1cta paired with a gvr_cutedsl anchor
      -> us_adj = us * us_base(orig) / us_base(local)         (direct)
  op26_mc   paired with a gvr_multicta_cutedsl anchor
      -> us_adj = us * us_mc_adj(074->orig) / us_mc(local)
      (CHAINED via the mc arm's own anchor-transferred rows)

Usage: python3 update_report_op26.py [<op26a_root>] [<op26b_root>]
       defaults ../results_b200_op26a  ../results_b200_op26b
       (radix/orig/mc/op25 roots as in update_report_radix.py)
"""
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP26A_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op26a"
OP26B_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    HERE.parents[0] / "results_b200_op26b"
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
BASE = "gvr_cutedsl"
COL_MAP = {MC: "#2ec4b6", OP25: "#ff7f0e", RD1: "#ff8c42", RDM: "#e84855",
           O26A: "#9b5de5", O26M: "#f15bb5"}
SHORT_MAP = {MC: "GVR-mCTA", OP25: "HLS-op25", RD1: "Radix-1CTA",
             RDM: "Radix-mCTA", O26A: "op26-1CTA", O26M: "op26-mCTA"}
# checkbox display names follow the report's long-label convention
# ("Radix single-CTA (CUDA)", "GVR op#21 ms_auto (HLS)", ...); SHORT_MAP
# stays terse for the plot legend.
LABEL_MAP = {O26A: "GVR op#26 logP2+RS (single-CTA)",
             O26M: "GVR op#26 logP2 (multi-CTA, PR#15198)"}


def _detect_nodes(glob_pat, default):
    import re
    nodes = []
    for log in sorted(HERE.glob(glob_pat)):
        for m in re.finditer(r"host=([\w.-]+)", log.read_text()):
            if m.group(1) not in nodes:
                nodes.append(m.group(1))
    return " + ".join(nodes) if nodes else default


OP25_NODE = _detect_nodes("op25hls028_gpu*.log", "umbriel-b200-028")
RADIX_NODE = _detect_nodes("radix027_gpu*.log", "umbriel-b200-027")
OP26_NODE = _detect_nodes("op26*_gpu*.log", "umbriel-b200-036 + umbriel-b200-027")


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
    """Anchor-transfer new_rows onto ref_rows' µs scale.

    ref_rows: rows already ON the report scale for the anchor op.
    anchor_rows_local: the SAME op measured in the new root (co-located)."""
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


OP25_NOTE_ID = "op25-note-2026-07-09"
RADIX_NOTE_ID = "radix-note-2026-07-09"
OP26_NOTE_ID = "op26-note-2026-07-09"

# reuse the radix updater's note builders verbatim so a missing note is
# re-inserted identically
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("update_report_radix",
                                     HERE / "update_report_radix.py")
_urx = _ilu.module_from_spec(_spec)
# prevent the radix updater from running main() on import
_urx.__name__ = "update_report_radix"
_spec.loader.exec_module(_urx)


def _op26_note():
    return (
        '<div class="card" style="border-color:' + COL_MAP[O26A] + '" '
        'id="' + OP26_NOTE_ID + '">'
        '<div class="i18n-en"><p><b>op26 arms added 2026-07-09</b> '
        '(<code>op26_1cta</code> / <code>op26_mc</code> — the HLS-op25 '
        'ideas ported back to the classic GVR kernels, bucket '
        '<code>op26_gvr_logfalsi_rs/</code>. <code>op26_1cta</code> = '
        'single-CTA GVR + the op13 iter8c P2 dispatch (fp32: K512 '
        'lin-narrow N&le;65536, K1024/K2048 log-count interpolation with '
        'narrowed [kK,kCC] windows; 16-bit: log-interp, stock windows, '
        'K1024/K2048 only) + a corrected P3 fallback (bounded two-sided '
        'refine with log-falsi aim — the fix for the vendored '
        'undershoot-exit that caused report.html &sect;5\'s real-data '
        'red card) + the op#7 EXACT rank-scatter P4 gated to its '
        'production-win region (fp32 anywhere, 16-bit at BS&ge;256). '
        '<code>op26_mc</code> = the PR#15198 cluster kernel + log-count '
        'P2 interpolation on K1024/K2048 (K512 keeps stock: the op13 '
        'K512-log falsification reproduced on this hardware). Measured '
        'on ' + OP26_NODE + ' (same B200 SKU, same bundles '
        'byte-for-byte, same nsys cold-L2 protocol, 20 cold / 50 warm '
        'reps) with co-located anchors: op26_1cta paired with '
        'gvr_cutedsl (direct transfer), op26_mc paired with '
        'gvr_multicta_cutedsl (CHAINED transfer via the mc arm\'s own '
        'anchor-adjusted rows). Per-cell exactness validated at BS=1 '
        'plus a dedicated 3-suite gate (op22rr bundles / hr&isin;{0,1} '
        'adversarial undershoot / 16-bit tie plateaus). Raw local '
        'values in <a href="op22rr_op26_raw.csv">'
        'op22rr_op26_raw.csv</a>.</p></div>'
        '<div class="i18n-zh"><p><b>op26 两臂已于 2026-07-09 补充</b>'
        '（<code>op26_1cta</code> / <code>op26_mc</code> —— 把 '
        'HLS-op25 的两个思想移植回经典 GVR 内核，bucket '
        '<code>op26_gvr_logfalsi_rs/</code>。<code>op26_1cta</code> = '
        '单 CTA GVR + op13 iter8c P2 调度（fp32：K512 线性窄窗 '
        'N&le;65536、K1024/K2048 log-count 插值 + 收窄 [kK,kCC] 窗口；'
        '16-bit：log 插值、原窗口、仅 K1024/K2048）+ 修正版 P3 回退'
        '（双向有界精化 + log-falsi 瞄准 —— 修复 vendored undershoot '
        '漏出，即 report.html &sect;5 真实数据红牌的根因）+ op#7 精确 '
        'rank-scatter P4（按其生产胜域门控：fp32 全域、16-bit '
        'BS&ge;256）。<code>op26_mc</code> = PR#15198 集群内核 + '
        'K1024/K2048 log-count P2 插值（K512 保持原样：op13 的 '
        'K512-log 证伪在本机复现）。采集于 ' + OP26_NODE + '（同 '
        'B200 SKU、bundle 逐字节相同、nsys cold-L2 协议不变、20 cold '
        '/ 50 warm reps），同机锚点配对：op26_1cta 配 gvr_cutedsl'
        '（直接迁移）、op26_mc 配 gvr_multicta_cutedsl（经 mc 臂已迁移'
        '行的链式迁移）。BS=1 逐格精确性校验，另有三套件独立门禁'
        '（op22rr bundles / hr&isin;{0,1} 对抗 undershoot / 16-bit '
        'tie 平台）。本机原始值见 <a href="op22rr_op26_raw.csv">'
        'op22rr_op26_raw.csv</a>。</p></div></div>\n')


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
        '"op21_hls":"#ffd700","gvr_multicta_cutedsl":"' + COL_MAP[MC] + '",'
        '"op25_hls":"' + COL_MAP[OP25] + '",'
        '"radix_single_cuda":"' + COL_MAP[RD1] + '",'
        '"radix_multi_cuda":"' + COL_MAP[RDM] + '",'
        '"op26_1cta":"' + COL_MAP[O26A] + '",'
        '"op26_mc":"' + COL_MAP[O26M] + '",'
        '"radix_cutedsl":"#4ea8de","sglang_streaming":"#d62728"},'
        'DASH={"real":"solid","best":"dash","worst":"dot"},'
        'SHORT={"gvr_cutedsl":"GVR-1CTA","op21_legacy":"op21-legacy",'
        '"op21_hls":"HLS","gvr_multicta_cutedsl":"' + SHORT_MAP[MC] + '",'
        '"op25_hls":"' + SHORT_MAP[OP25] + '",'
        '"radix_single_cuda":"' + SHORT_MAP[RD1] + '",'
        '"radix_multi_cuda":"' + SHORT_MAP[RDM] + '",'
        '"op26_1cta":"' + SHORT_MAP[O26A] + '",'
        '"op26_mc":"' + SHORT_MAP[O26M] + '",'
        '"radix_cutedsl":"Radix",'
        '"sglang_streaming":"SGLang"},MAIN="gvr_cutedsl"')
    t = t[:i] + consts + t[j:]

    # ---- 3. checkboxes: append o26a/o26m after the last existing arm ----
    # Presence is judged on the value attribute (display text has drifted
    # from SHORT_MAP in the live report); the insertion anchor is the full
    # radix_multi_cuda label element located by regex, robust to its text.
    import re as _re
    for cls in ("ock1", "ock2"):
        def lbl(op):
            return (f'<label class="ck"><input type="checkbox" '
                    f'class="{cls}" value="{op}" checked>'
                    f'{LABEL_MAP[op]}</label>')
        for op in (MC, OP25, RD1, RDM):
            assert f'class="{cls}" value="{op}"' in t, \
                f"expected existing {cls} checkbox for {op}"
        m = _re.search(
            f'<label class="ck"><input type="checkbox" class="{cls}" '
            f'value="radix_multi_cuda" checked>[^<]*</label>', t)
        assert m, f"{cls} radix_multi_cuda label not found"
        anchor = m.group(0)
        assert t.count(anchor) == 1, f"{cls} anchor not unique"
        for op in (O26A, O26M):
            if f'class="{cls}" value="{op}"' not in t:
                t = sub1(t, anchor, anchor + " " + lbl(op),
                         f"{cls} anchor for {op}")
            anchor = lbl(op) if lbl(op) in t else anchor

    # ---- 4. bilingual method notes (skip if already present) ----
    anchor1 = ('<h2><span class="i18n-en">1. Seq-len sweep (BS=1)</span>'
               '<span class="i18n-zh">1. 序列长度扫描（BS=1）</span></h2>')
    if OP25_NOTE_ID not in t:
        t = sub1(t, anchor1, _urx._op25_note() + anchor1, "op25 note")
    if RADIX_NOTE_ID not in t:
        t = sub1(t, anchor1, _urx._radix_note() + anchor1, "radix note")
    if OP26_NOTE_ID not in t:
        t = sub1(t, anchor1, _op26_note() + anchor1, "op26 note before s1")

    REPORT.write_text(t)
    counts = {o: sum(1 for r in all_rows if r["o"] == o)
              for o in (MC, OP25, RD1, RDM, O26A, O26M)}
    print(f"REPORT.html patched: D={len(all_rows)} rows  backfill={counts}")


def write_csvs(all_rows, op26_raw_local, anchors_local):
    ops_all = OPS[:4] + [MC, OP25, RD1, RDM, O26A, O26M] + OPS[4:]
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
        head += ["mc_cluster_size", "op25_ms_path", "op26_mc_cluster_size"]
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
            out.append(row)
        with open(HERE / name, "w", newline="") as f:
            csv.writer(f).writerows(out)
        print(f"wrote {name} ({len(out) - 1} rows)")

    # raw local side-file: op26 arms + their co-located anchors, unadjusted
    al = {(r["o"],) + key(r): r for r in anchors_local}
    anchor_of = {O26A: BASE, O26M: MC}
    head = ["scenario", "sweep", "K", "dtype", "N", "BS", "op",
            "op26_cold_us_local", "op26_warm_us_local", "anchor_op",
            "anchor_cold_us_local", "anchor_warm_us_local",
            "speedup_same_node_cold"]
    out = [head]
    for r in sorted(op26_raw_local, key=lambda r: (r["o"],) + key(r)):
        aop = anchor_of[r["o"]]
        b = al.get((aop,) + key(r))
        out.append([r["s"], r["w"], r["K"], r["d"], r["N"], r["B"],
                    r["o"], r["c"], r["h"], aop,
                    b["c"] if b else "", b["h"] if b else "",
                    round(b["c"] / r["c"], 4) if b else ""])
    with open(HERE / "op22rr_op26_raw.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"wrote op22rr_op26_raw.csv ({len(out) - 1} rows)")


def main():
    # exactness gates on every backfill root this updater re-derives
    for root, arms in ((RADIX_ROOT, (RD1, RDM)), (OP26A_ROOT, (O26A,)),
                       (OP26B_ROOT, (O26M,))):
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
    print(f"orig={len(orig_rows)} mc={len(mc_raw)}/{len(base_mc)} "
          f"op25={len(op25_raw)}/{len(base_25)} "
          f"radix={len(radix_raw)}/{len(base_rd)} "
          f"op26a={len(o26a_raw)}/{len(base_26a)} "
          f"op26b={len(o26m_raw)}/{len(mc_26b)}")
    assert all((orig_rows, mc_raw, base_mc, op25_raw, base_25, radix_raw,
                base_rd, o26a_raw, base_26a, o26m_raw, mc_26b))

    borig = [r for r in orig_rows if r["o"] == BASE]
    mc_adj = transfer_ref(borig, mc_raw, base_mc, "mc074")
    op25_adj = transfer_ref(borig, op25_raw, base_25, "op25local")
    rd_adj = transfer_ref(borig, radix_raw, base_rd, "radix027")
    o26a_adj = transfer_ref(borig, o26a_raw, base_26a, "op26a-038")
    # CHAINED: op26_mc onto the mc arm's already-adjusted rows
    o26m_adj = transfer_ref(mc_adj, o26m_raw, mc_26b, "op26b-038-chained")

    strip = lambda rows: [{k: v for k, v in r.items()  # noqa: E731
                           if k not in ("cs", "mp")} for r in rows]
    d_rows = (orig_rows + strip(mc_adj) + strip(op25_adj) + strip(rd_adj)
              + strip(o26a_adj) + strip(o26m_adj))
    patch_report(d_rows)
    write_csvs(orig_rows + mc_adj + op25_adj + rd_adj + o26a_adj + o26m_adj,
               o26a_raw + o26m_raw, base_26a + mc_26b)


if __name__ == "__main__":
    main()
