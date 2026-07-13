# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr — add the op26_r0auto production arm (iter6 R0 rung ladder +
1CTA/MC arm dispatch) to the re-tested REPORT.html sections 1/2.

THIN LAST-WRITER WRAPPER over update_report_op26_iter5.py (which wraps
update_report_op27.py, the canonical full re-deriver of the mc/op25/
radix/op26/op27 backfill arms, and re-points op26_1cta at the iter5
root): runs u5.main() first — so the report is byte-identical to the
iter5 last-writer state — then appends the op26_r0auto arm from
results_b200_op26_iter6final (074/069 relay, 2026-07-12):

  - exactness gate over the raw per-batch jsonls;
  - per-cell anchor transfer onto the original baseline scale via the
    co-located gvr_cutedsl anchor (house protocol);
  - D blob rows, COL/SHORT consts (full rewrite), section-1/2
    checkboxes, bilingual note card (same-node geomeans computed at
    updater run time from the fin root), methodology-table rows;
  - op26_r0auto columns inserted into op22rr_seqlen_data.csv /
    op22rr_bs_data.csv (which u5 has just rewritten);
  - raw side-file op22rr_op26r_raw.csv incl. the per-cell r0_arm
    (1cta|mc) the dispatch chose (read from the raw batch jsonls; the
    parse step drops it).

Any updater run AFTER this one must itself re-derive op26_r0auto or it
will silently erase the arm.

Usage: python3 parse_op22_cached.py ../results_b200_op26_iter6final
       python3 update_report_op26_iter6.py
"""
import csv
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "update_report_op26_iter5", HERE / "update_report_op26_iter5.py")
u5 = importlib.util.module_from_spec(spec)
u5.__name__ = "update_report_op26_iter5"
spec.loader.exec_module(u5)
u27 = u5.u27

FIN_ROOT = HERE.parents[0] / "results_b200_op26_fin2"
O26R = "op26_r0auto"
RADIX = "radix_cutedsl"
BASE = u27.BASE
O26R_COL = "#00bbf9"
O26R_CANON = "GVR op#26 R0 (auto 1CTA/MC dispatch)"
OP26_ITER6_NOTE_ID = "op26-r0auto-note-2026-07-12"

FIN_NODE = u27._detect_nodes("fin2*_gpu*.log", "umbriel-b200-027 + umbriel-b200-047")

CORE_LO, CORE_HI = 8192, 262144
HUGE_LO = 524288


def _gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan")


def same_node_stats(o26r_raw, base_fin, radix_fin):
    """Per-scenario same-node cold geomeans of radix/base over op26_r0auto
    (>1 = op26_r0auto faster), core domain and hugeN band."""
    b = {u27.key(r): r for r in base_fin}
    x = {u27.key(r): r for r in radix_fin}
    st = {}
    for scen in u27.SCENS:
        for tag, ref in (("rad", x), ("anc", b)):
            core, huge = [], []
            for r in o26r_raw:
                if r["s"] != scen:
                    continue
                p = ref.get(u27.key(r))
                if p is None:
                    continue
                ratio = p["c"] / r["c"]
                if CORE_LO <= r["N"] <= CORE_HI:
                    core.append(ratio)
                elif r["N"] >= HUGE_LO:
                    huge.append(ratio)
            st[(scen, tag, "core")] = _gm(core)
            st[(scen, tag, "huge")] = _gm(huge)
    return st


def _f(v):
    return f"{v:.3f}"


def _op26r_note(st):
    en_nums = (
        f'same-node core-domain (8K&ndash;262K) cold geomean vs Radix '
        f'(cuteDSL): real {_f(st[("real", "rad", "core")])} / best '
        f'{_f(st[("best", "rad", "core")])} / worst '
        f'{_f(st[("worst", "rad", "core")])}; hugeN (&ge;512K) real '
        f'{_f(st[("real", "rad", "huge")])}. Vs the co-located GVR '
        f'(cuteDSL) anchor: real {_f(st[("real", "anc", "core")])} / '
        f'best {_f(st[("best", "anc", "core")])} / worst '
        f'{_f(st[("worst", "anc", "core")])} (the worst-axis P1b '
        f'histogram tax band is a known structural residue), hugeN real '
        f'{_f(st[("real", "anc", "huge")])}.')
    zh_nums = (
        f'同机核心域(8K&ndash;262K)cold 几何均值对 Radix (cuteDSL):'
        f'real {_f(st[("real", "rad", "core")])} / best '
        f'{_f(st[("best", "rad", "core")])} / worst '
        f'{_f(st[("worst", "rad", "core")])};hugeN(&ge;512K)real '
        f'{_f(st[("real", "rad", "huge")])}。对同机 GVR (cuteDSL) 锚:'
        f'real {_f(st[("real", "anc", "core")])} / best '
        f'{_f(st[("best", "anc", "core")])} / worst '
        f'{_f(st[("worst", "anc", "core")])}(worst 轴 P1b 直方图税带为'
        f'已知结构残余),hugeN real {_f(st[("real", "anc", "huge")])}。')
    return (
        '<div class="card" style="border-color:' + O26R_COL + '" '
        'id="' + OP26_ITER6_NOTE_ID + '">'
        '<div class="i18n-en"><p><b>GVR op#26 R0 (auto 1CTA/MC dispatch) '
        'arm added 2026-07-12</b> (<code>op26_r0auto</code> — the op26 '
        'iter6 production head: P1b 256-bin histogram derives a '
        'prev-topK quantile rung ladder, one R0 counting pass picks the '
        'tightest acceptable rung with a cached-column zero-rescan P3, '
        'R0 misses fall back to an inline two-probe log-falsi R1 + '
        'fb_fix; P1b gathered-value cache is dtype-gated (on for '
        '16-bit, off for fp32); dispatch_r0_arm_op26 routes to the '
        'DSMEM-cluster port iff N&ge;65536 and BS&le;64, else the '
        'single-CTA kernel with the op#7 exact rank-scatter P4). '
        + en_nums + ' Measured on ' + FIN_NODE + ' (B200, marker-'
        'idempotent 8-GPU shard relay, same bundles byte-for-byte, same '
        'nsys cold-L2 protocol, 20 cold / 50 warm reps) TOGETHER with '
        'co-located GVR (cuteDSL) and Radix (cuteDSL) re-runs; plotted '
        'µs are anchor-transferred onto the original baseline scale '
        '(us &times; us_base(orig)/us_base(local) per cell). Per-cell '
        'exactness validated at BS=1. Raw local values incl. the '
        'per-cell dispatch choice in '
        '<a href="op22rr_op26r_raw.csv">op22rr_op26r_raw.csv</a>.'
        '</p></div>'
        '<div class="i18n-zh"><p><b>GVR op#26 R0(auto 1CTA/MC '
        'dispatch)臂已于 2026-07-12 补充</b>(<code>op26_r0auto</code> '
        '—— op26 iter6 生产头:P1b 256-bin 直方图从 prev-topK 推分位 '
        'rung 梯,R0 一趟计数选最紧可接受 rung 并用缓存列零重扫 P3,'
        'R0 miss 回退 inline 双实测 log-falsi R1 + fb_fix;P1b '
        'gathered-value cache 按 dtype 门控(16-bit 开、fp32 关);'
        'dispatch_r0_arm_op26 在 N&ge;65536 且 BS&le;64 时路由 DSMEM '
        'cluster 港,否则单 CTA 内核(含 op#7 精确 rank-scatter P4))。'
        + zh_nums + ' 采集于 ' + FIN_NODE + '(B200,marker 幂等 8 卡'
        '分片接力,bundle 逐字节相同,nsys cold-L2 协议不变,20 cold / '
        '50 warm reps),并与 GVR (cuteDSL)、Radix (cuteDSL) 锚臂同机'
        '复测;图中 µs 已按 cell 锚点迁移到原基线刻度(us &times; '
        'us_base(orig)/us_base(local))。BS=1 处逐格精确性校验。本机'
        '原始值(含逐格 dispatch 选择)见 '
        '<a href="op22rr_op26r_raw.csv">op22rr_op26r_raw.csv</a>。'
        '</p></div></div>\n')


def _method_row():
    return (f"<tr><td style='color:{O26R_COL}'>{O26R_CANON} — P1b hist "
            f"quantile rung ladder + one-pass R0 + cached-column P3 + "
            f"inline log-falsi R1 + fb_fix; mc iff (N&ge;65536 &amp; "
            f"BS&le;64); p1b_cache on 16-bit</td>"
            f"<td>CuTe DSL (+DSMEM cluster port)</td>"
            f"<td>exact (heuristic seed)</td>"
            f"<td>full — backfill 2026-07-12, anchor-transferred "
            f"(074/069 relay)</td></tr>")


def load_r0arm(root):
    w_of = {"seqlen": "seq", "bs": "bs", "bs_hugeN": "bs"}
    m = {}
    for f in sorted(root.glob("*/*/results_K*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("op") == O26R and "r0_arm" in r:
                k = (r["scenario"], w_of[r["sweep"]], r["K"], r["dtype"],
                     r["N"], r["BS"])
                m[k] = r["r0_arm"]
    return m


def patch_report_r0(o26r_adj):
    t = u27.REPORT.read_text()

    # ---- 1. D blob: append op26_r0auto rows (u5.main() has just fully
    # rewritten D without them; drop defensively anyway) ----
    i = t.find("const D=[")
    j = t.find("];", i)
    assert i > 0 and j > i
    d = json.loads(t[i + len("const D="):j + 1])
    d = [r for r in d if r["o"] != O26R]
    strip = [{k: v for k, v in r.items() if k not in ("cs", "mp")}
             for r in o26r_adj]
    d += strip
    t = t[:i] + "const D=" + json.dumps(d, separators=(",", ":")) \
        + ";" + t[j + 2:]

    # ---- 2. COL / SHORT consts: full rewrite with op26_r0auto ----
    i = t.find("const COL=")
    j = t.find(";", i)
    assert i > 0 and j > i
    order = ["gvr_cutedsl", "op21_legacy", "op21_hls", u27.MC, u27.OP25,
             u27.RD1, u27.RDM, u27.O26A, u27.O26M, O26R, u27.O27,
             "radix_cutedsl", "sglang_streaming"]
    col_all = {"gvr_cutedsl": "#b3e05a", "op21_legacy": "#c9a227",
               "op21_hls": "#ffd700", "radix_cutedsl": "#4ea8de",
               "sglang_streaming": "#d62728", O26R: O26R_COL,
               **u27.COL_MAP}
    canon = dict(u27.CANON)
    canon[O26R] = O26R_CANON
    consts = (
        "const COL=" + json.dumps({o: col_all[o] for o in order},
                                  separators=(",", ":")) + ","
        'DASH={"real":"solid","best":"dash","worst":"dot"},'
        "SHORT=" + json.dumps({o: canon[o] for o in order},
                              separators=(",", ":")) + ',MAIN="gvr_cutedsl"')
    t = t[:i] + consts + t[j:]

    # ---- 3. checkboxes: append op26_r0auto after op27 (value-keyed) ----
    for cls in ("ock1", "ock2"):
        assert f'class="{cls}" value="{u27.O27}"' in t, \
            f"expected existing {cls} checkbox for {u27.O27}"
        if f'class="{cls}" value="{O26R}"' in t:
            continue
        m = re.search(
            f'<label class="ck"><input type="checkbox" class="{cls}" '
            f'value="{u27.O27}" checked>[^<]*</label>', t)
        assert m, f"{cls} op27 label not found"
        anchor = m.group(0)
        assert t.count(anchor) == 1, f"{cls} anchor not unique"
        lbl = (f'<label class="ck"><input type="checkbox" '
               f'class="{cls}" value="{O26R}" checked>'
               f'{O26R_CANON}</label>')
        t = u27.sub1(t, anchor, anchor + " " + lbl, f"{cls} op26r anchor")
    return t


def insert_notes(t, note_html):
    anchor1 = ('<h2><span class="i18n-en">1. Seq-len sweep (BS=1)</span>'
               '<span class="i18n-zh">1. 序列长度扫描（BS=1）</span></h2>')
    # refresh-if-present: the card carries measured geomeans, so a re-run
    # with more data must replace it, not skip it
    old = re.search(
        '<div class="card"[^>]*id="' + OP26_ITER6_NOTE_ID
        + '">.*?</div></div>\n?', t, re.S)
    if old:
        t = t.replace(old.group(0), "", 1)
    t = u27.sub1(t, anchor1, note_html + anchor1, "op26r note before s1")

    # methodology rows (en+zh tables) — anchor on the full radix row
    row = _method_row()
    probe = f"<tr><td style='color:{O26R_COL}'>"
    if probe not in t:
        rad_row = ("<tr><td style='color:#4ea8de'>Radix (cuteDSL)</td>"
                   "<td>CuTe DSL</td><td>exact (hint-blind)</td>"
                   "<td>full — strongest hint-blind rival</td></tr>")
        assert t.count(rad_row) == 2, "methodology radix rows not found"
        t = u27.subn(t, rad_row, row + rad_row, 2, "methodology row op26r")
    return t


def extend_csvs(o26r_adj):
    """Insert op26_r0auto cold/warm + speedup columns into the two csv
    side-files u5.main() has just rewritten (idempotent: rebuilt fresh
    each run before we touch them)."""
    by = {}
    for r in o26r_adj:
        by[(r["w"], r["s"], r["K"], r["d"], r["N"], r["B"])] = r
    for w, name in (("seq", "op22rr_seqlen_data.csv"),
                    ("bs", "op22rr_bs_data.csv")):
        p = HERE / name
        rows = list(csv.reader(p.open()))
        head = rows[0]
        assert f"{O26R}_cold_us" not in head, f"{name} already extended"
        ci = head.index(f"{u27.O27}_warm_us") + 1
        si = head.index("mc_cluster_size")
        head[ci:ci] = [f"{O26R}_cold_us", f"{O26R}_warm_us"]
        head.insert(si + 2, f"speedup_vs_base_{O26R}")
        bci = head.index(f"{BASE}_cold_us")
        out = [head]
        n_fill = 0
        for row in rows[1:]:
            s, K, d, N, B = row[0], int(row[1]), row[2], int(row[3]), \
                int(row[4])
            r = by.get((w, s, K, d, N, B))
            row[ci:ci] = [r["c"], r["h"]] if r else ["", ""]
            base_c = row[bci]
            sp = round(float(base_c) / r["c"], 4) if (r and base_c) else ""
            row.insert(si + 2, sp)
            if r:
                n_fill += 1
        with open(p, "w", newline="") as f:
            csv.writer(f).writerows(out + rows[1:])
        print(f"extended {name} (+{O26R} cols, {n_fill} cells filled)")


def write_raw_csv(o26r_raw, base_fin, radix_fin, r0arm):
    bl = {u27.key(r): r for r in base_fin}
    xl = {u27.key(r): r for r in radix_fin}
    head = ["scenario", "sweep", "K", "dtype", "N", "BS", "r0_arm",
            "op26r_cold_us_local", "op26r_warm_us_local",
            "base_cold_us_local", "base_warm_us_local",
            "radix_cold_us_local", "radix_warm_us_local",
            "speedup_vs_base_same_node_cold",
            "speedup_vs_radix_same_node_cold"]
    out = [head]
    for r in sorted(o26r_raw, key=u27.key):
        b, x = bl.get(u27.key(r)), xl.get(u27.key(r))
        out.append([r["s"], r["w"], r["K"], r["d"], r["N"], r["B"],
                    r0arm.get(u27.key(r), ""), r["c"], r["h"],
                    b["c"] if b else "", b["h"] if b else "",
                    x["c"] if x else "", x["h"] if x else "",
                    round(b["c"] / r["c"], 4) if b else "",
                    round(x["c"] / r["c"], 4) if x else ""])
    with open(HERE / "op22rr_op26r_raw.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"wrote op22rr_op26r_raw.csv ({len(out) - 1} rows)")


def main():
    assert (FIN_ROOT / "real" / "seqlen_sweep" / "results.jsonl").exists(), \
        "run parse_op22_cached.py ../results_b200_op26_iter6final first"

    u5.main()

    u27.gate_exactness(FIN_ROOT, O26R)
    o26r_raw = u27.load(FIN_ROOT, {O26R})
    base_fin = u27.load(FIN_ROOT, {BASE})
    radix_fin = u27.load(FIN_ROOT, {RADIX})
    assert o26r_raw and base_fin and radix_fin
    print(f"fin: op26r={len(o26r_raw)} base={len(base_fin)} "
          f"radix={len(radix_fin)}")

    borig = [r for r in u27.load(u27.ORIG_ROOT, {BASE})]
    o26r_adj = u27.transfer_ref(borig, o26r_raw, base_fin, "op26r-fin")

    st = same_node_stats(o26r_raw, base_fin, radix_fin)
    for k in sorted(st):
        print(f"  gm[{k[0]:5s} vs {k[1]} {k[2]}] = {st[k]:.4f}")

    t = patch_report_r0(o26r_adj)
    t = insert_notes(t, _op26r_note(st))
    u27.REPORT.write_text(t)
    print(f"REPORT.html patched: +{O26R} ({len(o26r_adj)} rows)")

    extend_csvs(o26r_adj)
    write_raw_csv(o26r_raw, base_fin, radix_fin, load_r0arm(FIN_ROOT))


if __name__ == "__main__":
    main()
