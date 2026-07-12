# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr — re-point the op26_1cta arm at the iter5 re-test.

THIN LAST-WRITER WRAPPER over update_report_op27.py (which stays the
canonical full re-deriver of mc/op25/radix/op26/op27): overrides
OP26A_ROOT to results_b200_op26a_iter5 (umbriel-b200-037 8-GPU shard,
2026-07-10, dispatch @ iter5d = V1 center aim + per-cell pruning +
secant2 only on K2048 16-bit n==262144), runs u27.main(), then appends
a bilingual iter5 paragraph INSIDE the existing op26 note card
(idempotent via OP26_ITER5_MARK).

The op26_mc arm intentionally keeps the iter4 campaign root — the mc
kernel was untouched by iter5 and its no-ship verdict stands.

Any updater run AFTER this one must re-derive op26a from the iter5 root
or it will silently regress the arm to the iter4 numbers.

Usage: python3 update_report_op26_iter5.py
"""
import csv
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "update_report_op27", HERE / "update_report_op27.py")
u27 = importlib.util.module_from_spec(spec)
u27.__name__ = "update_report_op27"
spec.loader.exec_module(u27)

spec26 = importlib.util.spec_from_file_location(
    "update_report_op26", HERE / "update_report_op26.py")
u26 = importlib.util.module_from_spec(spec26)
u26.__name__ = "update_report_op26"
spec26.loader.exec_module(u26)

u27.OP26A_ROOT = HERE.parents[0] / "results_b200_op26a_iter5"
u26.OP26A_ROOT = u27.OP26A_ROOT   # raw-csv side-file follows the same root

OP26_ITER5_MARK = "op26-iter5-2026-07-10"

EN_TAIL = ('Raw local values in <a href="op22rr_op26_raw.csv">'
           'op22rr_op26_raw.csv</a>.</p>')
ZH_TAIL = ('本机原始值见 <a href="op22rr_op26_raw.csv">'
           'op22rr_op26_raw.csv</a>。</p>')

EN_ITER5 = (
    '<p id="' + OP26_ITER5_MARK + '"><b>iter5 re-dispatch (2026-07-10, '
    'umbriel-b200-037)</b>: the <code>op26_1cta</code> rows now come from '
    'a full 81-batch same-node re-test with the iter5d dispatch table. '
    'Silicon ablation falsified the host-replay V3 log-secant as a '
    'default (pass savings &lt; loop cost); the shipped table keeps the '
    'geometric-center aim &radic;(kK&middot;kCC) only where it measures '
    'ahead (K1024 fp32 N&le;32K; K1024 16-bit 32-64K), keeps the iter4 '
    'edge aim where THAT wins (K2048 fp32; K1024 16-bit 8192), retains '
    'the log-secant solely on K2048 16-bit N=262144 (0.996&rarr;1.14 '
    'both dtypes, does not extrapolate to 512K), and prunes persistent '
    'loss bands back to stock P2 (K1024@131K all dtypes, K1024 16-bit '
    '4K/16K/&ge;131K, K2048 16-bit 16K-131K &amp; &ge;512K). Grid '
    'geomean vs the co-located gvr_cutedsl anchor: 1.032&rarr;1.06+ '
    'overall; fp32 1.100&rarr;1.13 (win 90%); 16-bit flips positive '
    '(bf16 0.994&rarr;1.02+, fp16 1.005&rarr;1.03+).</p>')

ZH_ITER5 = (
    '<p><b>iter5 重调度（2026-07-10,umbriel-b200-037）</b>:'
    '<code>op26_1cta</code> 行已替换为 iter5d 调度表的同节点 81 批全网格'
    '重测。硅上消融证伪了 host 重放推荐的 log-secant 默认路径(趟数节省 '
    '&lt; 循环开销);上线表仅在实测占优处保留几何中心瞄准 '
    '&radic;(kK&middot;kCC)(K1024 fp32 N&le;32K、K1024 16-bit 32-64K),'
    '在 edge 瞄准占优处回退 iter4 瞄准(K2048 fp32、K1024 16-bit 8192),'
    'log-secant 仅存 K2048 16-bit N=262144(0.996&rarr;1.14,双 dtype,'
    '不向 512K 外推),并把持续损失带剪回 stock P2(K1024@131K 全 dtype、'
    'K1024 16-bit 4K/16K/&ge;131K、K2048 16-bit 16K-131K 与 &ge;512K)。'
    '对同机 gvr_cutedsl 锚的网格几何均值:总体 1.032&rarr;1.06+;fp32 '
    '1.100&rarr;1.13(胜率 90%);16-bit 转正(bf16 0.994&rarr;1.02+、'
    'fp16 1.005&rarr;1.03+)。</p>')


def rewrite_op26_raw_csv():
    """Refresh the raw side-file the op26 note links to, so it matches the
    iter5 root now feeding the op26_1cta rows (u26.write_csvs is not run by
    the op27-canonical updater; op26_mc stays on its iter4 root)."""
    o26a_raw = u26.load(u26.OP26A_ROOT, {u26.O26A})
    base_26a = u26.load(u26.OP26A_ROOT, {u26.BASE})
    o26m_raw = u26.load(u26.OP26B_ROOT, {u26.O26M})
    mc_26b = u26.load(u26.OP26B_ROOT, {u26.MC})
    assert o26a_raw and base_26a and o26m_raw and mc_26b
    al = {(r["o"],) + u26.key(r): r for r in base_26a + mc_26b}
    anchor_of = {u26.O26A: u26.BASE, u26.O26M: u26.MC}
    head = ["scenario", "sweep", "K", "dtype", "N", "BS", "op",
            "op26_cold_us_local", "op26_warm_us_local", "anchor_op",
            "anchor_cold_us_local", "anchor_warm_us_local",
            "speedup_same_node_cold"]
    out = [head]
    for r in sorted(o26a_raw + o26m_raw, key=lambda r: (r["o"],) + u26.key(r)):
        aop = anchor_of[r["o"]]
        b = al.get((aop,) + u26.key(r))
        out.append([r["s"], r["w"], r["K"], r["d"], r["N"], r["B"],
                    r["o"], r["c"], r["h"], aop,
                    b["c"] if b else "", b["h"] if b else "",
                    round(b["c"] / r["c"], 4) if b else ""])
    with open(HERE / "op22rr_op26_raw.csv", "w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"wrote op22rr_op26_raw.csv ({len(out) - 1} rows, "
          f"op26_1cta root={u26.OP26A_ROOT.name})")


def main():
    u27.main()
    rewrite_op26_raw_csv()
    t = u27.REPORT.read_text()
    if OP26_ITER5_MARK in t:
        print("iter5 note already present; skip")
        return
    assert t.count(EN_TAIL) == 1, "op26 note EN tail not unique"
    assert t.count(ZH_TAIL) == 1, "op26 note ZH tail not unique"
    t = t.replace(EN_TAIL, EN_TAIL + EN_ITER5, 1)
    t = t.replace(ZH_TAIL, ZH_TAIL + ZH_ITER5, 1)
    u27.REPORT.write_text(t)
    print("iter5 note appended")


if __name__ == "__main__":
    main()
