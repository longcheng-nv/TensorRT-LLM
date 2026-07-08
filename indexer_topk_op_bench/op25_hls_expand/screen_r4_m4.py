# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 round 4 — M=4 wide-triple re-screen after the silicon decomposition.

Silicon said: M=5 ladder tax +7..19% on fast rows (count-loop column is
NOT divided by C), slot_scale=2 free at N<65536 (t=512) and +12..21% at
N>=65536 (t=1024). So the ship geometry must be an M=4 triple (zero
fast-path tax vs base3) + slot_scale gated to n<65536.

Cap model here: 8192 for N<65536 rows, 4096 otherwise.
Axes/arms as screen_qfracs.py; M=4 wide candidates + references.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen_qfracs as S  # noqa: E402
import proto_hls as P  # noqa: E402

GEOMS = {
    "base3":  (0.75, 0.50, 0.25),
    "w3a":    (0.92, 0.45, 0.048),
    "w3b":    (0.92, 0.50, 0.060),
    "w3c":    (0.90, 0.42, 0.048),
    "w3d":    (0.92, 0.40, 0.048),
    "w3e":    (0.88, 0.40, 0.048),
    "w3f":    (0.92, 0.35, 0.050),
    "wide4b": (0.92, 0.60, 0.25, 0.048),   # M=5 reference (taxed)
}


def eval_arms_m4(row):
    cap = 8192 if row.N < 65536 else 4096
    out = {}
    for name, qf in GEOMS.items():
        cols, fr = P.cols_static(row, qfracs=qf)
        r0 = P.simulate_r0(row, cols, fr, cap=cap)
        r0["_cols"] = cols
        fbp = 0
        if r0["mode"] != "fast":
            fbp, _ = P.fb_logfalsi(row, r0, alpha=0.2)
        out[name] = {"mode": r0["mode"], "fbp": fbp, "m": len(qf)}
    return out


S.eval_arms = eval_arms_m4          # monkey-patch the arm evaluator


def main():
    recs = []
    outf = HERE / "results" / "screen_r4.jsonl"
    with open(outf, "w") as f:
        S.axis_a_op22rr(recs, f)
        S.axis_b_op24(recs, f)
        S.axis_c_pro(recs, f, stride=1)
    ok = [r for r in recs if "error" not in r]
    arms = list(GEOMS)

    def fast(rows, a):
        return sum(r["arms"][a]["mode"] == "fast" for r in rows) / len(rows)

    groups = []
    for scen in ("best", "worst", "real"):
        for m in ("v4flash", "v4pro", "v32"):
            groups.append((f"op22rr {scen}/{m}",
                           [r for r in ok if r["axis"] == "op22rr"
                            and r["scen"] == scen and r["model"] == m]))
    pro = [r for r in ok if r["axis"] == "pro"]
    groups += [("pro ALL", pro),
               ("pro h<0.5", [r for r in pro if r["h_true"] < 0.5]),
               ("pro h>=0.75", [r for r in pro if r["h_true"] >= 0.75])]
    for m in ("v4flash", "v4pro"):
        for hr in ("0.05", "samp"):
            rows = []
            for r in ok:
                if r["axis"] != "op24" or r["model"] != m:
                    continue
                s = r["scen"]
                if hr == "samp" and ("hrsamp" in s or "_st4" in s):
                    rows.append(r)
                elif hr != "samp" and f"hr{hr}_" in s:
                    rows.append(r)
            groups.append((f"op24 {m}/hr={hr}", rows))
    lines = [f"{'group':22s}" + "".join(f"{a:>9s}" for a in arms)]
    for g, rows in groups:
        if rows:
            lines.append(f"{g:22s}"
                         + "".join(f"{fast(rows, a):9.3f}" for a in arms)
                         + f"  n={len(rows)}")
    txt = "\n".join(lines)
    (HERE / "results" / "SCREEN_R4.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
