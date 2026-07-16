# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse the loser re-verdict reps (/tmp/op35_nsys/reab_s{0,1}.nsys-rep)."""
import statistics as st
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "report"))
from parse_nsys_full import parse_rep
per = {}
for sh in (0, 1):
    rep = Path(f"/tmp/op35_nsys/reab_s{sh}.nsys-rep")
    if rep.exists():
        for rng, us in parse_rep(rep).items():
            p = rng.split("|")
            if len(p) == 4 and p[0] == "c":
                per.setdefault((p[1], p[2]), {})[int(p[3][1:])] = us
for c in sorted({c for c, _ in per}):
    b, v = per.get((c, "base"), {}), per.get((c, "var"), {})
    rounds = sorted(set(b) & set(v))
    if rounds:
        rs = [b[r] / v[r] for r in rounds]
        print(f"{c:34s} med={st.median(rs):.4f} rounds={[round(x,3) for x in rs]} "
              f"base={st.median(b[r] for r in rounds):.2f}us")
