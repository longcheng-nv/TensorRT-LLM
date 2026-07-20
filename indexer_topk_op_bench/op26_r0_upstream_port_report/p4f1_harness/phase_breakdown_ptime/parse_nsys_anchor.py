#!/usr/bin/env python3
"""Parse nsys_anchor sqlite: per (cell, arm) NVTX section, median GVR kernel
duration (ns -> us). Writes nsys_anchor.csv next to this script."""
import csv
import sqlite3
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
db = sys.argv[1] if len(sys.argv) > 1 else None
assert db, "usage: parse_nsys_anchor.py <sqlite>"

con = sqlite3.connect(db)
cur = con.cursor()

# NVTX section ranges
secs = cur.execute(
    "SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'SEC|%'"
).fetchall()
assert secs, "no NVTX SEC ranges found"

# GVR kernels (exclude evict zero_ / memset etc.)
kerns = cur.execute(
    """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
       JOIN StringIds s ON k.demangledName = s.id
       WHERE lower(s.value) LIKE '%gvr%'"""
).fetchall()
assert kerns, "no GVR kernels found"

rows = []
for text, s0, s1 in sorted(secs, key=lambda r: r[1]):
    _, cell, arm = text.split("|")
    durs = [(e - s) / 1e3 for s, e, _ in kerns if s >= s0 and e <= s1]
    names = {n for s, e, n in kerns if s >= s0 and e <= s1}
    med = statistics.median(durs) if durs else float("nan")
    rows.append((cell, arm, len(durs), med))
    nm = next(iter(names)) if names else "?"
    print(f"{cell:20s} {arm:6s} n={len(durs):3d} med={med:7.2f} us  [{nm[:70]}]")

with open(HERE / "nsys_anchor.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cell", "arm", "n_kernels", "med_us"])
    w.writerows(rows)
print("wrote", HERE / "nsys_anchor.csv")

# paired summary
by = {(c, a): m for c, a, _, m in rows}
print("\ncell                  prod_us  timed_us  overhead")
for cell in dict.fromkeys(c for c, _, _, _ in rows):
    p, t = by[(cell, "prod")], by[(cell, "timed")]
    print(f"{cell:20s} {p:8.2f} {t:9.2f} {100 * (t / p - 1):+8.2f}%")
