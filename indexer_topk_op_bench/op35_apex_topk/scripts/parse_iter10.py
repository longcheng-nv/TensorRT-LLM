# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse iter10 nsys sqlite: per-cell median spans for pure-read vs apex."""
import sqlite3
import statistics as st
import subprocess
import sys

rep = sys.argv[1] if len(sys.argv) > 1 else "/tmp/op35_iter10"
subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true",
                "-o", rep + ".sqlite", rep + ".nsys-rep"], check=True,
               capture_output=True)
db = sqlite3.connect(rep + ".sqlite")
rows = db.execute(
    "SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON k.shortName = s.id ORDER BY k.start").fetchall()
CASES = [(1, 131072), (1, 262144), (1, 1048576), (32, 262144), (256, 262144), (1024, 65536)]
FRONTIER = {(1, 131072): 11.6, (1, 262144): 15.7, (1, 1048576): 20.6,
            (32, 262144): 24.0, (256, 262144): 105.7, (1024, 65536): 93.0}
# stream: 1 warm + 30; apex: 1 warm + 30, per cell, in order
ks = [(n, (e - s) / 1e3) for n, s, e in rows
      if "stream_reduce" in n or "apex" in n]
i = 0
print(f"{'cell':>16} {'read':>8} {'apex':>8} {'tax':>6} {'frontier':>9} {'F/apex':>7}")
for BS, N in CASES:
    grp = {"k_stream_reduce": [], "k_apex_topk": []}
    while i < len(ks) and len(grp["k_apex_topk"]) < 31:
        name = "k_apex_topk" if "apex" in ks[i][0] else "k_stream_reduce"
        grp[name].append(ks[i][1])
        i += 1
    r = st.median(grp["k_stream_reduce"][1:])
    ap = st.median(grp["k_apex_topk"][1:])
    f = FRONTIER[(BS, N)]
    print(f"BS{BS:<5} N{N:<8} {r:8.2f} {ap:8.2f} {ap / r:6.2f} {f:9.1f} {f / ap:7.2f}")
