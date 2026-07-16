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
# stream: 1 warm + 30; apex: 1 warm + 30 GROUPS (fused=1 kernel, split=3), in order
ks = [(n, s, e) for n, s, e in rows if "stream_reduce" in n or "apex" in n]
i = 0
print(f"{'cell':>16} {'read':>8} {'apex':>8} {'tax':>6} {'frontier':>9} {'F/apex':>7}")
for BS, N in CASES:
    reads, groups = [], []
    cur = None
    while i < len(ks) and len(groups) < 31:
        n, s0, e0 = ks[i]
        if "stream_reduce" in n:
            reads.append((e0 - s0) / 1e3)
        elif "thr" in n or "fused" in n:  # group start
            if cur:
                groups.append(cur)
            cur = [s0, e0]
            if "fused" in n:
                groups.append(cur)
                cur = None
        else:  # filter / tail continue the group
            cur[1] = e0
            if "tail" in n:
                groups.append(cur)
                cur = None
        i += 1
    spans = [(e0 - s0) / 1e3 for s0, e0 in groups]
    r = st.median(reads[1:])
    ap = st.median(spans[1:])
    f = FRONTIER[(BS, N)]
    print(f"BS{BS:<5} N{N:<8} {r:8.2f} {ap:8.2f} {ap / r:6.2f} {f:9.1f} {f / ap:7.2f}")
