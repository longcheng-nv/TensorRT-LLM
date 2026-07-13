# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dry-run update_report_realcap.py against synthetic rows on a REPORT.html
COPY under /tmp (never touches the real report). Verifies chapter build,
injection, idempotent re-run, and reports the size delta."""
import importlib
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_report_realcap as U
importlib.reload(U)

tmp = Path("/tmp/realcap_dryrun")
tmp.mkdir(exist_ok=True)
shutil.copy(Path(__file__).parent / "REPORT.html", tmp / "REPORT.html")

rows = []
random.seed(0)
layers = {"flash": list(range(2, 43, 2)), "pro": list(range(2, 61, 2)),
          "v32": [0, 1, 20, 21, 22, 40, 41, 42, 60]}
Ns = {"flash": (25154, 25216, 512, 4, 424),
      "pro": (14478, 14528, 1024, 4, 304),
      "v32": (70690, 70720, 2048, 1, 2024)}
for m, ls in layers.items():
    N, Npad, K, cr, s = Ns[m]
    for dt in ("fp32", "bf16", "fp16"):
        for arm in U.ARMS:
            if arm in U.__dict__.get("FP32_ONLY_ARMS", (
                    "sglang_streaming", "sglang_v2", "flashinfer_topk")) \
                    and dt != "fp32":
                continue
            if arm == "sglang_streaming" and K > 1024:
                continue
            for L in ls:
                for BS in U.BS_GRID:
                    base = 20 * BS ** 0.8 * (1 + random.random() * 0.3)
                    r = {"sweep": "realcap", "op": arm, "model": m, "K": K,
                         "dtype": dt, "N": N, "Npad": Npad, "BS": BS,
                         "cr": cr, "layer": L, "s_last": s,
                         "hit_rate": round(random.random(), 4),
                         "us_cold": base * (0.5 + random.random()),
                         "us_warm": base * (0.4 + random.random() * 0.8),
                         "us": 1}
                    if BS == 1:
                        r.update(vdiff=0.0, recall=1.0, n_neg=0)
                    rows.append(r)
(tmp / "realcap_sweep").mkdir(exist_ok=True)
with open(tmp / "realcap_sweep" / "results.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

U.OUT_ROOT = tmp
U.RESULTS = tmp / "realcap_sweep" / "results.jsonl"
U.REPORT = tmp / "REPORT.html"
U.BAK = tmp / "REPORT.bak"
U.HERE = tmp
size0 = (tmp / "REPORT.html").stat().st_size
U.main()
U.main()   # idempotency
t = (tmp / "REPORT.html").read_text()
n_begin, n_end = t.count(U.BEGIN), t.count(U.END)
print(f"chapters={n_begin}/{n_end} size {size0} -> {len(t)} "
      f"(+{len(t) - size0})")
assert n_begin == 1 and n_end == 1
print("DRYRUN OK")
