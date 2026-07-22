# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R4 baselines.jsonl from the parity04a0 nsys run (pinned head 04a0900ff7).

Platform gap D1 (baseline evaluator does not stage campaign assets) blocks
--baseline-solution; per campaign-1 precedent we feed local nsys cold-L2
kernel-time medians as the internal denominator. Scale note lives in prompt.md.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

kern = parse_rep(str(HERE / "nsys_reps" / "ab_parity04a0.nsys-rep"))
rows = {}
for rng, us in kern.items():
    mode, arm, uuid = rng.split("|", 2)
    if mode == "c" and arm == "gvr_pr":
        rows[uuid] = us

order = [json.loads(l)["uuid"] for l in open(HERE / "ws" / "workload.jsonl")]
assert set(order) == set(rows), (set(order) ^ set(rows))
out = HERE / "gvr-topk-cold60" / "baselines.jsonl"
with open(out, "w") as f:
    for u in order:
        f.write(json.dumps({"uuid": u, "execution_time_ms": round(rows[u] / 1000, 6)}) + "\n")
print(f"wrote {out} ({len(order)} rows)")
