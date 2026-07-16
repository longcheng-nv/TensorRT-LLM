# Parse the vseed A/B nsys rep -> per-cell 3-arm table (cold-L2 nvtx_kern_sum).
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/report")
from parse_nsys_full import parse_rep  # noqa: E402

d = Path("/tmp/gvrval1/vseed_ab")
us = parse_rep(next(d.glob("*.nsys-rep")))
cells = [json.loads(x) for x in (d / "cells.jsonl").read_text().splitlines()]

rows = {}
for c in cells:
    key = (c["model"], c["isl"], c["dtype"], c["BS"])
    rows.setdefault(key, {"N": c["N"], "hit": c["hit"]})[c["arm"]] = (
        us.get("c|" + c["tag"]), us.get("w|" + c["tag"]), c["exact"])

print(f"{'cell':>26} {'N':>7} {'hit':>5} | {'base':>7} {'pr':>7} {'vseed':>7} | "
      f"{'pr/b':>5} {'vs/b':>5} {'vs/pr':>6} | exact(b/p/v)")
gm_reg, gm_guard = [], []
for key in sorted(rows, key=lambda k: (k[0], k[3], k[1], k[2])):
    r = rows[key]
    b, p, v = (r.get(a, (None,) * 3) for a in ("base", "pr", "vseed"))
    if not (b[0] and p[0] and v[0]):
        continue
    tag = f"{key[0]}/{key[1]}/{key[2]}/BS{key[3]}"
    ex = f"{'T' if b[2] else 'F'}/{'T' if p[2] else 'F'}/{'T' if v[2] else 'F'}"
    print(f"{tag:>26} {r['N']:>7} {r['hit']:.2f} | {b[0]:7.1f} {p[0]:7.1f} {v[0]:7.1f} | "
          f"{b[0]/p[0]:5.2f} {b[0]/v[0]:5.2f} {p[0]/v[0]:6.2f} | {ex}")
