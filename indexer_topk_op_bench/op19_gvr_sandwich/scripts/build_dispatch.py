# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Build the op19 per-(K, N, BS-regime) dispatch table from the config-sweep
# jsonls. Picks the best exact config per cell; falls back to "baseline" when
# nothing beats 1.0 (records the gap). Emits results/dispatch_table.json and
# prints a coverage summary.
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RES = _HERE.parent / "results"


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "speedup" in r and r.get("exact"):
                rows.append(r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonls", nargs="+")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="config must beat this speedup to be dispatched")
    args = ap.parse_args()
    rows = [r for r in load(args.jsonls) if r["dtype"] == args.dtype]
    best = {}
    for r in rows:
        key = (r["K"], r["N"], r["BS"])
        if key not in best or r["speedup"] > best[key]["speedup"]:
            best[key] = r
    table = {}
    sp = []
    n_base = 0
    print(f"{'K':>5} {'N':>7} {'BS':>5} | best cfg      speedup")
    for key in sorted(best):
        r = best[key]
        if r["speedup"] >= args.margin:
            cfg = r["cfg"]
        else:
            cfg = "baseline"
            n_base += 1
        table["_".join(map(str, key))] = dict(cfg=cfg, speedup=round(r["speedup"], 3))
        sp.append(max(r["speedup"], 1.0) if cfg == "baseline" else r["speedup"])
        print(f"{key[0]:>5} {key[1]:>7} {key[2]:>5} | {cfg:>12} {r['speedup']:>7.3f}")
    out = RES / f"dispatch_table_{args.dtype}.json"
    json.dump(table, open(out, "w"), indent=1)
    print(f"\ncells={len(sp)} gm={statistics.geometric_mean(sp):.3f} "
          f"avg={statistics.mean(sp):.3f} min={min(sp):.3f} max={max(sp):.3f} "
          f"baseline-fallbacks={n_base}")
    print(f"wrote {out}")
