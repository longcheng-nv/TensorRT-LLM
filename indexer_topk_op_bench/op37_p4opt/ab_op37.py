# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op37 paired nsys A/B: gvrpkg37 flag arms vs base on real decode cells.

Arms (same compiled family, ctor flags only):
  base / d2a (rw search) / d2b (fine skip) / d2ab / d1a (peer push) / all
Per cell x arm: 10 warmup + 20 cold-L2 NVTX'd launches, back-to-back on ONE
GPU (ratio purity). Run under nsys (drive_ab_op37.sh, <=2 concurrent).
  python3 ab_op37.py --shard i/m --tag <tag>
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPORT = BENCH / "op26_r0_upstream_port_report"

sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE / "variant"))

import cutlass  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from validate_op37 import compile_arm, launch_cfg, exact_set  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
WARMUP, REPS = 10, 20
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

ARMS = [
    ("base", {}),
    ("d2a", dict(p4_rs_rw_search=True)),
    ("d2b", dict(p4_fine_skip=True)),
    ("d2ab", dict(p4_rs_rw_search=True, p4_fine_skip=True)),
    ("d1a", dict(p4_peer_push=True)),
    ("all", dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True)),
]

# one representative layer per (model, ISL) + the tail-heavy cell
CELLS = ([("flash", i, 2) for i in
          ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]]
         + [("flash", "128k", 42)]
         + [("pro", i, 30) for i in
            ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]]
         + [("v32", i, 34) for i in
            ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None)
    ap.add_argument("--tag", default="ab")
    args = ap.parse_args()
    cells = CELLS
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        cells = cells[i::m]
    print(f"[ab37] {len(cells)} cells x {len(ARMS)} arms", flush=True)

    rows = []
    prof.start()
    for model, isl, layer in cells:
        uuid = f"{model}_{isl}_L{layer:02d}"
        try:
            RD = RV32 if model == "v32" else RV4
            b = RD.get_bundle(model, isl, layer, "fp32")
            logits = b["logits"].contiguous()
            pre = b["preIdx"].contiguous()
            N, K, cr = b["N"], b["K"], b["cr"]
            sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
            cfg = launch_cfg(logits, N)
            for arm, flags in ARMS:
                fn = compile_arm(K, cr, cfg, flags)
                oi = torch.empty(1, K, dtype=torch.int32, device=DEV)
                fn(logits, pre, sl, None, oi, None)
                torch.cuda.synchronize()
                ok = exact_set(oi, logits[0], K, N)
                for _ in range(WARMUP):
                    fn(logits, pre, sl, None, oi, None)
                torch.cuda.synchronize()
                for _ in range(REPS):
                    _EVICT.random_()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_push(f"c|{arm}|{uuid}")
                    fn(logits, pre, sl, None, oi, None)
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                rows.append(dict(uuid=uuid, arm=arm, K=K, N=N,
                                 cs=cfg["cluster_size"], exact=ok))
                print(f"  {uuid} {arm} exact={ok}", flush=True)
            del b
            RV4._bundle_cache.clear()
            RV32._bundle_cache.clear()
        except Exception as e:
            rows.append(dict(uuid=uuid, arm="ERROR", K=0, N=0, cs=0,
                             exact=False))
            print(f"[ab37] ERROR {uuid}: {e!r}", flush=True)
    prof.stop()

    with open(HERE / f"ab37_{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ab37] done; inexact: "
          f"{[r['uuid']+'/'+r['arm'] for r in rows if not r['exact']] or 'none'}",
          flush=True)


if __name__ == "__main__":
    main()
