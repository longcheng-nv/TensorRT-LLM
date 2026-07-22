# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op38 (CS, MAXV, AR) ladder probe on the high-BS battleground cells.

For each reg-tier cell x BS, times every legal variant (vpc <= maxv*512) with
CUDA events (L2 zero'd per rep) and prints the winner. Exactness checked per
variant. Report pr (bs_real_layers.csv) shown as the target line."""
import csv
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "harness"))
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402
from probe import build, bundle, make_batch, exact_rows, timeit  # noqa: E402

CELLS = [  # reg-tier battleground (npad > 12288)
    ("v32", "16k", 34),      # npad 16448
    ("flash", "128k", 22),   # npad 32832
    ("pro", "128k", 30),
    ("flash", "256k", 22),   # npad 65600
    ("v32", "128k", 34),     # npad 131136
    ("flash", "512k", 22),   # npad 131136
    ("v32", "256k", 34),     # npad 163840
    ("flash", "1024k", 22),  # npad 262144
    ("pro", "1024k", 30),
]
VARIANTS = [  # (tb, cs, maxv, ar, hs) — HS=1 winners + HS/AR variations
    (512, 1, 9, 8, 1), (512, 1, 9, 8, 2), (512, 1, 9, 8, 4), (512, 1, 9, 4, 2),
    (1024, 1, 9, 8, 1), (1024, 1, 9, 8, 2), (1024, 1, 9, 8, 4),
    (1024, 1, 9, 6, 1), (1024, 1, 9, 6, 2), (1024, 1, 9, 4, 1), (1024, 1, 9, 4, 2),
    (1024, 2, 9, 8, 1), (1024, 2, 9, 8, 2), (1024, 2, 9, 8, 4), (1024, 2, 9, 6, 2),
    (1024, 4, 9, 8, 1), (1024, 4, 9, 8, 2), (1024, 4, 9, 8, 4),
    (1024, 4, 9, 6, 1), (1024, 4, 9, 6, 2),
    (1024, 8, 8, 6, 1), (1024, 8, 8, 6, 2), (1024, 8, 8, 6, 4), (1024, 8, 8, 4, 2),
    (512, 8, 10, 6, 1), (512, 8, 10, 6, 2), (512, 8, 10, 6, 4),
    (512, 16, 8, 6, 1), (512, 16, 8, 6, 2),
    (512, 1, 0, 8, 1), (512, 2, 0, 8, 1), (512, 4, 0, 8, 1),
    (1024, 1, 0, 8, 1), (1024, 2, 0, 8, 1), (1024, 4, 0, 8, 1),
    (1024, 1, 0, 8, 2), (1024, 1, 0, 8, 4), (1024, 1, 0, 6, 2), (1024, 1, 0, 6, 4),
    (1024, 1, 0, 4, 2), (1024, 1, 0, 4, 4), (1024, 1, 0, 6, 1), (1024, 1, 0, 4, 1),
    (1024, 2, 0, 8, 2), (1024, 2, 0, 6, 2), (1024, 2, 0, 4, 2), (1024, 2, 0, 6, 4),
    (512, 1, 0, 8, 2), (512, 1, 0, 6, 2), (512, 1, 0, 4, 2), (512, 1, 0, 6, 4)]
BS_LIST = [64, 256, 1024]


def rep_pr():
    d = {}
    for r in csv.DictReader(open(BENCH / "op26_r0_upstream_port_report" /
                                 "bs_real_layers.csv")):
        d[(r["model"], r["isl"], int(r["L"]), int(r["BS"]))] = float(r["pr"])
    return d


def main():
    mod = build("kernel_bs")
    pr = rep_pr()
    print("[probe_cfg] built", flush=True)
    for model, isl, L in CELLS:
        b = bundle(model, isl, L)
        K, N, Npad = b["K"], b["N"], b["Npad"]
        v4c = (Npad + 3) // 4
        for bs in BS_LIST:
            lg, pre = make_batch(b, bs)
            out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
            target = pr.get((model, isl, L, bs))
            results = []
            for tb, cs, mv, ar, hs in VARIANTS:
                vpc = (v4c + cs - 1) // cs
                if mv and vpc > mv * tb:
                    continue
                try:
                    mod.run_cfg(lg, pre, N, out, tb, cs, mv, ar, hs)
                    torch.cuda.synchronize()
                except RuntimeError as e:
                    print(f"  cfg({tb},{cs},{mv},{ar},{hs}) LAUNCH-FAIL {e}", flush=True)
                    torch.cuda.synchronize()
                    continue
                bad = exact_rows(b, out, bs)
                if bad:
                    print(f"  cfg({tb},{cs},{mv},{ar},{hs}) INEXACT {bad}", flush=True)
                    continue
                for _ in range(5):
                    mod.run_cfg(lg, pre, N, out, tb, cs, mv, ar, hs)
                torch.cuda.synchronize()
                us = timeit(lambda: mod.run_cfg(lg, pre, N, out, tb, cs, mv, ar, hs),
                            reps=11)
                results.append((us, (tb, cs), (mv, ar), hs))
            results.sort()
            best = results[0]
            line = " ".join(f"({c},{m},{a})={u:.1f}" for u, c, m, a in results[:4])
            print(f"{model}_{isl}_L{L:02d} Npad={Npad:6d} BS{bs:5d} "
                  f"rep_pr={target or 0:8.2f} best=({best[1]},{best[2]},{best[3]})"
                  f" {best[0]:8.2f}us x{(target or 0) / best[0]:6.3f} | {line}",
                  flush=True)
            del lg, pre, out
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
