# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op38 v3 probe: full (TB,CS,MAXV,AR,HS) ladder on every v2-losing (cell,BS)
case, plus paired LOCAL pr head timing (anchor-drift check vs report pr).

Answers the decisive question: is each loss fixable by a better variant, or an
arm ceiling (best variant still < pr)?  Output: v3_probe_s<i>.csv with one row
per case: prod time, best variant + time, local pr, report pr.

  python3 probe_v3.py --shard i/m --tag v3
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
KF = BENCH / "op26_r0_upstream_port_report" / "kf_campaign"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KF / "gvrpkg_04a0"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import build, bundle, make_batch, exact_rows, timeit  # noqa: E402
from probe_cfg import rep_pr, VARIANTS  # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402


def losing_cases():
    """[(model, isl, L, [bs...])] from v2_data.csv, speedup < 1.0, BS > 1."""
    cases = {}
    for r in csv.DictReader(open(HERE / "v2_data.csv")):
        if not r["speedup"] or int(r["BS"]) == 1:
            continue
        if float(r["speedup"]) >= 1.0:
            continue
        model, isl, L = r["cell"].rsplit("_", 1)[0].split("_")[0], \
            r["cell"].split("_")[1], int(r["cell"].rsplit("_L", 1)[1])
        cases.setdefault((model, isl, L), []).append(int(r["BS"]))
    return [(m, i, L, sorted(bs)) for (m, i, L), bs in sorted(cases.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()
    i, m = (int(x) for x in args.shard.split("/"))
    cells = losing_cases()[i::m]
    print(f"[probe_v3] shard {args.shard}: {len(cells)} cells", flush=True)

    mod = build("kernel_bs")
    pr = rep_pr()
    out_rows = []
    for model, isl, L, bss in cells:
        b = bundle(model, isl, L)
        K, N, Npad, cr = b["K"], b["N"], b["Npad"], b["cr"]
        v4c = (Npad + 3) // 4
        cname = f"{model}_{isl}_L{L:02d}"
        for bs in bss:
            lg, pre = make_batch(b, bs)
            out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
            # prod dispatch
            mod.run(lg, pre, N, out)
            torch.cuda.synchronize()
            assert not exact_rows(b, out, bs)
            for _ in range(5):
                mod.run(lg, pre, N, out)
            torch.cuda.synchronize()
            t_prod = timeit(lambda: mod.run(lg, pre, N, out), reps=11)
            # local pr head, paired
            sl = torch.full((bs,), N * cr, dtype=torch.int32, device="cuda")
            out_p = torch.empty(bs, K, dtype=torch.int32, device="cuda")
            GvrTopKKernel.launch(lg, pre, sl, out_p, K, compress_ratio=cr)
            torch.cuda.synchronize()
            for _ in range(5):
                GvrTopKKernel.launch(lg, pre, sl, out_p, K, compress_ratio=cr)
            torch.cuda.synchronize()
            t_prl = timeit(lambda: GvrTopKKernel.launch(
                lg, pre, sl, out_p, K, compress_ratio=cr), reps=11)
            # variant ladder
            best_us, best_cfg = t_prod, "prod"
            for tb, cs, mv, ar, hs in VARIANTS:
                vpc = (v4c + cs - 1) // cs
                if mv and vpc > mv * tb:
                    continue
                try:
                    mod.run_cfg(lg, pre, N, out, tb, cs, mv, ar, hs)
                    torch.cuda.synchronize()
                except RuntimeError:
                    torch.cuda.synchronize()
                    continue
                if exact_rows(b, out, bs):
                    continue
                for _ in range(3):
                    mod.run_cfg(lg, pre, N, out, tb, cs, mv, ar, hs)
                torch.cuda.synchronize()
                us = timeit(lambda: mod.run_cfg(lg, pre, N, out,
                                                tb, cs, mv, ar, hs), reps=11)
                if us < best_us:
                    best_us, best_cfg = us, f"{tb},{cs},{mv},{ar},{hs}"
            t_rep = pr.get((model, isl, L, bs)) or 0.0
            out_rows.append(dict(
                cell=cname, Npad=Npad, BS=bs, prod_us=round(t_prod, 2),
                best_cfg=best_cfg, best_us=round(best_us, 2),
                pr_local=round(t_prl, 2), pr_report=round(t_rep, 2),
                x_best_vs_replocal=round(t_prl / best_us, 3),
                x_best_vs_report=round(t_rep / best_us, 3) if t_rep else None,
                anchor_drift=round(t_prl / t_rep, 3) if t_rep else None))
            r = out_rows[-1]
            print(f"{cname} BS{bs:5d} prod={t_prod:8.2f} best[{best_cfg}]="
                  f"{best_us:8.2f} prL={t_prl:8.2f} prR={t_rep:8.2f} "
                  f"xR={r['x_best_vs_report']} drift={r['anchor_drift']}",
                  flush=True)
            del lg, pre, out, out_p, sl
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()

    with open(HERE / f"{args.tag}_probe_s{i}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[probe_v3] shard {i} done: {len(out_rows)} cases", flush=True)


if __name__ == "__main__":
    main()
