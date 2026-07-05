#!/usr/bin/env python3
"""op21 iter0.5 — host prototype: P1 order-statistic M-threshold seeding.

For each bundle (real pro/flash/v32 + synth K1024), gather prev-K values via
preIdx, sort desc -> g[0..Kg-1], and evaluate count_ge(row, g[round(f*K)-1])
over a grid of rank fractions f. From the c(f) curves, score M=2 / M=4
threshold placements: straddle rate (some adjacent pair c_hi < K <= c_lo),
band size (c_lo - c_hi at the straddling pair), and miss modes.

Placement convention: fractions are of K (not Kg); the last threshold is
always g_min = g[Kg-1], which guarantees c(g_min) >= Kg (~K) since all K
gathered positions hold values >= g_min. Rows with <90% valid preIdx are
reported separately (warmup / non-convergent rows).
"""
import sys
import json
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
import synth_data  # noqa: E402
import real_data_v2  # noqa: E402

DEV = "cuda"


def count_ge(row, thr):
    return int((row >= thr).sum().item())


def curves_for_bundle(row, pre, K, kernel_read_offset=0, N=None):
    """Return (g_sorted_desc, dict f -> count_ge) or None if preIdx unusable."""
    pre = pre.to(torch.int64)
    valid = pre >= 0
    if valid.float().mean().item() < 0.9:
        return None
    idx = pre[valid]
    if kernel_read_offset:
        idx = (idx + kernel_read_offset) % N
    g = row[idx].sort(descending=True).values
    fs = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
          0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    c = {}
    for f in fs:
        r = min(max(int(round(f * K)) - 1, 0), g.numel() - 1)
        c[f] = count_ge(row, g[r])
    return g, c


def score_placement(c, fracs, K):
    """Given count curve c (dict f->count) and sorted-ascending fracs
    (thresholds descend in value as f grows? NO: larger f -> smaller g value
    -> larger count). Returns (straddle, band, mode)."""
    cnts = [c[f] for f in fracs]  # non-decreasing with f
    if cnts[0] >= K:
        return False, None, "all_ge"   # even highest thr overshoots K counts
    if cnts[-1] < K:
        return False, None, "all_lt"   # even g_min undershoots (ties/degenerate)
    for i in range(len(fracs) - 1):
        if cnts[i] < K <= cnts[i + 1]:
            return True, cnts[i + 1] - cnts[i], "ok"
    return False, None, "nonmono"


def main():
    datasets = []
    # real: pro all 30 layers (P0 priority), flash every 3rd layer, v32 all 9
    for layer in range(2, 61, 2):
        datasets.append(("pro", layer))
    for layer in range(2, 43, 6):
        datasets.append(("flash", layer))
    for layer in (0, 1, 20, 21, 22, 40, 41, 42, 60):
        datasets.append(("v32", layer))
    # synth K1024 (h=0.6 target), 3 cfgs x 3 large N (P0 regime)
    synth_specs = [(cfg, n) for cfg in ("beta_shallow", "beta_moderate", "beta_deep")
                   for n in (65536, 131072, 262144)]

    M2 = [(a, 1.00) for a in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)]
    M4 = [
        (0.50, 0.70, 0.90, 1.00),
        (0.40, 0.60, 0.80, 1.00),
        (0.30, 0.55, 0.80, 1.00),
        (0.25, 0.50, 0.75, 1.00),
        (0.20, 0.45, 0.70, 1.00),
        (0.20, 0.50, 0.80, 1.00),
        (0.30, 0.50, 0.70, 1.00),
    ]
    M6 = [
        (0.20, 0.35, 0.50, 0.65, 0.80, 1.00),
        (0.25, 0.40, 0.55, 0.70, 0.85, 1.00),
        (0.30, 0.45, 0.60, 0.75, 0.90, 1.00),
    ]

    rows = []
    for kind in ("real", "synth"):
        specs = datasets if kind == "real" else synth_specs
        for spec in specs:
            if kind == "real":
                model, layer = spec
                try:
                    b = real_data_v2.get_real_bundle_v2(model, layer, "fp32")
                except Exception as e:  # missing capture file etc.
                    print(f"skip {model} L{layer}: {e}")
                    continue
                row = b["logits"][0, : b["N"]].float()
                off = 1 if model == "v32" else 0
                tag = f"{model}-L{layer}"
                hr = b["hit_rate"]
            else:
                cfg, n = spec
                b = synth_data.get_bundle(1024, torch.float32, n, cfg=cfg)
                row = b["logits"][0, : b["N"]].float()
                off = 0
                tag = f"synth-{cfg}-N{n}"
                hr = b.get("kernel_hit_rate")
            K, N = b["K"], b["N"]
            out = curves_for_bundle(row, b["preIdx"][0], K,
                                    kernel_read_offset=off, N=N)
            if out is None:
                print(f"{tag}: preIdx <90% valid, skipped")
                continue
            _, c = out
            rows.append(dict(tag=tag, model=tag.split("-")[0], K=K, N=N,
                             hit=hr, curve={str(f): v for f, v in c.items()}))

    # ---- score placements over all rows ----
    def agg(placements, label):
        print(f"\n== {label} ==")
        print(f"{'placement':<28}{'straddle%':>10}{'band p50':>10}{'p90':>8}"
              f"{'max':>8}{'all_ge%':>9}{'all_lt%':>9}")
        best = None
        for pl in placements:
            st, bands, ge, lt = 0, [], 0, 0
            for r in rows:
                c = {float(f): v for f, v in r["curve"].items()}
                ok, band, mode = score_placement(c, list(pl), r["K"])
                if ok:
                    st += 1
                    bands.append(band / r["K"])  # band as fraction of K
                elif mode == "all_ge":
                    ge += 1
                elif mode == "all_lt":
                    lt += 1
            n = len(rows)
            bands.sort()
            p50 = bands[len(bands) // 2] if bands else float("nan")
            p90 = bands[int(len(bands) * 0.9)] if bands else float("nan")
            mx = bands[-1] if bands else float("nan")
            print(f"{str(pl):<28}{100*st/n:>9.1f}%{p50:>10.3f}{p90:>8.3f}"
                  f"{mx:>8.3f}{100*ge/n:>8.1f}%{100*lt/n:>8.1f}%")
            if best is None or st > best[0]:
                best = (st, pl)
        return best

    agg(M2, "M=2 placements (f_hi, 1.00=g_min)  [band as fraction of K]")
    agg(M4, "M=4 placements")
    agg(M6, "M=6 placements")

    # per-model straddle for the canonical M=4
    canon = (0.25, 0.50, 0.75, 1.00)
    print(f"\n== per-dataset breakdown for M4 {canon} ==")
    from collections import defaultdict
    gb = defaultdict(list)
    for r in rows:
        c = {float(f): v for f, v in r["curve"].items()}
        ok, band, mode = score_placement(c, list(canon), r["K"])
        gb[r["model"]].append((ok, band / r["K"] if band is not None else None,
                               mode, r["hit"]))
    for m, v in gb.items():
        oks = [x for x in v if x[0]]
        bands = sorted(x[1] for x in oks)
        hits = [x[3] for x in v if x[3] is not None]
        modes = [x[2] for x in v if not x[0]]
        print(f"{m:<8} n={len(v):<4} straddle={100*len(oks)/len(v):5.1f}%  "
              f"band p50={bands[len(bands)//2] if bands else float('nan'):.3f} "
              f"max={bands[-1] if bands else float('nan'):.3f}  "
              f"hit~{sum(hits)/max(len(hits),1):.2f}  miss_modes={modes[:6]}")

    Path(_HERE.parents[0], "results").mkdir(exist_ok=True)
    with open(_HERE.parents[0] / "results" / "proto_p1_orderstat.json", "w") as f:
        json.dump(rows, f)
    print(f"\nsaved {len(rows)} rows -> results/proto_p1_orderstat.json")


if __name__ == "__main__":
    main()
