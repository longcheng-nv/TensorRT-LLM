# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 iter6 — host screen of R0 h-space qfrac ladders on op22rr bundles.

For each (scenario, K, dtype, N) bundle row, place M rung thresholds at
h-space quantiles of the prev-topK gathered values (kernel-faithful:
256-bin histogram over [pmin, pmax], rung = highest value v with
#(prev-topK >= v) >= ceil(q*K), taken at bin edge), count_ge each rung on
the full row (fp32 compare, matching the kernel's post-cast compare), and
check static admission into the CLASSIC window [K, kC].

Also screens the two adversarial poles from gate_op26 Suite B (hr=0
disjoint preIdx, hr=1 exact top-K preIdx) and records the baseline pmean
seed's first-pass admission for reference.

Output: screen_r0_qfracs.csv (one line per cell x ladder) + stdout summary
(per-ladder static accept rate by scenario/K/dtype + accepted-count size
stats for slot sizing + miss bracket quality).

Usage: python3 screen_r0_qfracs.py [--device cuda]
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))
import bundle_data_rr  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() and "--cpu" not in sys.argv else "cpu"

# classic stock candidate caps (vendored GvrParams; fp32 K2048 = 6144)
KC = {(512, "fp32"): 5120, (1024, "fp32"): 5120, (2048, "fp32"): 6144,
      (512, "bf16"): 5120, (1024, "bf16"): 5120, (2048, "bf16"): 5120,
      (512, "fp16"): 5120, (1024, "fp16"): 5120, (2048, "fp16"): 5120}

# candidate ladders (h-space fractions into the prev-topK value CCDF,
# deepest-first). "pm" = append the vendored pmean seed as an extra rung.
LADDERS = {
    "w3a3":      (0.92, 0.45, 0.048),
    "w3a4":      (0.92, 0.75, 0.45, 0.048),
    "w27_3":     (0.75, 0.45, 0.048),
    "base3":     (0.75, 0.50, 0.25),
    "uh4":       (0.90, 0.65, 0.40, 0.15),
    "w3a3_pm":   (0.92, 0.45, 0.048, "pmean"),
    "deep4":     (0.98, 0.92, 0.45, 0.048),
}

SCENARIOS = ("real", "best", "worst")
KS = (512, 1024, 2048)
DTS = ("fp32", "bf16", "fp16")
NS = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
TDT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def hist_quantile_rungs(vals_f32, qfracs, K, pmean):
    """Kernel-faithful rung extraction: 256-bin hist over [pmin,pmax] of the
    prev-topK values; rung_m = lower edge of the bin where the descending
    (from pmax) cumulative count first reaches ceil(q*K)."""
    pmin = vals_f32.min()
    pmax = vals_f32.max()
    if (pmax - pmin) <= 0:
        return [float(pmin)] * len([q for q in qfracs if q != "pmean"]) + \
               ([float(pmean)] if "pmean" in qfracs else [])
    nb = 256
    width = (pmax - pmin) / nb
    b = torch.clamp(((vals_f32 - pmin) / width).long(), 0, nb - 1)
    hist = torch.bincount(b, minlength=nb)
    # descending cumulative: cum_desc[i] = #(vals in bins >= i)
    cum_desc = torch.flip(torch.cumsum(torch.flip(hist, (0,)), 0), (0,))
    rungs = []
    for q in qfracs:
        if q == "pmean":
            rungs.append(float(pmean))
            continue
        need = max(1, math.ceil(q * K))
        # highest bin i with cum_desc[i] >= need -> threshold at bin lower edge
        ok = (cum_desc >= need).nonzero()
        i = int(ok.max()) if len(ok) else 0
        rungs.append(float(pmin + i * width))
    return rungs


def eval_row(row_f32, valid_n, prev_vals, K, kC, pmean, tag, writer, stats):
    for lname, qf in LADDERS.items():
        rungs = hist_quantile_rungs(prev_vals, qf, K, pmean)
        counts = [int((row_f32[:valid_n] >= t).sum()) for t in rungs]
        # admissible rungs; prefer the smallest admissible count (cheapest P3/P4)
        adm = [c for c in counts if K <= c <= kC]
        accept = bool(adm)
        best_c = min(adm) if adm else None
        # miss bracket quality: does some adjacent pair straddle the window?
        srt = sorted(counts)
        straddle = any(srt[i] < K and srt[i + 1] > kC for i in range(len(srt) - 1))
        below = all(c < K for c in counts)     # all rungs too deep? (count<K)
        above = all(c > kC for c in counts)    # all_ge-like pole
        writer.writerow([*tag, lname, accept, best_c,
                         "|".join(str(c) for c in counts),
                         straddle, below, above])
        s = stats[lname]
        s["n"] += 1
        s["acc"] += accept
        if accept:
            s["cand"].append(best_c)
        elif above:
            s["all_above"] += 1
        elif below:
            s["all_below"] += 1
        else:
            s["bracket"] += 1   # measured straddling bracket -> 1 falsi step away
        stats[(lname, tag[0])]["n"] += 1
        stats[(lname, tag[0])]["acc"] += accept


def main():
    out = open(HERE / "screen_r0_qfracs.csv", "w", newline="")
    w = csv.writer(out)
    w.writerow(["scenario", "K", "dtype", "N", "hr", "ladder", "accept",
                "best_cand", "counts", "straddle", "all_below", "all_above"])
    stats = defaultdict(lambda: defaultdict(int))
    for ln in LADDERS:
        stats[ln]["cand"] = []
        for sc in (*SCENARIOS, "hr0", "hr1"):
            stats[(ln, sc)] = defaultdict(int)

    n_cells = 0
    base_acc = defaultdict(lambda: [0, 0])
    for sc in SCENARIOS:
        for K in KS:
            for dt in DTS:
                for N in NS:
                    try:
                        b = bundle_data_rr.get_bundle(sc, K, dt, N, device=DEV)
                    except (FileNotFoundError, KeyError, AssertionError):
                        continue
                    row = b["logits"][0]
                    row_f32 = row.float()
                    cr = b["cr"]
                    off = 1 if cr == 1 else 0
                    idx = b["preIdx"][0].long() + off
                    val_mask = (idx >= 0) & (idx < N)
                    prev_vals = row_f32[idx[val_mask]]
                    if prev_vals.numel() < 8:
                        continue
                    pmean = float(prev_vals.mean())
                    kC = KC[(K, dt)]
                    n_cells += 1
                    # baseline pmean-seed first-pass admission (reference)
                    c0 = int((row_f32[:N] >= pmean).sum())
                    ba = base_acc[sc]
                    ba[0] += (K <= c0 <= kC)
                    ba[1] += 1
                    eval_row(row_f32, N, prev_vals, K, kC, pmean,
                             (sc, K, dt, N, round(b["kernel_hit_rate"], 3)),
                             w, stats)
                    # adversarial poles on the same row (Suite-B style)
                    tk = torch.topk(row_f32[:N], K)
                    hr1_vals = tk.values                      # hr = 1
                    bot = torch.topk(row_f32[:N], K, largest=False)
                    hr0_vals = bot.values                     # hr = 0 disjoint
                    for hrname, pv in (("hr1", hr1_vals), ("hr0", hr0_vals)):
                        pm2 = float(pv.mean())
                        for lname, qf in LADDERS.items():
                            rungs = hist_quantile_rungs(pv, qf, K, pm2)
                            counts = [int((row_f32[:N] >= t).sum()) for t in rungs]
                            adm = [c for c in counts if K <= c <= kC]
                            s = stats[(lname, hrname)]
                            s["n"] += 1
                            s["acc"] += bool(adm)
    out.close()

    print(f"cells screened: {n_cells}  (x {len(LADDERS)} ladders) -> "
          f"{HERE / 'screen_r0_qfracs.csv'}")
    print("\nbaseline pmean-seed first-pass admission:",
          {sc: f"{a}/{n}" for sc, (a, n) in base_acc.items()})
    hdr = f"{'ladder':10s} {'ALL':>9s}" + "".join(
        f" {sc:>9s}" for sc in (*SCENARIOS, "hr0", "hr1"))
    print("\nstatic admission rate (any rung count in [K, kC]):")
    print(hdr)
    for ln in LADDERS:
        s = stats[ln]
        row = f"{ln:10s} {s['acc']/max(s['n'],1):9.3f}"
        for sc in (*SCENARIOS, "hr0", "hr1"):
            t = stats[(ln, sc)]
            row += f" {t['acc']/max(t['n'],1):9.3f}"
        print(row)
    print("\nmiss decomposition (of non-accepted grid cells) + accepted cand size:")
    for ln in LADDERS:
        s = stats[ln]
        miss = s["n"] - s["acc"]
        cand = sorted(s["cand"])
        med = cand[len(cand) // 2] if cand else -1
        p90 = cand[int(len(cand) * 0.9)] if cand else -1
        print(f"{ln:10s} miss={miss:3d} (bracket={s['bracket']:3d} "
              f"all_above={s['all_above']:3d} all_below={s['all_below']:3d})  "
              f"cand med={med} p90={p90}")


if __name__ == "__main__":
    main()
