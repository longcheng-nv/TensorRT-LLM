# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""rung-0.2: Floyd-Rivest sampling-band validation on REAL decode captures.

For each (model, isl, layer, dtype): draw a contiguous-block sample, estimate
the K-th-value band [t_lo, t_hi] from sample order statistics +- z*sigma, then
measure on the full row: admitted counts, band size, miss rate. Also an IID
(randperm) control to quantify the spatial-autocorrelation penalty of block
sampling, and bf16/fp16 tie inflation.

Run: PYTHONNOUSERSITE=1 python3 scripts/rung0_band.py --gpu 2
Writes results/rung0_band.csv + prints summary.
"""
import argparse
import csv
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../harness"))
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=2)
a = ap.parse_args()
torch.cuda.set_device(a.gpu)

g = torch.Generator(device="cuda")

def band_trial(vals, N, K, s, z, mode, seed):
    """vals: [N] float32 cuda (full row). Returns dict of band metrics."""
    g.manual_seed(seed)
    if mode == "iid":
        idx = torch.randperm(N, device="cuda", generator=g)[:s]
    elif mode == "strat":  # stratified-jittered: 1 random elem per N/s stripe
        stride = N / s
        base = (torch.arange(s, device="cuda", dtype=torch.float64) * stride)
        jit = torch.rand(s, device="cuda", generator=g, dtype=torch.float64) * stride
        idx = (base + jit).long().clamp_(0, N - 1)
    else:  # contiguous micro-blocks: blk4 / blk32
        B = 4 if mode == "blk4" else 32
        nb = s // B
        starts = torch.randint(0, max(1, N - B), (nb,), device="cuda", generator=g)
        idx = (starts[:, None] + torch.arange(B, device="cuda")[None, :]).flatten()
        idx = idx.clamp_(0, N - 1)
    samp = vals[idx]
    ss = samp.sort(descending=True).values
    q = K / N
    r0 = s * q
    sig = math.sqrt(max(1.0, s * q * (1 - q)))
    i_hi = max(0, int(math.floor(r0 - z * sig)) - 1)
    i_lo = min(s - 1, int(math.ceil(r0 + z * sig)) - 1)
    t_hi = ss[i_hi].item()
    t_lo = ss[i_lo].item()
    n_hi = int((vals >= t_hi).sum())
    n_lo = int((vals >= t_lo).sum())
    admit = n_hi if n_hi >= K else n_lo          # tightest rung with >= K
    miss = n_lo < K
    return dict(n_hi=n_hi, n_lo=n_lo, admit=admit, miss=int(miss),
                band=n_lo - n_hi)

rows = []
def run_model(tag, get_bundle, isls, layers_probe):
    for isl in isls:
        for L in layers_probe:
            try:
                b = get_bundle(tag if tag != "v32" else "v32", isl, L, "fp32")
            except Exception as ex:
                continue
            N, K = b["N"], b["K"]
            base = b["logits"][0, :N].float()
            for dt in ("fp32", "bf16"):
                vals = base if dt == "fp32" else base.to(torch.bfloat16).float()
                for s in (1024, 2048, 4096):
                    if s >= N:
                        continue
                    for z, mode in ((3.0, "strat"), (3.0, "iid")):
                        ms = [band_trial(vals, N, K, s, z, mode, seed)
                              for seed in range(8)]
                        rows.append(dict(model=tag, isl=isl, layer=L, dtype=dt,
                                         N=N, K=K, s=s, z=z, mode=mode,
                                         admit=sum(m["admit"] for m in ms) / 8 / K,
                                         band=sum(m["band"] for m in ms) / 8 / K,
                                         miss=sum(m["miss"] for m in ms)))
            print(f"{tag} {isl} L{L} N={N} done", flush=True)

# layer ids: probe the slim dicts
def layers_of(slim_mod, model, isl):
    try:
        s = slim_mod._slim(model, isl) if slim_mod is v4 else slim_mod._slim(isl)
        return sorted(s["cur"].keys())[:3] if hasattr(s["cur"], "keys") else list(range(len(s["cur"])))[:3]
    except Exception:
        return []

for isl in list(v4.ISL_DIR.keys()):
    for model in ("flash", "pro"):
        Ls = layers_of(v4, model, isl)
        run_model(model, v4.get_bundle, [isl], Ls)
for isl in list(v32.ISL_DIR.keys()):
    Ls = layers_of(v32, "v32", isl)
    run_model("v32", v32.get_bundle, [isl], Ls)

os.makedirs(os.path.join(HERE, "../results"), exist_ok=True)
out = os.path.join(HERE, "../results/rung0_band.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# summary
import statistics as st
print("\n==== SUMMARY (admit/K multiples; miss counts over 8 seeds/cell) ====")
for mode in ("strat", "iid"):
    for s in (1024, 2048, 4096):
        sel = [r for r in rows if r["mode"] == mode and r["s"] == s and float(r["z"]) == 3.0]
        if not sel:
            continue
        adm = [r["admit"] for r in sel]
        miss = sum(r["miss"] for r in sel)
        print(f"mode={mode} s={s} z=3: admit/K med {st.median(adm):.2f} "
              f"p95 {sorted(adm)[int(.95 * len(adm))]:.2f} max {max(adm):.2f} "
              f"miss {miss}/{len(sel) * 8}")
print("wrote", out)
