# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op34 Phase-1 data characterization of the V4 decode-capture indexer logits.

Answers three questions that drive every GVR threshold-method assumption:

  A. TEMPORAL hit-rate across ALL decode steps (not just the last):
       hr(s) = |topk.out[s] ∩ topk.out[s-1]| / K   (the exact GVR preIdx hint).
     GVR bets the prev-step top-K is a near-superset of the current top-K, so a
     threshold seeded from the hint converges in ~1 pass. If hr is low or noisy
     the hint is worthless and GVR degenerates to a full scan (+ threshold tax).

  B. BOUNDARY difficulty of the LAST-step logits (per model,ISL,layer):
       gap    = v[K-1] - v[K]          (separation of the K-th from the (K+1)-th)
       band_r = #{v : v[K] < v <= v[K-1]+eps} ...  density of near-boundary values
     A tiny gap / dense boundary band => a threshold has to be placed with high
     precision => secant needs more refine passes => GVR's per-pass cost grows.
     This is the mechanism behind op26_r0auto's 2.7e-6 boundary-precision need.

  C. VALUE distribution shape: per-cell mean/std/percentiles + the fraction of
     the row above the K-th value (selectivity = K/N, trivially, but we also
     report how "peaked" the top-K mass is: v[K-1] vs max).

Pass A + partial C read only decode.topk.out.pt (cheap, all 16 steps).
Pass B/C read the cached slim last-step logits (data_v4cap/).

Output: analysis/realdata_props.json  (+ printed tables). No timing here.
"""
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_OPBENCH = _HERE.parents[1]
sys.path.insert(0, str(_OPBENCH / "harness"))
import real_data_v4cap as RD4  # noqa: E402

CAP = RD4.CAP_ROOT
OUT = _HERE / "realdata_props.json"


def per_step_hitrate(model, isl):
    """Return {layer: [hr(1)..hr(S-1)]} where hr(s)=|topk[s]∩topk[s-1]|/K,
    plus the last-step hint hr already in the bundle. Reads topk.out only."""
    m = RD4.MODELS[model]
    K = m["K"]
    out = {}
    for L in m["layers"]:
        d = RD4._layer_dir(model, isl, L) / "decode.topk.out.pt"
        if not d.exists():
            return None
        pk = torch.load(d, map_location="cpu", weights_only=False)
        steps = sorted(pk.keys())
        hrs = []
        for i in range(1, len(steps)):
            cur = set(pk[steps[i]].flatten().tolist())
            prev = set(pk[steps[i - 1]].flatten().tolist())
            hrs.append(len(cur & prev) / K)
        out[L] = hrs
    return out


def boundary_props(model, isl, layer):
    """Last-step boundary difficulty + value-distribution shape from slim."""
    b = RD4.get_bundle(model, isl, layer, "fp32")
    lg = b["logits"][0, :b["N"]].float()
    K, N = b["K"], b["N"]
    sv = lg.sort(descending=True).values
    vK1 = sv[K - 1].item()          # K-th largest (last kept)
    vK = sv[K].item() if N > K else float("-inf")  # (K+1)-th (first dropped)
    gap = vK1 - vK
    vmax = sv[0].item()
    # near-boundary density: how many values lie within [vK, vK1] band width
    # scaled up (a proxy for "how many candidates the threshold must resolve")
    span = (vmax - sv[-1].item()) + 1e-9
    band = ((lg > vK) & (lg <= vK1 + 1e-6)).sum().item()  # ~= K trivially; noise
    # tighter: count within one relative gap above the boundary => contention
    contest = ((lg > vK1) & (lg <= vK1 + max(gap, 1e-6) * 4)).sum().item()
    return dict(
        N=N, K=K, hit_rate=round(b["hit_rate"], 4),
        gap=gap, gap_rel=gap / span, vmax=vmax, vK1=vK1, vK=vK,
        mean=lg.mean().item(), std=lg.std().item(),
        p50=sv[N // 2].item(), band=band, contest4gap=contest,
        top_mass_ratio=(vK1 - lg.mean().item()) / (span),
    )


def main():
    models = [a for a in sys.argv[1:] if a in RD4.MODELS] or list(RD4.MODELS)
    res = {}
    for model in models:
        m = RD4.MODELS[model]
        for isl in RD4.ISLS:
            L0 = m["layers"][0]
            if not (RD4._layer_dir(model, isl, L0) /
                    "decode.topk.out.pt").exists():
                continue
            RD4.prepare(model, isl)  # ensure slim cache
            steps_hr = per_step_hitrate(model, isl)
            # aggregate per-step hr over layers -> per-step mean/min/max
            S = len(next(iter(steps_hr.values())))
            step_agg = []
            for s in range(S):
                vals = [steps_hr[L][s] for L in steps_hr]
                vals.sort()
                step_agg.append(dict(mean=sum(vals) / len(vals),
                                     min=vals[0], med=vals[len(vals) // 2],
                                     max=vals[-1]))
            bprops = {L: boundary_props(model, isl, L) for L in m["layers"]}
            hrs = sorted(bp["hit_rate"] for bp in bprops.values())
            gaps = sorted(bp["gap_rel"] for bp in bprops.values())
            key = f"{model}/{isl}"
            res[key] = dict(
                model=model, isl=isl, N=bprops[L0]["N"], K=m["K"],
                n_layers=len(bprops), n_steps=S + 1,
                per_step_hr=step_agg,
                last_hr_min=hrs[0], last_hr_med=hrs[len(hrs) // 2],
                last_hr_max=hrs[-1],
                gaprel_min=gaps[0], gaprel_med=gaps[len(gaps) // 2],
                gaprel_max=gaps[-1],
                boundary=bprops,
            )
            print(f"{key:14s} N={bprops[L0]['N']:>7d} steps={S+1:>2d} "
                  f"last-hr[min/med/max]={hrs[0]:.3f}/{hrs[len(hrs)//2]:.3f}/"
                  f"{hrs[-1]:.3f}  gap_rel[min/med/max]="
                  f"{gaps[0]:.1e}/{gaps[len(gaps)//2]:.1e}/{gaps[-1]:.1e}  "
                  f"step-hr med(first→last)="
                  f"{step_agg[0]['med']:.3f}→{step_agg[-1]['med']:.3f}")
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT} ({len(res)} cells)")


if __name__ == "__main__":
    main()
