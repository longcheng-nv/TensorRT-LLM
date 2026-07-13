#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""HBE-C rung-0 CRUX — host replay of the hint-ladder placement (DESIGN §6).

Replicates C0 bit-level-faithful to the intended kernel (P1b lineage,
gvr_ms_op.phase1b_rank_quantile semantics):
  - hint gather = logits[preIdx (+1 mod N for cr=1)], STRIDE-4 subsample
    (every 4th preIdx entry -> ~K/4 values), invalid idx skipped;
  - 256-bin histogram over [lo, hi] of the subsample, bin = (v-lo)*inv with
    inv = (QBINS-1+0.99)/rng (extract-coarse-bin analogue of P1b);
  - suffix scan; rung m = bin LEFT-edge where ge-count crosses
    frac[m] * total_subsample (largest bin with suffix >= tgt);
  - fracs per K = op25/op27 ship geometry: K512/K1024 (0.92,0.45,0.048),
    K2048 tail (0.75,0.45,0.048); "stock" (0.75,0.5,0.25) as comparison arm.

Then evaluates C2 bracket outcome on the FULL row (ground truth):
  cnt_r = count(row >= v_r);  r* = tightest (highest-value) rung with
  cnt >= K;  MISS = lt_K (no rung brackets) or cand-overflow (cnt at the
  lowest rung, i.e. total collected candidates, > cap policy).
  E[passes] = 1 + 2*miss  (miss rows redo stock cluster Phase1+Phase2).

GO line (DESIGN §6, real scenario): E[passes] <= ~1.2 and miss <= ~10%.

Sources:
  op22rr bundles: (best|worst|real) x K{512,1024,2048} x cluster-domain N
                  {65536,131072,262144,524288,1048576}, fp32 (1 row each;
                  pilot cells replicate one row to BS -> row replay exact).
  realcap: real_data_v2 last-decode-step rows, all layers x 3 models
                  (flash N=25154, pro N=14478, v32 N=70690 — estimator-
                  quality probe; N mostly below the cluster-domain prize).

Usage: python3 replay_hbec_ladder.py [--json out.jsonl] [--skip-realcap]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OPB = HERE.parents[1]
sys.path.insert(0, str(OPB / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(OPB / "harness"))

import bundle_data_rr  # noqa: E402
import real_data_v2 as RD2  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
QBINS = 256
SUB_STRIDE = 4
FRACS_SHIP = {512: (0.92, 0.45, 0.048), 1024: (0.92, 0.45, 0.048),
              2048: (0.75, 0.45, 0.048)}
FRACS_STOCK = (0.75, 0.5, 0.25)
NS_RR = (65536, 131072, 262144, 524288, 1048576)
CAPS_XK = (8, 16, 32)          # cand cap policies, units of K


def hint_subsample(row, pre, N, cr):
    """C0 gather: stride-4 subsample of the hint, valid-filtered values."""
    idx = pre.long()[::SUB_STRIDE]
    if cr == 1:
        idx = (idx + 1) % N            # production cr=1 preIdxOffset
    ok = (idx >= 0) & (idx < N)
    return row[idx[ok]]


def place_ladder(sub, fracs):
    """P1b-faithful 256-bin left-edge rank-quantile rungs (ASCENDING value
    order like the HLS column array; caller treats last = tightest)."""
    total = sub.numel()
    if total == 0:
        return None
    lo = float(sub.min())
    hi = float(sub.max())
    rng = hi - lo
    if rng <= 0:
        return [lo] * len(fracs)
    inv = (QBINS - 1 + 0.99) / rng
    b = torch.clamp(((sub - lo) * inv).long(), 0, QBINS - 1)
    sfx = torch.bincount(b, minlength=QBINS).flip(0).cumsum(0).flip(0)
    binw = rng / QBINS
    sfx_c = sfx.cpu()
    rungs = []
    for f in fracs:                      # descending fracs -> ascending value
        tgt = max(int(total * f), 1)
        hit = (sfx_c >= tgt).nonzero()
        bi = int(hit.max()) if hit.numel() else 0
        rungs.append(lo + bi * binw)
    # enforce non-descending (P1b epilogue)
    for i in range(1, len(rungs)):
        rungs[i] = max(rungs[i], rungs[i - 1])
    return rungs


def replay_row(row, pre, N, K, cr, fracs):
    """One row, one arm. Returns outcome record (no aggregation)."""
    sub = hint_subsample(row, pre, N, cr)
    rungs = place_ladder(sub, fracs)      # ascending: [loosest..tightest]
    if rungs is None:
        return {"miss": True, "miss_mode": "no_valid_hint"}
    cnts = [int((row >= v).sum()) for v in rungs]   # descending counts
    # r* = tightest rung with cnt >= K  (highest index in ascending order)
    r_star = -1
    for m in range(len(rungs)):
        if cnts[m] >= K:
            r_star = m
    cand_all = cnts[0]                    # everything >= lowest rung is stored
    rec = {
        "sub_n": int(sub.numel()),
        "rungs": [round(v, 6) for v in rungs],
        "cnts": cnts,
        "r_star": r_star,                 # -1 = lt_K miss
        "cand_rstar": cnts[r_star] if r_star >= 0 else None,
        "cand_all": cand_all,
        "cand_rstar_xK": round(cnts[r_star] / K, 2) if r_star >= 0 else None,
        "cand_all_xK": round(cand_all / K, 2),
        "lt_K": r_star < 0,
    }
    for c in CAPS_XK:
        rec[f"miss_cap{c}K"] = rec["lt_K"] or cand_all > c * K
    return rec


def emit(rec, out):
    print(json.dumps(rec), flush=True)
    if out:
        out.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-realcap", action="store_true")
    args = ap.parse_args()
    out = open(args.json, "a") if args.json else None
    # ship resolved per K; w3a = 0.92 top column for ALL K (fixes the K2048
    # h>0.75 lt_K hole found in the first pass: rr-real N524288 h=0.82)
    arms = {"ship": None, "stock": FRACS_STOCK,
            "w3a": (0.92, 0.45, 0.048)}

    # ---- op22rr bundles ----
    for scen in ("real", "best", "worst"):
        for K in (512, 1024, 2048):
            for N in NS_RR:
                try:
                    b = bundle_data_rr.get_bundle(scen, K, torch.float32, N,
                                                  device=DEV)
                except (FileNotFoundError, AssertionError) as e:
                    print(f"# skip {scen} K{K} N{N}: {type(e).__name__}",
                          flush=True)
                    continue
                row = b["logits"][0, :N].float()
                pre = b["preIdx"][0]
                for arm, fr in arms.items():
                    fracs = FRACS_SHIP[K] if fr is None else fr
                    r = replay_row(row, pre, N, K, b["cr"], fracs)
                    r.update({"src": "rr", "scenario": scen, "K": K, "N": N,
                              "arm": arm, "hr_meta": b["kernel_hit_rate"]})
                    emit(r, out)

    # ---- real captures (all layers) ----
    if not args.skip_realcap:
        for model in ("flash", "pro", "v32"):
            K = RD2.MODELS[model]["K"]
            for layer in RD2.MODELS[model]["layers"]:
                b = RD2.get_real_bundle_v2(model, layer, "fp32")
                N = b["N"]
                row = b["logits"][0, :N].float()
                pre = b["preIdx"][0]
                for arm, fr in arms.items():
                    fracs = FRACS_SHIP[K] if fr is None else fr
                    r = replay_row(row, pre, N, K, b["cr"], fracs)
                    r.update({"src": "realcap", "scenario": model, "K": K,
                              "N": N, "arm": arm, "layer": layer,
                              "hr_meta": b["hit_rate"]})
                    emit(r, out)
    if out:
        out.close()


if __name__ == "__main__":
    main()
