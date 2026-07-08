# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 Step 3 (S1b) — marginal value of rho-hat EMA placement ON TOP of
the S1a ship ladder (wide4b + cap 8192), on real Pro multi-turn chains.

The op21 envelope correction (iter15.3) deferred Step 3 BEHIND the P2
diet; S1a moves the static baseline again, so the question is now: does
causal EMA placement (with the low-side asymmetric band the value-level
replay prescribed) still buy anything over wide4b_c8?

Arms (all judged by proto_hls.simulate_r0, cap=8192 unless noted):
    base3       static (0.75,0.5,0.25), cap 4096      [pre-op25 ship]
    wide4b_c8   static (0.92,0.6,0.25,0.048), cap 8192 [S1a ship]
    ema_sym     cols_h_aware(hhat, 2*sig)               [old Step-3 spec]
    ema_asym    hhat*{1+1*sig_rel, 1, 1-3*sig_rel} + 0.048 tail, cap 8192
                (low-side widened: residual misses are all_ge downward
                jumps — value-level replay 2026-07-08)
    oracle      cols_h_aware(h_true, 0.35)

Output: fallback rates + model E[T] priced at the Pro geometry (N~9.4K,
C=1) AND projected at envelope N (65K/131K/262K, C=4, iter16 dist fallback)
— projection keeps the admission outcome, reprices the passes.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[0]
WSROOT = HERE.parents[1]
sys.path.insert(0, str(BENCH / "ops"))
sys.path.insert(0, str(BENCH / "op21_gvr_prod" / "scripts"))
sys.path.insert(0, str(WSROOT / "multi_turn_indexer_studies" / "pro"
                       / "analysis" / "hls_assumption_check"))

import proto_hls as P  # noqa: E402
P.DEV = "cpu"
from check_hls_assumptions import ema_track, ALPHA_DEF, BETA  # noqa: E402

PRO = WSROOT / "multi_turn_indexer_studies" / "pro"
CAPS = {"turn1": PRO / "captures/turn1_20260603T031341Z_pro_both",
        "turn2": PRO / "captures/turn2_20260603T033921Z_pro_both"}
K, CR = 1024, 4
WIDE4B = (0.92, 0.60, 0.25, 0.048)
TAU = {3: 1.20, 4: 1.46, 5: 1.75}
ALPHA_FB = 0.2


def cols_ema_asym(row, hhat, sig):
    """h-centered 3 cols, low side widened 3x, + 0.048 all_ge tail."""
    sr = max(min(sig / max(hhat, 1e-3), 0.6), 0.05)   # relative sigma
    fr = [min(hhat * (1 + 1.0 * sr), 0.97),
          min(max(hhat, 0.02), 0.95),
          max(hhat * (1 - 3.0 * sr), 0.02)]
    fr.append(0.048)
    fr = sorted(set(round(f, 4) for f in fr), reverse=True)
    cols = [row.gmin] + [row.q(f) for f in fr]
    return P._mono(cols, [1.0] + fr)


def judge(row, cols, fr, cap):
    r0 = P.simulate_r0(row, cols, fr, cap=cap)
    r0["_cols"] = cols
    fbp = 0
    if r0["mode"] != "fast":
        fbp, _ = P.fb_logfalsi(row, r0, alpha=ALPHA_FB)
    return r0["mode"], max(fbp, 1) if r0["mode"] != "fast" else 0


def main():
    arms = ("base3", "wide4b_c8", "ema_sym", "ema_asym", "oracle")
    stats = {a: {"n": 0, "fast": 0, "fbp": 0, "m": 0} for a in arms}
    for turn, cap_dir in CAPS.items():
        skip = 2 if turn == "turn2" else 0
        for L in range(2, 61, 2):
            tk = torch.load(cap_dir / f"layer_{L:02d}" / "decode.topk.out.pt",
                            map_location="cpu")
            lg = torch.load(cap_dir / f"layer_{L:02d}"
                            / "decode.logits.in.pt", map_location="cpu")
            steps = [s for s in sorted(tk.keys())
                     if not (tk[s] < 0).any().item()]
            sets = [set(tk[s].flatten().tolist()) for s in steps]
            h_full = np.array([len(sets[i - 1] & sets[i]) / K
                               for i in range(1, len(sets))])
            h = h_full[skip:]
            hhat, sig = ema_track(h, ALPHA_DEF, BETA)
            vls, run_vl = [], 0
            for s in steps:
                run_vl = max(run_vl, int(tk[s].max().item()) + 1)
                vls.append(run_vl)
            for j in range(len(h)):
                i_cur = skip + j + 1
                n = vls[i_cur]
                x = lg[steps[i_cur]].flatten().float()[:n]
                pre = tk[steps[i_cur - 1]].flatten().long()
                row = P.Row("pro", K, n,
                            {"cr": CR, "logits": x.unsqueeze(0),
                             "preIdx": pre.unsqueeze(0)})
                specs = {
                    "base3": (P.cols_static(row), 4096, 3),
                    "wide4b_c8": (P.cols_static(row, qfracs=WIDE4B), 8192, 4),
                    "ema_sym": (P.cols_h_aware(
                        row, float(hhat[j]), 2.0 * float(sig[j]), m_thr=4),
                        8192, 3),
                    "ema_asym": (cols_ema_asym(
                        row, float(hhat[j]), float(sig[j])), 8192, 4),
                    "oracle": (P.cols_h_aware(
                        row, max(row.h_true, 0.02), 0.35, m_thr=4), 8192, 3),
                }
                for a, ((cols, fr), cap, m) in specs.items():
                    mode, fbp = judge(row, cols, fr, cap)
                    st = stats[a]
                    st["n"] += 1
                    st["fast"] += mode == "fast"
                    st["fbp"] += fbp
                    st["m"] = m
            print(f"{turn} L{L:02d} done", flush=True)

    print("\n=== S1b marginal replay (real Pro chains, 29.8k transitions) ===")
    print(f"{'arm':12s} {'fast':>7s} {'mean_fbp':>9s} "
          + "".join(f"  E[T]us@{n // 1024}K" for n in (9600, 65536, 131072,
                                                       262144)))
    out = {}
    for a in arms:
        st = stats[a]
        fast = st["fast"] / st["n"]
        fbp = st["fbp"] / st["n"]
        row = f"{a:12s} {fast:7.4f} {fbp:9.3f}"
        ets = []
        for n, c in ((9600, 1), (65536, 4), (131072, 4), (262144, 4)):
            p = n * 4.0 / P.BW_PASS
            t = TAU[st["m"]] * p / c + P.FIXED_FLOOR \
                + (P.S_CLUSTER if c > 1 else 0) + fbp * p / c
            ets.append(t * 1e6)
            row += f" {t * 1e6:10.2f}"
        out[a] = {"fast": fast, "fbp": fbp, "et_us": ets}
        print(row)
    (HERE / "results" / "s1b_replay.json").write_text(json.dumps(out,
                                                                 indent=1))
    w, s = out["wide4b_c8"], out["ema_asym"]
    for i, n in enumerate(("9.4K", "65K", "131K", "262K")):
        d = (w["et_us"][i] - s["et_us"][i]) / w["et_us"][i] * 100
        print(f"ema_asym vs wide4b_c8 @{n}: {d:+.2f}% model")


if __name__ == "__main__":
    main()
