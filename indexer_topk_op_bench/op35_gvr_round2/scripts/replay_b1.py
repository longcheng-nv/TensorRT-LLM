# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op35 iter0 rung-1: host replay of the P3-collect scan geometry to bound
three EXACT sparsification designs, on the 77 §6 metric cells (synth 52 +
real 25, BS=1 fp32).

Designs bounded (all keep P2 count pass intact, offsets stay exact):
  A) warp early-exit  — warp stops after all 32 lanes wrote their quota
     (zero sideband; saving = mean over warps of suffix windows after the
      warp's last candidate, weighted by windows).
  B) (warp,window) sideband skip — P2 records per-(warp,window) "contains
     any v >= admitted rung" (u8 rung-class); P3 skips empty quanta
     (saving = fraction of (warp,window) quanta with zero candidates).
  C) block-window skip (8K-granularity, the proposal's literal B1) —
     saving = fraction of whole windows with zero candidates.

Replays the REAL launch geometry per pick_config (cs/nt/vec_w) and simulates
the R0 admission (order-stat rungs + vseed pmean) to get thr_final.

Output: results/replay_b1.csv + printed summary.
"""
import csv
import math
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_OP35 = _HERE.parent
_BENCH = _OP35.parent
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_BENCH / "op22_temporal_fixed_hr_bench"))
os.environ.setdefault("SYNTH_POSITIONAL", "1")

import bundle_data_env as SYNTH          # noqa: E402
import real_data_v4cap as RV4            # noqa: E402
import real_data_v32 as RV32             # noqa: E402

DEV = "cuda"
N_SEQ = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
REAL_LAYER = {"flash": 22, "pro": 30, "v32": 34}
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
# fp32 GvrParams kC by (K, cr) — from gvr_topk_decode.py table + kC-diet
KC = {(512, 1): 5120, (1024, 1): 5120, (2048, 1): 6144,
      (512, 4): 5120, (1024, 4): 5120, (2048, 4): 6144}


def pick_shape(N, BS=1, num_sms=148):
    if N < 65536:
        cs = 1
    elif BS <= 4 and N >= 131072:
        cs = 8
    elif BS * 4 <= num_sms:
        cs = 4
    elif BS * 2 <= num_sms:
        cs = 2
    else:
        cs = 1
    npc = N // cs
    nt = 1024 if (BS <= num_sms and npc >= 65536) else 512
    vec_w = 8 if npc >= 16384 else 4       # fp32: 256-bit vs 128-bit
    return cs, nt, vec_w


def sim_r0(vals, pre_gather, K, kC):
    """Simulate R0 admission: rungs at qneeds-th largest gathered value +
    vseed pmean column; admit tightest count in [K,kC]; on miss return the
    K-th value (falsi-converged proxy). Returns (thr, admitted_bool)."""
    qfracs = (0.85, 0.35) if K == 2048 else (0.85,)
    g = pre_gather.float()
    gs = g.sort(descending=True).values
    rungs = [gs[min(max(1, math.ceil(q * K)), g.numel()) - 1].item() for q in qfracs]
    rungs.append(g.mean().item())          # vseed
    best = None
    for t in rungs:
        c = int((vals >= t).sum())
        if K <= c <= kC and (best is None or c < best[1]):
            best = (t, c)
    if best is not None:
        return best[0], True
    kth = torch.kthvalue(vals.neg(), K).values.neg().item()   # K-th largest
    return kth, False


def replay_cell(vals_f32, pre_idx, K, cr, N):
    kC = KC[(K, cr)]
    if K == 512 and N < 65536:
        kC = 3072                          # kC-diet (cs==1)
    cs, nt, vec_w = pick_shape(N)
    pre_g = vals_f32[pre_idx.clamp(0, N - 1).long()]
    thr, admitted = sim_r0(vals_f32, pre_g, K, kC)
    mask = (vals_f32 >= thr)
    cand = int(mask.sum())

    tot_q = tot_qz = 0                     # (warp,window) quanta / zero quanta
    tot_win = tot_winz = 0                 # whole windows / zero windows
    ee_scan = full_scan = 0.0              # early-exit scanned vs full (warp-windows)
    n_warps = nt // 32
    quant = 32 * vec_w                     # elems per (warp,window) quantum
    win = nt * vec_w                       # elems per window
    for c in range(cs):
        s0 = (N * c) // cs
        s1 = (N * (c + 1)) // cs
        sl = s1 - s0
        m = math.ceil(sl / win)
        pad = m * win - sl
        msk = mask[s0:s1]
        if pad:
            msk = torch.cat([msk, torch.zeros(pad, dtype=torch.bool, device=msk.device)])
        # [m, n_warps, quant] — window-major, warp covers contiguous quant
        q = msk.view(m, nt * vec_w).view(m, n_warps, quant).any(dim=2)   # [m, n_warps]
        tot_q += m * n_warps
        tot_qz += int((~q).sum())
        wz = q.any(dim=1)                  # window has any candidate
        tot_win += m
        tot_winz += int((~wz).sum())
        # early-exit: warp w scans until its LAST candidate window (inclusive);
        # warps with zero candidates still execute window 0 check (~1 window).
        has = q.any(dim=0)                 # [n_warps]
        last = torch.where(has, m - 1 - q.flip(0).float().argmax(dim=0), torch.zeros(n_warps, dtype=torch.long, device=q.device))
        scanned = torch.where(has, last + 1, torch.ones_like(last))
        ee_scan += float(scanned.sum())
        full_scan += m * n_warps
    return dict(cs=cs, nt=nt, vec_w=vec_w, thr_admitted=admitted, cand=cand, kC=kC,
                skipB=tot_qz / tot_q,          # sideband (warp,window) skip frac
                skipC=tot_winz / tot_win,      # whole-window skip frac
                skipA=1.0 - ee_scan / full_scan)   # early-exit saved frac


def main():
    rows = []
    torch.manual_seed(0)
    for scen in ("best", "worst"):
        for K in (512, 1024, 2048):
            for N in N_SEQ:
                if N <= 2 * K:
                    continue
                b = SYNTH.get_bundle(scen, K, torch.float32, N)
                lg = b["logits"][0, :N].float().to(DEV)
                pre = b["preIdx"][0].to(DEV)
                r = replay_cell(lg, pre, K, b["cr"], N)
                r.update(family="synth", scen=scen, model="", isl="", K=K, N=N)
                rows.append(r)
                print(f"synth {scen} K{K} N{N}: cs{r['cs']} cand={r['cand']} adm={r['thr_admitted']} "
                      f"A={r['skipA']:.3f} B={r['skipB']:.3f} C={r['skipC']:.3f}", flush=True)
    for model in ("flash", "pro", "v32"):
        RD = RV32 if model == "v32" else RV4
        for isl in REAL_ISLS[model]:
            b = RD.get_bundle(model, isl, REAL_LAYER[model], "fp32")
            N = b["N"]
            lg = b["logits"][0, :N].float().to(DEV)
            pre = b["preIdx"][0].to(DEV)
            r = replay_cell(lg, pre, b["K"], b["cr"], N)
            r.update(family="real", scen="", model=model, isl=isl, K=b["K"], N=N)
            rows.append(r)
            print(f"real {model} {isl} K{b['K']} N{N}: cs{r['cs']} cand={r['cand']} adm={r['thr_admitted']} "
                  f"A={r['skipA']:.3f} B={r['skipB']:.3f} C={r['skipC']:.3f}", flush=True)

    out = _OP35 / "results" / "replay_b1.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # summary
    import statistics as st
    for fam in ("synth", "real"):
        sub = [r for r in rows if r["family"] == fam]
        for reg, lo in (("N<64K", 0), ("N>=64K", 65536)):
            s2 = [r for r in sub if (r["N"] >= 65536) == (lo == 65536)]
            if not s2:
                continue
            print(f"{fam} {reg} ({len(s2)}): "
                  f"A med={st.median(r['skipA'] for r in s2):.3f} "
                  f"B med={st.median(r['skipB'] for r in s2):.3f} "
                  f"C med={st.median(r['skipC'] for r in s2):.3f} "
                  f"miss={sum(not r['thr_admitted'] for r in s2)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
