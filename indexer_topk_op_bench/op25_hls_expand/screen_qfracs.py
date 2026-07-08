# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 Step 1 (S1a) — host-replay screening of static ladder geometries.

Question: can a re-spaced static qfracs ladder raise the HLS fast-path
admission rate on the axes where HLS currently loses (op22rr worst/best
scenarios, real Pro multi-turn h in [0.62, 0.90]) without regressing the
shipped (0.75, 0.5, 0.25) geometry anywhere?

Three data axes, one Row build per bundle, all candidate arms replayed on
the shared Row (proto_hls.simulate_r0 = kernel-exact R0 admission):

  A. op22rr grid   : bundles_rr/{best,worst} + bundles/real, fp32,
                     3 models x 9 N  (the REPORT.html verdict axis)
  B. op24 sweep    : 392-combo parameter grid (hr 0.05..0.90 + sampled)
                     -> per-geometry coverage window in h
  C. Pro real      : multi-turn value-level replay (2 turns x 30 layers,
                     ~29.8k transitions, the real-production axis;
                     N~9.4K single-CTA ms path)

Arms: static geometries (anchor g_min implicit, collect col = qfracs[0])
      + 'oracle035' = cols_h_aware(h_true, delta=0.35) upper bound.

Cost model (post-iter16 pricing): t = tau(M-1)*p/C + floor (+S_CLUSTER)
      + fb_passes*p/C_fb, C = 4 iff N>=65536 (BS=1), C_fb = C (iter16
      dist-fallback default gate n>=65536).  Model ranks arms only —
      silicon (Step 2) is the verdict.

Output: results/screen_qfracs.jsonl + results/SCREEN_RANK.md
"""
import json
import math
import sys
import time
from collections import defaultdict
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

OUT = HERE / "results"
ALPHA = 0.2                      # iter13 shipped fb_logfalsi default
MODEL_K = {"v4flash": 512, "v4pro": 1024, "v32": 2048}

# ---- candidate ladders (descending rank fractions; collect = [0]) ----
# each arm: qf tuple + collect-cap model (4096 = current kernel;
# 8192 = doubled P3 slot buffer, the S1a-companion kernel lever)
GEOMS = {
    "base3":    {"qf": (0.75, 0.50, 0.25), "cap": 4096},   # shipped
    "deep3a":   {"qf": (0.92, 0.60, 0.30), "cap": 4096},
    "deep3b":   {"qf": (0.90, 0.55, 0.28), "cap": 4096},
    "deep3c":   {"qf": (0.95, 0.65, 0.33), "cap": 4096},
    "deep3d":   {"qf": (0.88, 0.50, 0.25), "cap": 4096},
    "deep4":    {"qf": (0.92, 0.75, 0.50, 0.25), "cap": 4096},
    "deep4b":   {"qf": (0.90, 0.60, 0.35, 0.18), "cap": 4096},
    "geo4low":  {"qf": (0.75, 0.30, 0.12, 0.048), "cap": 4096},
    "deep5":    {"qf": (0.94, 0.75, 0.55, 0.35, 0.18), "cap": 4096},
    # round 2: wide ladders (deep col for high-h + low col for worst)
    "wide4a":   {"qf": (0.92, 0.60, 0.30, 0.10), "cap": 4096},
    "wide4b":   {"qf": (0.92, 0.60, 0.25, 0.048), "cap": 4096},
    "wide5":    {"qf": (0.92, 0.65, 0.40, 0.20, 0.06), "cap": 4096},
    # round 2: cap-raise variants (kernel P3 slot buffer x2)
    "base3_c8": {"qf": (0.75, 0.50, 0.25), "cap": 8192},
    "deep3a_c8": {"qf": (0.92, 0.60, 0.30), "cap": 8192},
    "wide4a_c8": {"qf": (0.92, 0.60, 0.30, 0.10), "cap": 8192},
    "wide5_c8": {"qf": (0.92, 0.65, 0.40, 0.20, 0.06), "cap": 8192},
    # round 3: low-tail combinations under cap 8192
    "wide4b_c8": {"qf": (0.92, 0.60, 0.25, 0.048), "cap": 8192},
    "wide4c_c8": {"qf": (0.92, 0.55, 0.15, 0.048), "cap": 8192},
    "wide4d_c8": {"qf": (0.90, 0.50, 0.12, 0.048), "cap": 8192},
    "wide5b_c8": {"qf": (0.92, 0.60, 0.30, 0.12, 0.048), "cap": 8192},
}
TAU = dict(P.TAU)
TAU[5] = 1.75                    # interp between measured 4:1.46 and 6:2.05


def t_model(N, m_thr_cols, mode, fbp):
    """Model seconds for one row (BS=1, fp32), post-iter16 pricing."""
    p = N * 4.0 / P.BW_PASS
    C = 4 if N >= 65536 else 1
    t = TAU[m_thr_cols] * p / C + P.FIXED_FLOOR + (P.S_CLUSTER if C > 1 else 0)
    if mode != "fast":
        t += max(fbp, 1) * p / C
    return t


def eval_arms(row):
    """Replay every arm on one Row; returns {arm: rec}."""
    out = {}
    for name, spec in GEOMS.items():
        qf, cap = spec["qf"], spec["cap"]
        cols, fr = P.cols_static(row, qfracs=qf)
        r0 = P.simulate_r0(row, cols, fr, cap=cap)
        r0["_cols"] = cols
        fbp, fb_ok = 0, True
        if r0["mode"] != "fast":
            fbp, fb_ok = P.fb_logfalsi(row, r0, alpha=ALPHA)
        out[name] = {"mode": r0["mode"], "fbp": fbp, "fb_ok": fb_ok,
                     "m": len(qf),
                     "t": t_model(row.N, len(qf), r0["mode"], fbp)}
    # oracle: h-aware placement at true h, delta=0.35 (math-doc oracle)
    cols, fr = P.cols_h_aware(row, max(row.h_true, 0.02), 0.35, m_thr=4)
    r0 = P.simulate_r0(row, cols, fr)
    r0["_cols"] = cols
    fbp, fb_ok = 0, True
    if r0["mode"] != "fast":
        fbp, fb_ok = P.fb_logfalsi(row, r0, alpha=ALPHA)
    out["oracle035"] = {"mode": r0["mode"], "fbp": fbp, "fb_ok": fb_ok,
                        "m": 3, "t": t_model(row.N, 3, r0["mode"], fbp)}
    return out


def screen_bundle(d, model, K, N, tag, scen):
    logits = torch.load(d / "logits.pt", map_location=P.DEV)
    preIdx = torch.load(d / "preIdx.pt", map_location=P.DEV)
    meta = json.loads((d / "meta.json").read_text())
    bundle = {"logits": logits, "preIdx": preIdx.to(torch.int32),
              "cr": meta["compress_ratio"],
              "kernel_hit_rate": meta.get("realised_hr_mean")}
    row = P.Row(scen, K, N, bundle)
    arms = eval_arms(row)
    return {"axis": tag, "scen": scen, "model": model, "K": K, "N": N,
            "h_true": round(row.h_true, 4), "rho": round(row.rho_true, 4),
            "arms": arms}


def axis_a_op22rr(recs, out_f):
    b22 = BENCH / "op22_temporal_fixed_hr_bench"
    for scen, root in (("best", b22 / "bundles_rr" / "best"),
                       ("worst", b22 / "bundles_rr" / "worst"),
                       ("real", b22 / "bundles" / "real")):
        for md in sorted(root.glob("*_fp32_N*")):
            model = md.name.split("_")[0]
            N = int(md.name.split("_N")[1])
            leaf = sorted(md.glob("*_bs1"))
            if not leaf:
                continue
            try:
                r = screen_bundle(leaf[0], model, MODEL_K[model], N,
                                  "op22rr", scen)
            except Exception as e:  # noqa: BLE001 — screening must not die
                r = {"axis": "op22rr", "scen": scen, "model": model, "N": N,
                     "error": f"{type(e).__name__}: {str(e)[:120]}"}
            recs.append(r)
            out_f.write(json.dumps(r) + "\n")
            out_f.flush()
    print(f"axis A op22rr done: {sum(r['axis'] == 'op22rr' for r in recs)}",
          flush=True)


def axis_b_op24(recs, out_f):
    root = BENCH / "op24_synth_favorability" / "bundles"
    dirs = sorted(root.glob("*/N*"))
    for i, d in enumerate(dirs):
        combo = d.parent.name
        model = combo.split("_")[0]
        N = int(d.name[1:])
        try:
            r = screen_bundle(d, model, MODEL_K[model], N, "op24", combo)
        except Exception as e:  # noqa: BLE001
            r = {"axis": "op24", "scen": combo, "model": model, "N": N,
                 "error": f"{type(e).__name__}: {str(e)[:120]}"}
        recs.append(r)
        out_f.write(json.dumps(r) + "\n")
        if (i + 1) % 80 == 0:
            out_f.flush()
            print(f"  op24 {i + 1}/{len(dirs)}", flush=True)
    out_f.flush()


def axis_c_pro(recs, out_f, stride=1):
    pro = WSROOT / "multi_turn_indexer_studies" / "pro"
    caps = {"turn1": pro / "captures/turn1_20260603T031341Z_pro_both",
            "turn2": pro / "captures/turn2_20260603T033921Z_pro_both"}
    K, cr = 1024, 4
    old_dev = P.DEV
    P.DEV = "cpu"                 # tiny rows; CPU beats launch overhead
    try:
        for turn, cap in caps.items():
            skip = 2 if turn == "turn2" else 0
            for L in range(2, 61, 2):
                tk = torch.load(cap / f"layer_{L:02d}" / "decode.topk.out.pt",
                                map_location="cpu")
                lg = torch.load(cap / f"layer_{L:02d}"
                                / "decode.logits.in.pt", map_location="cpu")
                steps = [s for s in sorted(tk.keys())
                         if not (tk[s] < 0).any().item()]
                vls, run_vl = [], 0
                for s in steps:
                    run_vl = max(run_vl, int(tk[s].max().item()) + 1)
                    vls.append(run_vl)
                for i in range(skip + 1, len(steps), stride):
                    n = vls[i]
                    x = lg[steps[i]].flatten().float()[:n]
                    pre = tk[steps[i - 1]].flatten().long()
                    row = P.Row("pro", K, n,
                                {"cr": cr, "logits": x.unsqueeze(0),
                                 "preIdx": pre.unsqueeze(0)})
                    arms = eval_arms(row)
                    r = {"axis": "pro", "scen": f"{turn}_L{L:02d}",
                         "model": "pro_real", "K": K, "N": n,
                         "h_true": round(row.h_true, 4),
                         "rho": round(row.rho_true, 4), "arms": arms}
                    recs.append(r)
                    out_f.write(json.dumps(r) + "\n")
                print(f"  pro {turn} L{L:02d} done", flush=True)
                out_f.flush()
    finally:
        P.DEV = old_dev


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def summarize(recs):
    arms = list(GEOMS) + ["oracle035"]
    lines = ["# op25 S1a qfracs screening", ""]

    def block(title, rows, keyfn=None):
        lines.append(f"## {title}  (n={len(rows)})")
        lines.append("| arm | fast | all_ge | pair01 | ovfl | band>kC | "
                     "mean fbp | t gm vs base3 | fb_fail |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        base_t = [r["arms"]["base3"]["t"] for r in rows]
        for a in arms:
            xs = [r["arms"][a] for r in rows]
            md = lambda m: sum(x["mode"] == m for x in xs) / len(xs)  # noqa: E731
            rel = gm([bt / x["t"] for bt, x in zip(base_t, xs)])
            fbf = sum((not x["fb_ok"]) and x["mode"] != "fast" for x in xs)
            lines.append(
                f"| {a} | {md('fast'):.3f} | {md('all_ge'):.3f} | "
                f"{md('pair01'):.3f} | {md('overflow'):.3f} | "
                f"{md('band_gt_kC'):.3f} | "
                f"{sum(x['fbp'] for x in xs) / len(xs):.2f} | {rel:.3f} | "
                f"{fbf} |")
        lines.append("")

    ok = [r for r in recs if "error" not in r]
    for axis in ("op22rr", "op24", "pro"):
        rows = [r for r in ok if r["axis"] == axis]
        if not rows:
            continue
        block(f"axis {axis} — ALL", rows)
        if axis == "op22rr":
            for scen in ("best", "worst", "real"):
                block(f"op22rr / {scen}",
                      [r for r in rows if r["scen"] == scen])
        if axis == "op24":
            # coverage window: fast-rate per target-hr bucket
            byhr = defaultdict(list)
            for r in rows:
                s = r["scen"]
                hr = ("samp" if "hrsamp" in s else
                      s.split("hr")[-1].split("_")[0] if "hr" in s else "?")
                byhr[hr].append(r)
            for hr in sorted(byhr):
                block(f"op24 / target_hr={hr}", byhr[hr])
    errs = [r for r in recs if "error" in r]
    lines.append(f"errors: {len(errs)}")
    for r in errs[:8]:
        lines.append(f"  ERR {r['axis']} {r.get('scen')} N{r.get('N')} "
                     f"{r['error']}")
    md = "\n".join(lines)
    (OUT / "SCREEN_RANK.md").write_text(md + "\n")
    print(md)


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    recs = []
    with open(OUT / "screen_qfracs.jsonl", "w") as out_f:
        axis_a_op22rr(recs, out_f)
        axis_b_op24(recs, out_f)
        axis_c_pro(recs, out_f, stride=int(sys.argv[1])
                   if len(sys.argv) > 1 else 1)
    summarize(recs)
    print(f"TOTAL {len(recs)} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
