# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op30 phase 1 — pick the GVR-base BEST/WORST (cfg, hr) per model from the
calibration nsys sweep.

Score(cfg, hr) = geomean over N of [ cold_us(gvr_cutedsl; cfg,hr,N)
                                     / min over (cfg',hr') at same N ].
BEST = argmin score (fastest GVR-base), WORST = argmax (slowest).
Sanity: radix_cutedsl control spread (max/min per N) should be ~1.0x —
prints a warning above 1.05x.

Writes scen_op30.json + CALIBRATION.md.
Usage: python3 pick_scen_op30.py [<calib_out_root>]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

from gen_calib_bundles_op30 import MODELS, CAL_NS  # noqa: E402


def load(out_root):
    rows = []
    for model in MODELS:
        jl = out_root / f"calib_{model}.jsonl"
        rep = out_root / "nsys_reps" / f"calib_{model}.nsys-rep"
        if not jl.exists():
            print(f"MISSING {jl}")
            continue
        kern = parse_rep(rep) if rep.exists() else {}
        for line in jl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                print(f"  cell error: {r['model']} {r['cfg']} "
                      f"hr{r['target_hr']} N{r['N']} {r['op']}: {r['error']}")
                continue
            us = kern.get(r["range_cold"])
            if us is None:
                print(f"  no range: {r['range_cold']}")
                continue
            r["us_cold"] = us
            r["us_warm"] = kern.get(r["range_warm"])
            rows.append(r)
    return rows


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    out_root = (Path(sys.argv[1]) if len(sys.argv) > 1
                else HERE.parents[0] / "results_b200_op30_calib")
    rows = load(out_root)
    exact_fail = [r for r in rows if r.get("exact") == "FAIL"]
    assert not exact_fail, f"exactness failures: {exact_fail[:5]}"

    scen = {"best": {}, "worst": {}}
    md = ["# op30 calibration — GVR-base (cuteDSL) favorability grid",
          "",
          f"Source: {out_root} (nsys cold-L2 canonical, fp32 BS=1, "
          f"N∈{CAL_NS}, seed=42+crc32(K|N)%1e6)", ""]
    for model, K in MODELS.items():
        gvr = defaultdict(dict)    # (cfg,hr) -> {N: us}
        rad = defaultdict(dict)
        hrs = {}
        for r in rows:
            if r["model"] != model:
                continue
            key = (r["cfg"], r["target_hr"])
            if r["op"] == "gvr_cutedsl":
                gvr[key][r["N"]] = r["us_cold"]
                hrs[key] = r.get("realised_hr")
            elif r["op"] == "radix_cutedsl":
                rad[key][r["N"]] = r["us_cold"]
        ns = sorted({n for v in gvr.values() for n in v})
        full = {k: v for k, v in gvr.items() if set(v) == set(ns)}
        if not full:
            print(f"{model}: NO complete cells, skipping")
            continue
        best_per_n = {n: min(v[n] for v in full.values()) for n in ns}
        score = {k: geomean([v[n] / best_per_n[n] for n in ns])
                 for k, v in full.items()}
        k_best = min(score, key=score.get)
        k_worst = max(score, key=score.get)
        scen["best"][model] = list(k_best)
        scen["worst"][model] = list(k_worst)

        # radix control spread per N
        ctrl = []
        for n in ns:
            us = [v[n] for v in rad.values() if n in v]
            if us:
                ctrl.append(max(us) / min(us))
        ctrl_max = max(ctrl) if ctrl else float("nan")
        flag = "  ** WARN control spread > 1.05x **" if ctrl_max > 1.05 else ""

        md += [f"## {model} (K={K})", "",
               f"- **BEST**  = `{k_best[0]}` hr={k_best[1]:.2f} "
               f"(score {score[k_best]:.3f}, realised hr "
               f"{hrs.get(k_best):.3f})",
               f"- **WORST** = `{k_worst[0]}` hr={k_worst[1]:.2f} "
               f"(score {score[k_worst]:.3f}, realised hr "
               f"{hrs.get(k_worst):.3f})",
               f"- WORST/BEST time ratio (geomean): "
               f"{score[k_worst] / score[k_best]:.3f}x",
               f"- radix control max spread over cfg×hr: {ctrl_max:.3f}x{flag}",
               "", "| cfg | hr | " + " | ".join(f"N={n}" for n in ns)
               + " | score |",
               "|---|---|" + "---|" * (len(ns) + 1)]
        for k in sorted(full, key=score.get):
            cfg, hr = k
            cells = " | ".join(f"{full[k][n]:.1f}" for n in ns)
            mark = (" **B**" if k == k_best else
                    " **W**" if k == k_worst else "")
            md.append(f"| {cfg} | {hr:.2f} | {cells} | "
                      f"{score[k]:.3f}{mark} |")
        # per-N argmin/argmax consistency note
        for n in ns:
            amin = min(full, key=lambda k: full[k][n])
            amax = max(full, key=lambda k: full[k][n])
            md.append(f"* per-N N={n}: argmin={amin[0]}/hr{amin[1]:.2f} "
                      f"argmax={amax[0]}/hr{amax[1]:.2f}")
        md.append("")
        print(f"{model}: BEST={k_best} WORST={k_worst} "
              f"(W/B {score[k_worst]/score[k_best]:.3f}x, "
              f"ctrl {ctrl_max:.3f}x)")

    (HERE / "scen_op30.json").write_text(json.dumps(scen, indent=2))
    (HERE / "CALIBRATION.md").write_text("\n".join(md) + "\n")
    print("wrote scen_op30.json + CALIBRATION.md")


if __name__ == "__main__":
    main()
