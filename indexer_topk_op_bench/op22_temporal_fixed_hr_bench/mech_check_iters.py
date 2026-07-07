# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 mechanism check — GVR convergence-iteration counts per scenario.

Discriminates artifact-vs-mechanism for the headline `best`-scenario slowdown:
replays the GVR P2-secant / P3-retry-shrink / P4-histogram-snap control flow
(harness/count_gvr_iters.count_iters, a verified host re-implementation of the
vendored GvrTopKKernel) on the ACTUAL op22 bundles and compares iteration
counts across scenarios. Hypothesis: best (beta_deep, hr .90) produces a
tie-dense selection boundary -> P2 non-convergence + P4 snap blowup, while
worst (beta_shallow, hr .05) and real converge normally.

No timing; GPU used only for count reductions. Run on the idle GPU
(CUDA_VISIBLE_DEVICES) while the nsys grid owns the other one.

Output: mech_check_iters.jsonl (one record per cell) + a per-scenario summary.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE))

from count_gvr_iters import count_iters  # noqa: E402
import bundle_data  # noqa: E402

SCENARIOS = ("best", "worst", "real")
KS = (512, 1024, 2048)
DTS = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
NS = (8192, 32768, 131072, 262144, 524288, 1048576)

OUT = HERE / "mech_check_iters.jsonl"


def main():
    recs = []
    with OUT.open("w") as fh:
        for scen in SCENARIOS:
            for K in KS:
                for dt_name, dt in DTS.items():
                    for N in NS:
                        if N <= 2 * K:
                            continue
                        b = bundle_data.get_bundle(scen, K, dt, N)
                        st = count_iters(b["logits"][0], b["preIdx"][0],
                                         N, K, b["cr"], dt)
                        rec = {
                            "scenario": scen, "K": K, "dtype": dt_name,
                            "N": N, "hr": round(b["kernel_hit_rate"], 4),
                            "layer": b["row_meta"].get("layer"),
                            "p2_iters": st.p2_iters,
                            "p2_evals": st.p2_evals,
                            "p2_converged": st.p2_converged,
                            "p4_snap_iters": st.p4_snap_iters,
                            "cand_count": st.cand_count,
                            "branch": st.branch,
                        }
                        recs.append(rec)
                        fh.write(json.dumps(rec) + "\n")
                        print(f"{scen:5s} K={K:4d} {dt_name} N={N:7d} "
                              f"hr={rec['hr']:.3f} L{rec['layer']} "
                              f"p2={st.p2_iters:2d}(ev{st.p2_evals:2d},"
                              f"{'cv' if st.p2_converged else 'GV'}) "
                              f"p4snap={st.p4_snap_iters:4d} "
                              f"cand={st.cand_count:4d} br{st.branch}",
                              flush=True)
        # free bundle cache between scenarios to bound GPU mem
        bundle_data._mem_cache.clear()

    print("\n=== per-scenario summary ===")
    for scen in SCENARIOS:
        rs = [r for r in recs if r["scenario"] == scen]
        n = len(rs)
        p2 = sorted(r["p2_iters"] for r in rs)
        p4 = sorted(r["p4_snap_iters"] for r in rs)
        ncv = sum(not r["p2_converged"] for r in rs)
        brB = sum(r["branch"] == "B" for r in rs)
        print(f"{scen:5s} n={n:3d} p2_iters med={p2[n // 2]:2d} "
              f"max={p2[-1]:2d} nonconv={ncv}/{n} "
              f"p4snap med={p4[n // 2]:4d} max={p4[-1]:4d} branchB={brB}/{n}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
