# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op34 BS=1 real-v4cap cold-L2 nsys pure-kernel A/B harness.

Same timing protocol as sweep_op22_v4cap.py (measure_cell: warmup, cold-L2
512MB evict per timed launch, cudaProfilerApi window; cold-L2 canonical). Times
an arbitrary arm set on the real decode bundles at BS=1 fp32, per (model,ISL,
layer). Exactness (tie-aware value-multiset vs same-dtype torch.topk) recorded.

Arm spec = name -> callable factory. Built-in arms:
  sglang_v2            (rival), op26_r0auto (start incumbent), gvr_cutedsl (base),
  op26_r0@kc<X>        (kc_override probe: collect+P4 cand-cap sizing),
  op34_ss              (the new single-scan kernel, when implemented; flag arm).

Run UNDER nsys (drive_nsys_op34.sh) so the .nsys-rep contains only the timed
window. Writes results/<tag>/results_<model>_<isl>.jsonl (append, resumable).
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
OPBENCH = HERE.parents[1]
sys.path.insert(0, str(OPBENCH / "harness"))
sys.path.insert(0, str(OPBENCH / "op26_gvr_logfalsi_rs" / "src"))

from sweep_nsys import measure_cell                       # noqa: E402
from sglang_v2_op import topk_v2, plan as sglv2_plan      # noqa: E402
import real_data_v4cap as RD4                             # noqa: E402
from gvr_op26_r0_op import gvr_r0_op26                    # noqa: E402
from gvr_op26_op import gvr_cutedsl_op26                  # noqa: E402
sys.path.insert(0, str(HERE.parent / "src"))
from op34_mcta_op import mcta_topk, dispatch_C            # noqa: E402

DEV = "cuda"


def _seq(N, cr, BS=1):
    return torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)


def build_arm(arm, K, N, cr, logits_row, preidx_row):
    """Return (call, keep, out) for a given arm on a BS=1 real bundle.
    logits_row [1,Npad] fp32; preidx_row [1,K] int32. out = keep[-1]."""
    lg = logits_row                          # already [1,Npad] fp32 cuda
    pre = preidx_row
    out = torch.empty((1, K), dtype=torch.int32, device=DEV)
    if arm == "sglang_v2":
        sl = torch.full((1,), N, dtype=torch.int32, device=DEV)
        md = sglv2_plan(sl)
        torch.cuda.synchronize()
        topk_v2(lg, sl, K, out=out, metadata=md, max_seq_len=N)  # warm
        return (lambda: topk_v2(lg, sl, K, out=out, metadata=md,
                                max_seq_len=N)), [lg, sl, md, out], out
    if arm == "gvr_cutedsl":
        sl = _seq(N, cr)
        return (lambda: gvr_cutedsl_op26(lg, pre, sl, K, cr, out=out)), \
               [lg, pre, sl, out], out
    if arm == "op26_r0auto":
        sl = _seq(N, cr)
        return (lambda: gvr_r0_op26(lg, pre, sl, K, cr, out=out)), \
               [lg, pre, sl, out], out
    if arm == "op34_mcta" or arm.startswith("op34_mcta@C"):
        C = int(arm.split("@C")[1]) if "@C" in arm else dispatch_C(N)
        mcta_topk(lg, pre, N, K, out, C=C)   # warm/compile
        return (lambda: mcta_topk(lg, pre, N, K, out, C=C)), \
               [lg, pre, out], out
    if arm in ("op34_mcta_oracle", "op34_collect_only", "op34_collect_oracle"):
        C = dispatch_C(N)
        # precompute the oracle threshold OUTSIDE the timed loop (UB cheat)
        t_or = float(torch.topk(lg[0, :N], K).values.min().item())
        co = arm.startswith("op34_collect")
        t_ov = t_or if arm.endswith("oracle") else None
        mode = "oracle" if arm.endswith("oracle") else "hint"
        mcta_topk(lg, pre, N, K, out, C=C, mode=mode, collect_only=co,
                  t_override=t_ov)
        return (lambda: mcta_topk(lg, pre, N, K, out, C=C, mode=mode,
                                  collect_only=co, t_override=t_ov)), \
               [lg, pre, out], out
    if arm.startswith("op26_r0@kc"):
        kc = int(arm.split("kc")[1])
        sl = _seq(N, cr)
        return (lambda: gvr_r0_op26(lg, pre, sl, K, cr, out=out,
                                    kc_override=kc)), [lg, pre, sl, out], out
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RD4.MODELS))
    ap.add_argument("--isl", required=True, choices=list(RD4.ISLS))
    ap.add_argument("--arms", required=True, help="comma arm list")
    ap.add_argument("--layers", default=None, help="comma layer subset")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    args = ap.parse_args()

    m = RD4.MODELS[args.model]
    K, cr = m["K"], m["cr"]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else m["layers"])
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{args.model}_{args.isl}.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            try:
                r = json.loads(ln)
                done.add((r["arm"], r["layer"]))
            except Exception:
                pass
    f = open(out_path, "a")
    print(f"# op34 nsys {args.model}/{args.isl} K={K} arms={arms} "
          f"layers={len(layers)}", flush=True)
    prof.start()
    try:
        for L in layers:
            b = RD4.get_bundle(args.model, args.isl, L, "fp32")
            lg, pre, N = b["logits"], b["preIdx"], b["N"]
            for arm in arms:
                if (arm, L) in done:
                    continue
                base = f"{arm}|{args.model}|{args.isl}|L{L}|{N}"
                rec = {"sweep": "op34", "arm": arm, "model": args.model,
                       "isl": args.isl, "K": K, "N": N, "layer": L, "BS": 1,
                       "hit_rate": round(b["hit_rate"], 4),
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}"}
                try:
                    call, keep, out = build_arm(arm, K, N, cr, lg, pre)
                    call()
                    torch.cuda.synchronize()
                    vd, rc, nn = RD4.value_metrics(out[0], lg, b["ref"], K)
                    rec.update(vdiff=vd, recall=round(rc, 5), n_neg=nn)
                    measure_cell(call, base, args.reps, args.reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            print(f"[op34 {args.model}/{args.isl}] L{L} done", flush=True)
    finally:
        prof.stop()
    f.close()
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
