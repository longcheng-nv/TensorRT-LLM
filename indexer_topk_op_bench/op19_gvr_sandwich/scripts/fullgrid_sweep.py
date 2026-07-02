# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op19 FULL report-grid sweep: gvr_sw_auto (dispatch) vs gvr_cutedsl over the
# 720-cell B200 grid (K x dtype x N x BS). Cold-L2 CUDA-graph median,
# exactness per cell, resumable via jsonl.
#
# usage: python3 fullgrid_sweep.py [--reps 20] [--dtypes fp32,bf16,fp16]
#            [--BSs 1,...,2048] [--out results/fullgrid_b200.jsonl]
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_BENCH / "op17_gvr_portfolio" / "src"))
import synth_data  # noqa: E402
from gvr_cutedsl_op import gvr_cutedsl  # noqa: E402
from gvr_sw_op import gvr_sw_auto  # noqa: E402
from ab_grid import cold_us, exact_all  # noqa: E402

DEV = "cuda"
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--dtypes", default="fp32,bf16,fp16")
    ap.add_argument("--Ks", default="512,1024,2048")
    ap.add_argument("--Ns", default="4096,8192,16384,32768,65536,131072,262144")
    ap.add_argument("--BSs", default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    crmap = {512: 4, 1024: 4, 2048: 1}
    out_path = Path(args.out) if args.out else (
        _HERE.parent / "results" / "fullgrid_b200.jsonl")
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r["K"], r["dtype"], r["N"], r["BS"]))
            except (json.JSONDecodeError, KeyError):
                pass
    fout = open(out_path, "a")
    sp = []
    for dtn in args.dtypes.split(","):
        dtype = _DTYPES[dtn]
        for K in (int(x) for x in args.Ks.split(",")):
            for N in (int(x) for x in args.Ns.split(",")):
                if N <= 2 * K:
                    continue
                for BS in (int(x) for x in args.BSs.split(",")):
                    if (K, dtn, N, BS) in done:
                        continue
                    cr_val = crmap[K]
                    b = synth_data.get_bundle(K, dtype, N)
                    logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
                    pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
                    seq = torch.full((BS,), b["Npad"] * cr_val,
                                     dtype=torch.int32, device=DEV)
                    ob = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                    om = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                    cb = lambda: gvr_cutedsl(logits, pre, seq, K, cr_val, out=ob)
                    cm = lambda: gvr_sw_auto(logits, pre, seq, K, cr_val, out=om)
                    try:
                        cb(); cm(); torch.cuda.synchronize()
                        ok = exact_all(om, logits, K)
                        tb = cold_us(cb, reps=args.reps)
                        tm = cold_us(cm, reps=args.reps)
                    except Exception as e:
                        print(f"{K:>5} {dtn} {N:>7} {BS:>5}: ERROR {e}", flush=True)
                        fout.write(json.dumps(dict(K=K, dtype=dtn, N=N, BS=BS,
                                                   error=str(e)[:200])) + "\n")
                        fout.flush()
                        del logits, pre, ob, om
                        torch.cuda.empty_cache()
                        continue
                    spd = tb / tm
                    sp.append(spd)
                    print(f"{K:>5} {dtn} {N:>7} {BS:>5}: base={tb:8.1f} "
                          f"sw={tm:8.1f} {spd:6.3f}x "
                          f"{'OK' if ok else '**FAIL**'}", flush=True)
                    fout.write(json.dumps(dict(K=K, dtype=dtn, N=N, BS=BS,
                                               base_us=tb, sw_us=tm,
                                               speedup=spd, exact=ok)) + "\n")
                    fout.flush()
                    del logits, pre, ob, om
                    torch.cuda.empty_cache()
    if sp:
        print(f"\ncells={len(sp)} gm={statistics.geometric_mean(sp):.3f} "
              f"avg={statistics.mean(sp):.3f} min={min(sp):.3f} max={max(sp):.3f}")
    fout.close()
