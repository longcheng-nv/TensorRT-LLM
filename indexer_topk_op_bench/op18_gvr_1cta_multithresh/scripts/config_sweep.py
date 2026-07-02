# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op18 config sweep: time baseline ONCE per cell, then each (M,R,acc,place,thr)
# config; emit JSONL for dispatch-table construction. Cold-L2 event median.
import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data  # noqa: E402
from gvr_cutedsl_op import gvr_cutedsl  # noqa: E402
from gvr_mt_op import gvr_mt  # noqa: E402
from ab_grid import cold_us, exact  # noqa: E402

DEV = "cuda"

# (name, M, R, acc_mult, place_mode, threads[0=auto])
CONFIGS = [
    ("M4R1u", 4, 1, 1.0, 0, 0),
    ("M4R1dy", 4, 1, 1.0, 1, 0),
    ("M4R2u_a2", 4, 2, 2.0, 0, 0),
    ("M4R2dy_a15", 4, 2, 1.5, 1, 0),
    ("M6R2dy_a15", 6, 2, 1.5, 1, 0),
    ("M8R2dy_a125", 8, 2, 1.25, 1, 0),
    ("M8R1dy", 8, 1, 1.0, 1, 0),
    ("M2R1u", 2, 1, 1.0, 0, 0),
    ("M4R2pm_a15", 4, 2, 1.5, 2, 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--Ns", default="4096,8192,16384,32768,65536,131072,262144")
    ap.add_argument("--out", default=str(_HERE.parent / "results" / "config_sweep_fp32.jsonl"))
    ap.add_argument("--configs", default="")  # comma names filter
    args = ap.parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    crmap = {512: 4, 1024: 4, 2048: 1}
    Ns = [int(x) for x in args.Ns.split(",")]
    cfgs = CONFIGS if not args.configs else [c for c in CONFIGS if c[0] in args.configs.split(",")]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(outp, "a")
    for K in (512, 1024, 2048):
        for N in Ns:
            if N <= 2 * K:
                continue
            cr_val = crmap[K]
            b = synth_data.get_bundle(K, dtype, N)
            logits, pre = b["logits"].to(DEV), b["preIdx"].to(DEV)
            seq_lens = torch.full((1,), b["Npad"] * cr_val, dtype=torch.int32, device=DEV)
            ob = torch.empty(1, K, dtype=torch.int32, device=DEV)
            cb = lambda: gvr_cutedsl(logits, pre, seq_lens, K, cr_val, out=ob)
            cb(); torch.cuda.synchronize()
            tb = cold_us(cb, reps=args.reps)
            print(f"K={K} N={N} base={tb:.1f}us", flush=True)
            rec = {"K": K, "N": N, "dtype": args.dtype, "base_us": tb}
            for name, M, R, acc, pl, thr in cfgs:
                om = torch.empty(1, K, dtype=torch.int32, device=DEV)
                cm = lambda: gvr_mt(logits, pre, seq_lens, K, cr_val, out=om, M=M, R=R,
                                    accept_mult=acc, place_mode=pl,
                                    threads=(None if thr == 0 else thr))
                cm(); torch.cuda.synchronize()
                ok = exact(om, logits, K)
                tm = cold_us(cm, reps=args.reps)
                rec[name] = {"us": tm, "speedup": tb / tm, "exact": ok}
                print(f"   {name:>14}: {tm:6.1f}us  {tb/tm:6.3f}x  {'OK' if ok else '**FAIL**'}", flush=True)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
    fh.close()


if __name__ == "__main__":
    main()
