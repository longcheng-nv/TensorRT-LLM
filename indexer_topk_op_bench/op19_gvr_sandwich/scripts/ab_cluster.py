# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op19 cluster-sandwich A/B: gvr_swc (Strategy-B) vs gvr_cutedsl baseline AND
# vs op17 gvr_portfolio_cluster (same G). Cold-L2 CUDA-graph median.
#
# usage: python3 ab_cluster.py [--G 8|auto] [--dtype fp32] [--BSs 1,4,16]
#                              [--reps 20] [--out results/ab_cluster.jsonl]
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
from gvr_swc_op import gvr_swc, pick_G  # noqa: E402
from gvr_portfolio_cluster_op import gvr_portfolio_cluster  # noqa: E402
from ab_grid import cold_us, exact_all  # noqa: E402

DEV = "cuda"
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", default="auto")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--Ks", default="512,1024,2048")
    ap.add_argument("--Ns", default="4096,8192,16384,32768,65536,131072,262144")
    ap.add_argument("--BSs", default="1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dtype = _DTYPES[args.dtype]
    crmap = {512: 4, 1024: 4, 2048: 1}
    fout = open(args.out, "a") if args.out else None
    print(f"op19 gvr_swc vs (gvr_cutedsl, op17 portfolio) — {args.dtype} "
          f"G={args.G} — cold-L2 median us")
    print(f"{'K':>5} {'N':>8} {'BS':>4} {'G':>3} | base_us  op17_us   swc_us | "
          f"swc/base swc/op17  exact")
    sb, so = [], []
    for K in (int(x) for x in args.Ks.split(",")):
        for N in (int(x) for x in args.Ns.split(",")):
            if N <= 2 * K:
                continue
            for BS in (int(x) for x in args.BSs.split(",")):
                G = pick_G(BS) if args.G == "auto" else int(args.G)
                if G < 2:
                    continue
                cr_val = crmap[K]
                b = synth_data.get_bundle(K, dtype, N)
                logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
                pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
                seq = torch.full((BS,), b["Npad"] * cr_val, dtype=torch.int32, device=DEV)
                ob = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                op17o = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                oc = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                cb = lambda: gvr_cutedsl(logits, pre, seq, K, cr_val, out=ob)
                c17 = lambda: gvr_portfolio_cluster(logits, pre, seq, K, cr_val, out=op17o, G=G)
                cc = lambda: gvr_swc(logits, pre, seq, K, cr_val, out=oc, G=G, use_push=False)
                cb(); c17(); cc(); torch.cuda.synchronize()
                ok = exact_all(oc, logits, K)
                tb = cold_us(cb, reps=args.reps)
                t17 = cold_us(c17, reps=args.reps)
                tc = cold_us(cc, reps=args.reps)
                sb.append(tb / tc); so.append(t17 / tc)
                flag = "OK" if ok else "**FAIL**"
                print(f"{K:>5} {N:>8} {BS:>4} {G:>3} | {tb:7.1f} {t17:8.1f} "
                      f"{tc:8.1f} | {tb/tc:7.3f}x {t17/tc:7.3f}x  {flag}",
                      flush=True)
                if fout:
                    fout.write(json.dumps(dict(
                        K=K, dtype=args.dtype, N=N, BS=BS, G=G, base_us=tb,
                        op17_us=t17, swc_us=tc, vs_base=tb / tc,
                        vs_op17=t17 / tc, exact=ok)) + "\n")
                    fout.flush()
                del logits, pre, ob, op17o, oc
                torch.cuda.empty_cache()
    print(f"vs base: gm={statistics.geometric_mean(sb):.3f} min={min(sb):.3f} max={max(sb):.3f}")
    print(f"vs op17: gm={statistics.geometric_mean(so):.3f} min={min(so):.3f} max={max(so):.3f}")
    if fout:
        fout.close()
