# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op18 A/B: single-CTA multi-threshold (gvr_mt) vs single-CTA baseline
# gvr_cutedsl. Cold-L2 CUDA-graph event median (harness/sweep.py protocol),
# report synth data (seed=42), exactness checked per cell.
#
# usage: python3 ab_grid.py [--M 4] [--R 2] [--acc 2.0] [--place 0]
#                           [--threads 0(auto)] [--dtype fp32] [--reps 40]
import argparse
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

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def cold_us(call, reps=40, warmup=5):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup): call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): call()
    for _ in range(10): g.replay()
    torch.cuda.synchronize()
    cold = []
    for _ in range(reps):
        _EVICT.uniform_(0, 1); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)
    cold.sort(); del g
    return cold[len(cold) // 2]


def exact(out, logits, K):
    idx = out[0].clamp(min=0).long()
    v = logits[0].float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits[0].float(), K).values
    return (v - ref).abs().max().item() == 0.0 and len(set(out[0].tolist())) == K


def run(K, dtype, N, cr_val, args):
    b = synth_data.get_bundle(K, dtype, N)
    logits, pre = b["logits"].to(DEV), b["preIdx"].to(DEV)
    Npad = b["Npad"]
    seq_lens = torch.full((1,), Npad * cr_val, dtype=torch.int32, device=DEV)
    ob = torch.empty(1, K, dtype=torch.int32, device=DEV)
    om = torch.empty(1, K, dtype=torch.int32, device=DEV)
    thr = None if args.threads == 0 else args.threads
    cb = lambda: gvr_cutedsl(logits, pre, seq_lens, K, cr_val, out=ob)
    cm = lambda: gvr_mt(logits, pre, seq_lens, K, cr_val, out=om, M=args.M, R=args.R,
                        accept_mult=args.acc, place_mode=args.place, threads=thr)
    cb(); cm(); torch.cuda.synchronize()
    ok = exact(om, logits, K)
    tb = cold_us(cb, reps=args.reps); tm = cold_us(cm, reps=args.reps)
    return tb, tm, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--acc", type=float, default=2.0)
    ap.add_argument("--place", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--Ks", default="512,1024,2048")
    ap.add_argument("--Ns", default="4096,8192,16384,32768,65536,131072,262144")
    args = ap.parse_args()
    dtype = _DTYPES[args.dtype]
    crmap = {512: 4, 1024: 4, 2048: 1}
    Ks = [int(x) for x in args.Ks.split(",")]
    Ns = [int(x) for x in args.Ns.split(",")]
    print(f"op18 gvr_mt vs gvr_cutedsl — {args.dtype} M={args.M} R={args.R} "
          f"acc={args.acc} place={args.place} threads={args.threads or 'auto'} — cold-L2 median us")
    print(f"{'K':>5} {'N':>8} | base_us    mt_us  speedup  exact")
    sp = []
    for K in Ks:
        for N in Ns:
            if N <= 2 * K:
                continue
            tb, tm, ok = run(K, dtype, N, crmap[K], args)
            flag = "OK" if ok else "**FAIL**"
            sp.append(tb / tm)
            print(f"{K:>5} {N:>8} | {tb:7.1f}  {tm:7.1f}  {tb/tm:6.3f}x  {flag}", flush=True)
    import statistics
    print(f"cells={len(sp)} min={min(sp):.3f} avg={statistics.mean(sp):.3f} max={max(sp):.3f}")
