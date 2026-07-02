# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op19 A/B: single-CTA sandwich (gvr_sw) vs single-CTA baseline gvr_cutedsl.
# Cold-L2 CUDA-graph event median (harness/sweep.py protocol), report synth
# data (seed=42), exactness checked per cell. Supports the BS axis (rows are
# replicated copies, matching the report grid method).
#
# usage: python3 ab_grid.py [--M 4] [--R 2] [--bacc 64] [--place 3]
#                           [--dtype fp32] [--reps 40] [--BSs 1]
#                           [--Ks ...] [--Ns ...] [--out results/x.jsonl]
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
import synth_data  # noqa: E402
from gvr_cutedsl_op import gvr_cutedsl  # noqa: E402
from gvr_sw_op import gvr_sw  # noqa: E402

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


def exact_all(out, logits, K):
    """Exactness on EVERY row (sandwich direct-write must hold at any BS)."""
    bs = out.shape[0]
    lf = logits.float()
    ref = torch.topk(lf, K, dim=1).values
    idx = out.clamp(min=0).long()
    v = lf.gather(1, idx).sort(dim=1, descending=True).values
    if (v - ref).abs().max().item() != 0.0:
        return False
    for r in range(bs):
        if len(set(out[r].tolist())) != K:
            return False
    return True


def run(K, dtype, N, BS, cr_val, args):
    b = synth_data.get_bundle(K, dtype, N)
    logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
    pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
    Npad = b["Npad"]
    seq_lens = torch.full((BS,), Npad * cr_val, dtype=torch.int32, device=DEV)
    ob = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    om = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    thr = None if args.threads == 0 else args.threads
    cb = lambda: gvr_cutedsl(logits, pre, seq_lens, K, cr_val, out=ob)
    cm = lambda: gvr_sw(logits, pre, seq_lens, K, cr_val, out=om, M=args.M,
                        R=args.R, band_acc=args.bacc, place_mode=args.place,
                        threads=thr)
    cb(); cm(); torch.cuda.synchronize()
    ok = exact_all(om, logits, K)
    tb = cold_us(cb, reps=args.reps); tm = cold_us(cm, reps=args.reps)
    del logits, pre, ob, om
    torch.cuda.empty_cache()
    return tb, tm, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--bacc", type=int, default=64)
    ap.add_argument("--place", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--Ks", default="512,1024,2048")
    ap.add_argument("--Ns", default="4096,8192,16384,32768,65536,131072,262144")
    ap.add_argument("--BSs", default="1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dtype = _DTYPES[args.dtype]
    crmap = {512: 4, 1024: 4, 2048: 1}
    Ks = [int(x) for x in args.Ks.split(",")]
    Ns = [int(x) for x in args.Ns.split(",")]
    BSs = [int(x) for x in args.BSs.split(",")]
    cfg = f"M{args.M}R{args.R}b{args.bacc}p{args.place}"
    fout = open(args.out, "a") if args.out else None
    print(f"op19 gvr_sw vs gvr_cutedsl — {args.dtype} {cfg} "
          f"threads={args.threads or 'auto'} — cold-L2 median us")
    print(f"{'K':>5} {'N':>8} {'BS':>5} | base_us    sw_us  speedup  exact")
    sp = []
    for K in Ks:
        for N in Ns:
            if N <= 2 * K:
                continue
            for BS in BSs:
                tb, tm, ok = run(K, dtype, N, BS, crmap[K], args)
                flag = "OK" if ok else "**FAIL**"
                sp.append(tb / tm)
                print(f"{K:>5} {N:>8} {BS:>5} | {tb:8.1f} {tm:8.1f}  "
                      f"{tb/tm:6.3f}x  {flag}", flush=True)
                if fout:
                    fout.write(json.dumps(dict(
                        K=K, dtype=args.dtype, N=N, BS=BS, cfg=cfg,
                        base_us=tb, sw_us=tm, speedup=tb / tm, exact=ok)) + "\n")
                    fout.flush()
    print(f"cells={len(sp)} min={min(sp):.3f} avg={statistics.mean(sp):.3f} "
          f"gm={statistics.geometric_mean(sp):.3f} max={max(sp):.3f}")
    if fout:
        fout.close()
