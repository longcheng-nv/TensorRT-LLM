# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op20 priority-tier bench: gvr_x (op20 kernel under optimization) vs
#   base  = single-CTA gvr_cutedsl   (GVR structural baseline)
#   rival = radix_cutedsl (auto)     (the op to beat, measured IN-RUN)
# Cold-L2 CUDA-graph event median (harness/sweep.py protocol), report synth
# bundles (seed=42), exactness on every row.
#
# Tiers (user priority 2026-07-03):
#   tier1 = fp32 K{512,1024}   (highest priority)
#   tier2 = fp32 K2048
#   tier3 = bf16/fp16 all K
#
# usage: python3 tier_bench.py --tier 1 [--BSs 1,4,16,64,256,1024]
#                              [--reps 40] [--out results/x.jsonl] [--gpu 0]
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
from radix_cutedsl_op import radix_cutedsl  # noqa: E402
from gvr_x_op import gvr_sw_auto as gvr_x_auto  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

TIERS = {
    1: [("fp32", K) for K in (512, 1024)],
    2: [("fp32", 2048)],
    3: [(dt, K) for dt in ("bf16", "fp16") for K in (512, 1024, 2048)],
}
NS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
CRMAP = {512: 4, 1024: 4, 2048: 1}


def cold_us(call, reps=40, warmup=5):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for _ in range(10):
        g.replay()
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


def run(K, dtn, N, BS, args):
    cr = CRMAP[K]
    b = synth_data.get_bundle(K, _DTYPES[dtn], N)
    logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
    pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
    Npad = b["Npad"]
    seq_cr = torch.full((BS,), Npad * cr, dtype=torch.int32, device=DEV)
    seq_nod = torch.full((BS,), Npad, dtype=torch.int32, device=DEV)
    ob = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ox = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    orv = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    cb = lambda: gvr_cutedsl(logits, pre, seq_cr, K, cr, out=ob)
    cx = lambda: gvr_x_auto(logits, pre, seq_cr, K, cr, out=ox)
    cv = lambda: radix_cutedsl(logits, seq_nod, K, out=orv)
    cb(); cx(); cv(); torch.cuda.synchronize()
    ok = exact_all(ox, logits, K)
    tb = cold_us(cb, reps=args.reps)
    tx = cold_us(cx, reps=args.reps)
    tv = cold_us(cv, reps=args.reps)
    del logits, pre, ob, ox, orv
    torch.cuda.empty_cache()
    return tb, tx, tv, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--BSs", default="1,4,16,64,256,1024")
    ap.add_argument("--Ns", default=",".join(map(str, NS)))
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    BSs = [int(x) for x in args.BSs.split(",")]
    Ns = [int(x) for x in args.Ns.split(",")]
    # resume: skip (dtype,K,N,BS) keys already in the out JSONL (node-loss
    # recovery; done rows still feed the summary)
    done = {}
    if args.out and Path(args.out).exists():
        for line in open(args.out):
            try:
                r = json.loads(line)
                done[(r["dtype"], r["K"], r["N"], r["BS"])] = r
            except ValueError:
                pass
    fout = open(args.out, "a") if args.out else None
    import math
    sx, sv, nwin = [], [], 0
    print(f"op20 tier{args.tier} {args.label} — gvr_x vs base vs radix_cutedsl "
          f"(cold-L2 median us, in-run)")
    print(f"{'dt':>5} {'K':>5} {'N':>7} {'BS':>5} | {'base':>7} {'gvr_x':>7} "
          f"{'radix':>7} | x/base rvl/x  exact")
    for dtn, K in TIERS[args.tier]:
        for N in Ns:
            if N <= 2 * K:
                continue
            for BS in BSs:
                d = done.get((dtn, K, N, BS))
                if d:
                    tb, tx, tv, ok = d["base_us"], d["x_us"], d["rival_us"], d["exact"]
                    sx.append(tb / tx); sv.append(tv / tx); nwin += tv / tx >= 1.0
                    continue
                tb, tx, tv, ok = run(K, dtn, N, BS, args)
                f = "OK" if ok else "**FAIL**"
                sx.append(tb / tx); sv.append(tv / tx); nwin += tv / tx >= 1.0
                print(f"{dtn:>5} {K:>5} {N:>7} {BS:>5} | {tb:7.1f} {tx:7.1f} "
                      f"{tv:7.1f} | {tb/tx:6.3f} {tv/tx:6.3f}  {f}", flush=True)
                if fout:
                    fout.write(json.dumps(dict(
                        tier=args.tier, label=args.label, dtype=dtn, K=K, N=N,
                        BS=BS, base_us=tb, x_us=tx, rival_us=tv, exact=ok)) + "\n")
                    fout.flush()
    gm = lambda a: math.exp(sum(math.log(v) for v in a) / len(a))
    print(f"\nSUMMARY n={len(sx)}: x/base gm={gm(sx):.3f}  rival/x gm={gm(sv):.3f} "
          f"min={min(sv):.3f} | fastest-vs-rival {nwin}/{len(sv)} "
          f"({100*nwin/len(sv):.0f}%)")
    if fout:
        fout.close()


if __name__ == "__main__":
    main()
