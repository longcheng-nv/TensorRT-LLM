# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op19 config sweep: baseline measured ONCE per (K,dtype,N,BS) cell, then each
# sandwich config. Cold-L2 CUDA-graph median, exactness per config. Resumable
# (skips (cell,cfg) pairs already in the jsonl).
#
# usage: python3 config_sweep.py --BSs 1,16 --configs "M4R2p4b64,M6R2p4b32" \
#            [--dtype fp32] [--reps 20] [--out results/config_sweep.jsonl]
import argparse
import json
import re
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
from ab_grid import cold_us, exact_all  # noqa: E402

DEV = "cuda"
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
_CFG_RE = re.compile(r"M(\d+)R(\d+)p(\d+)(?:b(\d+))?$")


def parse_cfg(s):
    m = _CFG_RE.match(s)
    if not m:
        raise ValueError(f"bad config {s} (want e.g. M4R1p4 or M6R2p4b32)")
    return dict(M=int(m.group(1)), R=int(m.group(2)),
                place_mode=int(m.group(3)),
                band_acc=int(m.group(4)) if m.group(4) else 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--Ks", default="512,1024,2048")
    ap.add_argument("--Ns", default="4096,8192,16384,32768,65536,131072,262144")
    ap.add_argument("--BSs", default="1")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dtype = _DTYPES[args.dtype]
    crmap = {512: 4, 1024: 4, 2048: 1}
    cfgs = [(s, parse_cfg(s)) for s in args.configs.split(",")]
    out_path = Path(args.out) if args.out else (
        _HERE.parent / "results" / f"config_sweep_{args.dtype}.jsonl")
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r["K"], r["dtype"], r["N"], r["BS"], r["cfg"]))
            except (json.JSONDecodeError, KeyError):
                pass
    fout = open(out_path, "a")
    sp_by_cfg = {s: [] for s, _ in cfgs}
    print(f"op19 config sweep {args.dtype} reps={args.reps} -> {out_path}")
    for K in (int(x) for x in args.Ks.split(",")):
        for N in (int(x) for x in args.Ns.split(",")):
            if N <= 2 * K:
                continue
            for BS in (int(x) for x in args.BSs.split(",")):
                todo = [(s, c) for s, c in cfgs
                        if (K, args.dtype, N, BS, s) not in done]
                if not todo:
                    continue
                cr_val = crmap[K]
                b = synth_data.get_bundle(K, dtype, N)
                logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
                pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
                seq = torch.full((BS,), b["Npad"] * cr_val, dtype=torch.int32,
                                 device=DEV)
                ob = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                om = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                cb = lambda: gvr_cutedsl(logits, pre, seq, K, cr_val, out=ob)
                cb(); torch.cuda.synchronize()
                tb = cold_us(cb, reps=args.reps)
                for s, c in todo:
                    cm = lambda: gvr_sw(logits, pre, seq, K, cr_val, out=om,
                                        M=c["M"], R=c["R"],
                                        band_acc=c["band_acc"],
                                        place_mode=c["place_mode"])
                    try:
                        cm(); torch.cuda.synchronize()
                        ok = exact_all(om, logits, K)
                        tm = cold_us(cm, reps=args.reps)
                    except Exception as e:  # compile/launch failure -> record
                        print(f"{K:>5} {N:>7} {BS:>5} {s:>12}: ERROR {e}",
                              flush=True)
                        fout.write(json.dumps(dict(
                            K=K, dtype=args.dtype, N=N, BS=BS, cfg=s,
                            error=str(e)[:200])) + "\n")
                        fout.flush()
                        continue
                    spd = tb / tm
                    sp_by_cfg[s].append(spd)
                    flag = "OK" if ok else "**FAIL**"
                    print(f"{K:>5} {N:>7} {BS:>5} {s:>12}: base={tb:8.1f} "
                          f"sw={tm:8.1f} {spd:6.3f}x {flag}", flush=True)
                    fout.write(json.dumps(dict(
                        K=K, dtype=args.dtype, N=N, BS=BS, cfg=s, base_us=tb,
                        sw_us=tm, speedup=spd, exact=ok)) + "\n")
                    fout.flush()
                del logits, pre, ob, om
                torch.cuda.empty_cache()
    for s, v in sp_by_cfg.items():
        if v:
            print(f"{s}: n={len(v)} min={min(v):.3f} "
                  f"gm={statistics.geometric_mean(v):.3f} max={max(v):.3f}")
    fout.close()


if __name__ == "__main__":
    main()
