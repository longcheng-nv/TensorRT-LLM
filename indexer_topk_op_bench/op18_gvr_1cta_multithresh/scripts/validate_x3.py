# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Final dispatch validation: gvr_mt_auto vs gvr_cutedsl, x3-median of cold-L2
# event medians, per dtype. Writes JSONL: one record per (dtype, K, N).
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
from gvr_mt_op import gvr_mt_auto, pick_config  # noqa: E402
from ab_grid import cold_us, exact  # noqa: E402

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="fp32,bf16,fp16")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default=str(_HERE.parent / "results" / "validate_x3.jsonl"))
    args = ap.parse_args()
    dts = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    crmap = {512: 4, 1024: 4, 2048: 1}
    fh = open(args.out, "a")
    for dtn in args.dtypes.split(","):
        dtype = dts[dtn]
        for K in (512, 1024, 2048):
            for N in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
                if N <= 2 * K:
                    continue
                cr_val = crmap[K]
                b = synth_data.get_bundle(K, dtype, N)
                logits, pre = b["logits"].to(DEV), b["preIdx"].to(DEV)
                seq_lens = torch.full((1,), b["Npad"] * cr_val, dtype=torch.int32, device=DEV)
                ob = torch.empty(1, K, dtype=torch.int32, device=DEV)
                om = torch.empty(1, K, dtype=torch.int32, device=DEV)
                cb = lambda: gvr_cutedsl(logits, pre, seq_lens, K, cr_val, out=ob)
                cm = lambda: gvr_mt_auto(logits, pre, seq_lens, K, cr_val, out=om)
                cb(); cm(); torch.cuda.synchronize()
                ok = exact(om, logits, K)
                tbs, tms = [], []
                for _ in range(3):
                    tbs.append(cold_us(cb, reps=args.reps))
                    tms.append(cold_us(cm, reps=args.reps))
                tb = sorted(tbs)[1]; tm = sorted(tms)[1]
                M, R, acc = pick_config(K, N)
                rec = {"dtype": dtn, "K": K, "N": N, "cfg": f"M{M}R{R}a{acc}",
                       "base_us": tb, "mt_us": tm, "speedup": tb / tm, "exact": ok,
                       "base_x3": tbs, "mt_x3": tms}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"{dtn} K={K:>4} N={N:>7} {rec['cfg']:>9}: base={tb:6.1f} mt={tm:6.1f} "
                      f"{tb/tm:6.3f}x {'OK' if ok else '**FAIL**'}", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
