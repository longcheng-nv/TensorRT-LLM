# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op18 nsys pure-kernel runner: one op, cold-L2 flush before each launch, inside
# a cudaProfilerApi window. Run under: nsys profile -c cudaProfilerApi ...
# Usage: python nsys_run.py <base|mt> <K> <dtype> <N> [iters] [M] [R] [acc] [place]
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data  # noqa: E402
from gvr_cutedsl_op import gvr_cutedsl  # noqa: E402
from gvr_mt_op import gvr_mt  # noqa: E402

_DTS = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")


def main():
    which, K, dtn, N = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    M = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    R = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    acc = float(sys.argv[8]) if len(sys.argv) > 8 else 1.5
    place = int(sys.argv[9]) if len(sys.argv) > 9 else 1
    dt = _DTS[dtn]
    b = synth_data.get_bundle(K, dt, N)
    lo, pr = b["logits"].cuda(), b["preIdx"].cuda()
    Npad, crv = b["Npad"], b["cr"]
    sl = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
    out = torch.empty(1, K, dtype=torch.int32, device="cuda")
    if which == "base":
        call = lambda: gvr_cutedsl(lo, pr, sl, K, crv, out=out)
    else:
        call = lambda: gvr_mt(lo, pr, sl, K, crv, out=out, M=M, R=R,
                              accept_mult=acc, place_mode=place)
    for _ in range(20):
        _EVICT.uniform_(0, 1); call()
    torch.cuda.synchronize()
    prof.start()
    for _ in range(iters):
        _EVICT.uniform_(0, 1)
        call()
    torch.cuda.synchronize()
    prof.stop()
    print(f"done {which} K={K} {dtn} N={N} iters={iters} M={M} R={R} acc={acc} place={place}")


if __name__ == "__main__":
    main()
