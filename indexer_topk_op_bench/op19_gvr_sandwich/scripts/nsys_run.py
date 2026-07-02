# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op19 nsys pure-kernel runner: one op, cold-L2 flush before each launch,
# inside a cudaProfilerApi window. Run under: nsys profile -c cudaProfilerApi
# Usage: python nsys_run.py <base|auto> <K> <dtype> <N> <BS> [iters]
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_BENCH / "op17_gvr_portfolio" / "src"))
import synth_data  # noqa: E402
from gvr_cutedsl_op import gvr_cutedsl  # noqa: E402
from gvr_sw_op import gvr_sw_auto  # noqa: E402

_DTS = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
_CR = {512: 4, 1024: 4, 2048: 1}


def main():
    which, K, dtn, N, BS = (sys.argv[1], int(sys.argv[2]), sys.argv[3],
                            int(sys.argv[4]), int(sys.argv[5]))
    iters = int(sys.argv[6]) if len(sys.argv) > 6 else 60
    dt = _DTS[dtn]
    crv = _CR[K]
    b = synth_data.get_bundle(K, dt, N)
    lo = b["logits"].cuda().expand(BS, -1).contiguous()
    pr = b["preIdx"].cuda().expand(BS, -1).contiguous()
    sl = torch.full((BS,), b["Npad"] * crv, dtype=torch.int32, device="cuda")
    out = torch.empty(BS, K, dtype=torch.int32, device="cuda")
    if which == "base":
        call = lambda: gvr_cutedsl(lo, pr, sl, K, crv, out=out)
    else:
        call = lambda: gvr_sw_auto(lo, pr, sl, K, crv, out=out)
    for _ in range(20):
        _EVICT.uniform_(0, 1); call()
    torch.cuda.synchronize()
    prof.start()
    for _ in range(iters):
        _EVICT.uniform_(0, 1)
        call()
    torch.cuda.synchronize()
    prof.stop()
    print(f"done {which} K={K} {dtn} N={N} BS={BS} iters={iters}")


if __name__ == "__main__":
    main()
