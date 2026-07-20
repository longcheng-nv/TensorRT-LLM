#!/usr/bin/env python3
"""nsys-anchored absolute wall time for prod (gvrpkgprod2) vs timed
(gvrpkgtimed) kernels on the 6 phase-breakdown cells.

CUDA events on this node (umbriel-b200-081) quantize to 2.048us ticks, which
is too coarse for the ~15-30us BS=1 launches — so the trusted absolute comes
from nsys kernel durations (house discipline). Each (cell, arm) section is
wrapped in an NVTX range: 10 warmup then cudaProfilerStart-gated 30 cold-L2
launches (512MB evict zero_ between launches, excluded by kernel-duration
accounting). Run under:
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi ...
Then parse with parse_nsys_anchor.py.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure_phases_prod2 as M  # noqa: E402

REPS = 30


def main():
    sections = []
    for model, isl, layer in M.CELLS:
        RD = M.RV32 if model == "v32" else M.RV4
        b = RD.get_bundle(model, isl, layer, "fp32")
        logits = b["logits"].to(torch.float32).contiguous()
        pre = b["preIdx"].contiguous()
        N, K, cr = b["N"], b["K"], b["cr"]
        seq_lens = torch.full((1,), N * cr, dtype=torch.int32, device="cuda")
        cfg = M.ProdK.pick_config(torch.float32, 1, N, max_seq_len=N * cr)
        out = torch.empty(1, K, dtype=torch.int32, device="cuda")
        ts = torch.zeros(1, 8, dtype=torch.int64, device="cuda")
        prod = M.make_kernel(M.ProdK, K, cr, cfg, timed=False)
        timed = M.make_kernel(M.TimedK, K, cr, cfg, timed=True)
        cell = f"{model}/{isl}/L{layer}"
        sections.append((cell, "prod", lambda p=prod, l=logits, pr=pre, s=seq_lens, o=out: p(l, pr, s, None, o, None)))
        sections.append((cell, "timed", lambda t=timed, l=logits, pr=pre, s=seq_lens, o=out, x=ts: t(l, pr, s, None, o, None, x)))
        print(f"compiled {cell} cfg={cfg}", flush=True)

    torch.cuda.cudart().cudaProfilerStart()
    for cell, arm, call in sections:
        for _ in range(10):
            call()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"SEC|{cell}|{arm}")
        for _ in range(REPS):
            M._EVICT.zero_()
            call()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(f"measured {cell} {arm}", flush=True)
    torch.cuda.cudart().cudaProfilerStop()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
