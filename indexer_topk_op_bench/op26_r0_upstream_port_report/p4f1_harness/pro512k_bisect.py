#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""pro/512k regression bisect: 4 kernel builds same-process, paired, x3
rounds, on the regression cell (pro/512k bench L30) + control cells
(pro/256k, pro/1024k, flash/512k). Arms:
  snap  = @018251950f pre-vseed (gvrpkg_snapshot)
  vseed = @88a563b145 vseed + per-K rung defaults
  ptail = @eae374554c + p4_exact_tail
  head  = @0d6fc4f1f2 + K2048 rung recalib
Usage (under nsys): --out-root DIR
"""
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
_REPORT = _HERE.parent
_BENCH = _REPORT.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPORT / "gvrpkg_snapshot"))
sys.path.insert(0, str(_BENCH / "harness"))

from sweep_nsys import measure_cell                                    # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel as SnapK        # noqa: E402
from gvrpkgvseed.top_k.gvr_topk_decode import GvrTopKKernel as VseedK  # noqa: E402
from gvrpkgprod.top_k.gvr_topk_decode import GvrTopKKernel as PtailK   # noqa: E402
from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as HeadK   # noqa: E402
import real_data_v4cap as RV4                                          # noqa: E402

ARMS = [("snap", SnapK), ("vseed", VseedK), ("ptail", PtailK), ("head", HeadK)]
CELLS = [("pro", "512k", 30), ("pro", "256k", 30), ("pro", "1024k", 30),
         ("flash", "512k", 22)]


def main():
    out = Path(sys.argv[sys.argv.index("--out-root") + 1])
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / "bisect.jsonl", "w")
    prof.start()
    try:
        for model, isl, L in CELLS:
            bd = RV4.get_bundle(model, isl, L, "fp32")
            K, N, cr = bd["K"], bd["N"], bd["cr"]
            lg = bd["logits"].contiguous()
            pre = bd["preIdx"].contiguous()
            sl = torch.full((1,), N * cr, dtype=torch.int32, device="cuda")
            outb = torch.empty(1, K, dtype=torch.int32, device="cuda")
            for rnd in range(3):
                for arm, KK in ARMS:
                    base = f"{arm}|{model}|{isl}|{N}|r{rnd}"
                    call = (lambda lg=lg, pre=pre, sl=sl, outb=outb, KK=KK:
                            KK.launch(lg, pre, sl, outb, K, compress_ratio=cr))
                    call()
                    torch.cuda.synchronize()
                    idx = outb[0].long()
                    v = lg[0, :N].float()
                    ex = bool(idx.unique().numel() == K and torch.equal(
                        v[idx].sort().values, torch.topk(v, K).values.sort().values))
                    measure_cell(call, base, 20, 50)
                    f.write(json.dumps(dict(model=model, isl=isl, N=N, arm=arm,
                                            rnd=rnd, exact=ex, hit=bd["hit_rate"],
                                            range_cold=f"c|{base}")) + "\n")
                    f.flush()
            print(f"{model}/{isl} done", flush=True)
    finally:
        prof.stop()
    f.close()


if __name__ == "__main__":
    main()
