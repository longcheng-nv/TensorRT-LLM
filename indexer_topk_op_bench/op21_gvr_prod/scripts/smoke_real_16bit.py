#!/usr/bin/env python3
"""op21 iter8 16-bit gate: real-capture exactness for gvr_ms + gvr_msc
C {4,8} on bf16/fp16 (dtype-truncated captures, per-dtype refs, tie-robust
value check — real_data_v2 contract). 60 layers x 3 ops x 2 dtypes."""
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parent / "src"))
import real_data_v2  # noqa: E402
from real_data_v2 import value_metrics  # noqa: E402
from gvr_ms_op import gvr_ms  # noqa: E402
from gvr_msc_op import gvr_msc  # noqa: E402

ok = bad = 0
for dt in ("bf16", "fp16"):
    for model, layers in (("pro", range(2, 61, 2)), ("flash", range(2, 43, 2)),
                          ("v32", (0, 1, 20, 21, 22, 40, 41, 42, 60))):
        for L in layers:
            b = real_data_v2.get_real_bundle_v2(model, L, dt)
            K, cr, N = b["K"], b["cr"], b["N"]
            lg = b["logits"][:, :].contiguous()
            sl = torch.tensor([N * cr if cr > 1 else N], dtype=torch.int32,
                              device="cuda")
            for name, fn in (
                    ("ms", lambda: gvr_ms(lg, b["preIdx"], sl, K, cr)),
                    ("C4", lambda: gvr_msc(lg, b["preIdx"], sl, K, cr, C=4)),
                    ("C8", lambda: gvr_msc(lg, b["preIdx"], sl, K, cr, C=8))):
                out = fn()
                torch.cuda.synchronize()
                vd, rc, nn = value_metrics(out, lg[:, :N].float(), b["ref"], K)
                u = torch.unique(out[0][out[0] >= 0]).numel()
                good = (vd == 0 and nn == 0 and u == K)
                ok += good; bad += not good
                if not good:
                    print(f"FAIL {dt} {name} {model} L{L}: vdiff={vd:.2e} "
                          f"recall={rc:.4f} nneg={nn} uniq={u}")
print(f"real 16-bit: {ok} ok / {bad} fail")
