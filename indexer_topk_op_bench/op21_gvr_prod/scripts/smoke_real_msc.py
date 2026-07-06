#!/usr/bin/env python3
"""op21 cluster gate: real-capture exactness for gvr_msc, C in {2,4,8}
(pro 30L + flash 21L + v32 9L = 60 layers x 3 C = 180 checks), plus the
adversarial preIdx gate (random + half-invalid) on gvr_ms and gvr_msc."""
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data, real_data_v2  # noqa: E402
from real_data_v2 import value_metrics  # noqa: E402
from gvr_ms_op import gvr_ms  # noqa: E402
from gvr_msc_op import gvr_msc  # noqa: E402

ok = bad = 0
for C in (2, 4, 8):
    for model, layers in (("pro", range(2, 61, 2)), ("flash", range(2, 43, 2)),
                          ("v32", (0, 1, 20, 21, 22, 40, 41, 42, 60))):
        for L in layers:
            b = real_data_v2.get_real_bundle_v2(model, L, "fp32")
            K, cr, N = b["K"], b["cr"], b["N"]
            lg = b["logits"][:, :].contiguous()
            sl = torch.tensor([N * cr if cr > 1 else N], dtype=torch.int32,
                              device="cuda")
            out = gvr_msc(lg, b["preIdx"], sl, K, compress_ratio=cr, C=C)
            torch.cuda.synchronize()
            vd, rc, nn = value_metrics(out, lg[:, :N].float(), b["ref"], K)
            u = torch.unique(out[0][out[0] >= 0]).numel()
            good = (vd == 0 and nn == 0 and u == K)
            ok += good; bad += not good
            if not good:
                print(f"FAIL real C{C} {model} L{L}: vdiff={vd:.2e} "
                      f"recall={rc:.4f} nneg={nn} uniq={u}")
print(f"real x C: {ok} ok / {bad} fail")

# ---- adversarial preIdx: random + half-invalid, gvr_ms + gvr_msc C4/C8 ----
ok_a = bad_a = 0
torch.manual_seed(0)
for K, crv in ((512, 4), (1024, 4), (2048, 1)):
    for N in (65536, 262144):
        b = synth_data.get_bundle(K, torch.float32, N)
        lg = b["logits"].cuda()
        Npad = b["Npad"]
        sl = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
        pre_rand = torch.randint(0, N, (1, K), dtype=torch.int32,
                                 device="cuda")
        pre_half = pre_rand.clone()
        pre_half[0, ::2] = -1
        for tag, pre in (("rand", pre_rand), ("half", pre_half)):
            for name, fn in (("ms", lambda: gvr_ms(lg, pre, sl, K, crv)),
                             ("C4", lambda: gvr_msc(lg, pre, sl, K, crv, C=4)),
                             ("C8", lambda: gvr_msc(lg, pre, sl, K, crv, C=8))):
                out = fn()
                torch.cuda.synchronize()
                idx = out[0].clamp(min=0).long()
                v = lg[0].float().gather(0, idx).sort(descending=True).values
                ref = torch.topk(lg[0].float(), K).values
                d = (v - ref).abs().max().item()
                nu = len(set(out[0].tolist()))
                good = (d == 0.0 and nu == K)
                ok_a += good; bad_a += not good
                if not good:
                    print(f"FAIL adv {tag} {name} K{K} N{N}: "
                          f"valdiff={d:.2e} uniq={nu}/{K}")
print(f"adversarial: {ok_a} ok / {bad_a} fail")
