#!/usr/bin/env python3
"""op21 iter1 smoke: exactness of gvr_ms on synth (x seeds) + ALL real captures."""
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data, real_data_v2
from real_data_v2 import value_metrics
from gvr_ms_op import gvr_ms

def run_case(logits, pre, K, cr, N):
    bs = logits.shape[0]
    sl = torch.full((bs,), N * cr if cr > 1 else N, dtype=torch.int32, device="cuda")
    out = gvr_ms(logits, pre, sl, K, compress_ratio=cr)
    torch.cuda.synchronize()
    return out

bad = ok = 0
# ---- synth: K x N x BS x seeds ----
for K in (512, 1024, 2048):
    for N in (8192, 65536, 262144):
        if N <= 2 * K: continue
        for BS in (1, 16):
            for seed in (42, 7, 1234):
                b = synth_data.get_bundle(K, torch.float32, N, seed=seed)
                lg = b["logits"][:1].repeat(BS, 1).contiguous()
                pre = b["preIdx"][:1].repeat(BS, 1).contiguous()
                out = run_case(lg, pre, K, b["cr"], N)
                ref = torch.topk(lg[0, :N].float(), K).indices
                vd, rc, nn = value_metrics(out[:1], lg[:1, :N].float().unsqueeze(0)[0], ref, K)
                u = torch.unique(out[0][out[0] >= 0]).numel()
                good = (vd == 0 and nn == 0 and u == K)
                ok += good; bad += not good
                if not good:
                    print(f"FAIL synth K{K} N{N} BS{BS} s{seed}: vdiff={vd} nneg={nn} uniq={u}")
print(f"synth: {ok} ok / {bad} fail")

# ---- real captures: all pro(30) + flash(21) + v32(9) layers ----
ok_r = bad_r = 0
for model, layers in (("pro", range(2, 61, 2)), ("flash", range(2, 43, 2)),
                      ("v32", (0, 1, 20, 21, 22, 40, 41, 42, 60))):
    for L in layers:
        b = real_data_v2.get_real_bundle_v2(model, L, "fp32")
        K, cr, N = b["K"], b["cr"], b["N"]
        lg = b["logits"][:, :].contiguous()   # [1, Npad]
        sl = torch.tensor([N * cr if cr > 1 else N], dtype=torch.int32, device="cuda")
        out = gvr_ms(lg, b["preIdx"], sl, K, compress_ratio=cr)
        torch.cuda.synchronize()
        vd, rc, nn = value_metrics(out, lg[:, :N].float(), b["ref"], K)
        u = torch.unique(out[0][out[0] >= 0]).numel()
        good = (vd == 0 and nn == 0 and u == K)
        ok_r += good; bad_r += not good
        if not good:
            print(f"FAIL real {model} L{L}: vdiff={vd:.2e} recall={rc:.4f} nneg={nn} uniq={u} (hit={b['hit_rate']:.2f})")
print(f"real: {ok_r} ok / {bad_r} fail")
