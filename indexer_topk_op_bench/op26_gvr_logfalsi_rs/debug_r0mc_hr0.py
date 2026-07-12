# Repro: hr0 (bottom-K adversarial preIdx) K512 N32768 fp32 cs=1.
# Three arms on identical data: op26_r0mc (fails) vs vendored cluster anchor
# (envelope question: does fb-less vendored survive?) vs 1cta op26_r0 (fb_fix).
import torch

torch.manual_seed(0)
K, N, crv = 512, 32768, 4
logits = torch.randn(1, N, dtype=torch.float32, device="cuda")
row = logits[0].float()
pre_miss = torch.topk(-row, K).indices.int().view(1, K).contiguous()
seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
ref = torch.topk(row, K).values


def judge(name, out):
    idx = out[0].clamp(min=0).long()
    v = row.gather(0, idx).sort(descending=True).values
    d = (v - ref).abs().max().item()
    nuniq = len(set(out[0].tolist()))
    nneg = int((out[0] < 0).sum())
    print(f"{name:24s} uniq={nuniq}/{K} neg={nneg} valdiff={d:.3e} "
          f"{'OK' if d == 0.0 and nuniq == K and nneg == 0 else 'FAIL'}")


import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))
sys.path.insert(0, os.path.join(_here, ".."))

from gvr_op26_r0mc_op import gvr_r0_mc_op26  # noqa: E402
out1 = gvr_r0_mc_op26(logits, pre_miss, seq_lens, K, crv)
torch.cuda.synchronize()
judge("op26_r0mc", out1)

from gvr_op26_r0_op import gvr_r0_op26  # noqa: E402
out2 = gvr_r0_op26(logits, pre_miss, seq_lens, K, crv)
torch.cuda.synchronize()
judge("op26_r0 (1cta fb_fix)", out2)

# vendored cluster anchor — same import the op22 sweep uses for
# gvr_multicta_cutedsl
from harness.gvr_multicta_cutedsl_op import gvr_multicta_cutedsl  # noqa: E402
out3 = gvr_multicta_cutedsl(logits, pre_miss, seq_lens, K, crv)
torch.cuda.synchronize()
judge("vendored cluster anchor", out3)
