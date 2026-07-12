# Repro: K2048 N262144 fp32 cs=4 hr1 duplicate-index failure (uniq 1950/2048).
# Four arms on identical data to localize: iter6b new code vs iter5 mc port
# vs vendored anchor.
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))
sys.path.insert(0, os.path.join(_here, ".."))

import torch

torch.manual_seed(0)
K, N, crv = 2048, 262144, 1
# same data as smoke: seed 0 but smoke burns draws before this cell; use
# fresh draws — failure should not be seed-specific. Verify below.
logits = torch.randn(1, N, dtype=torch.float32, device="cuda")
row = logits[0].float()
pre_hit = torch.topk(row, K).indices.int().view(1, K).contiguous()
seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
ref = torch.topk(row, K).values


def judge(name, out):
    idx = out[0].clamp(min=0).long()
    v = row.gather(0, idx).sort(descending=True).values
    d = (v - ref).abs().max().item()
    nuniq = len(set(out[0].tolist()))
    print(f"{name:28s} uniq={nuniq}/{K} valdiff={d:.3e} "
          f"{'OK' if d == 0.0 and nuniq == K else 'FAIL'}")


from gvr_op26_r0mc_op import gvr_r0_mc_op26  # noqa: E402
judge("op26_r0mc cs=auto(4)", gvr_r0_mc_op26(logits, pre_hit, seq_lens, K, crv))
judge("op26_r0mc cs=2",
      gvr_r0_mc_op26(logits, pre_hit, seq_lens, K, crv, cluster_size=2))
judge("op26_r0mc cs=1",
      gvr_r0_mc_op26(logits, pre_hit, seq_lens, K, crv, cluster_size=1))

from gvr_op26_op import gvr_multicta_op26  # noqa: E402
judge("op26_mc (iter5)", gvr_multicta_op26(logits, pre_hit, seq_lens, K, crv))

from harness.gvr_multicta_cutedsl_op import gvr_multicta_cutedsl  # noqa: E402
judge("vendored anchor", gvr_multicta_cutedsl(logits, pre_hit, seq_lens, K, crv))

from gvr_op26_r0_op import gvr_r0_op26  # noqa: E402
judge("op26_r0 (1cta)", gvr_r0_op26(logits, pre_hit, seq_lens, K, crv))

# hr~ and hr0(random-disjoint) on the same row for the failing arm
noisy = row + 0.8 * row.std() * torch.randn_like(row)
pre_mid = torch.topk(noisy, K).indices.int().view(1, K).contiguous()
judge("op26_r0mc hr~ cs=4", gvr_r0_mc_op26(logits, pre_mid, seq_lens, K, crv))
topk_idx = torch.topk(row, 2 * K).indices
mask = torch.ones(N, dtype=torch.bool)
mask[topk_idx.cpu()] = False
rest = torch.arange(N)[mask]
pre_miss = rest[torch.randperm(rest.numel())[:K]].int().cuda().view(1, K).contiguous()
judge("op26_r0mc hr0 cs=4", gvr_r0_mc_op26(logits, pre_miss, seq_lens, K, crv))
torch.cuda.synchronize()
