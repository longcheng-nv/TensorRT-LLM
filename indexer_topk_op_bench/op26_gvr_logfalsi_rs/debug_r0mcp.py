# Repro: iter7 D2 p4_coop first-silicon failure — fp32 K512 N131072 hr~.
# Signature localization: coop vs p4_rs vs snap on identical data; failure
# shape (dups / -1 pads / value diff) + cs sweep.
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))
sys.path.insert(0, os.path.join(_here, ".."))

import torch

torch.manual_seed(0)
K, N, crv = 512, 131072, 4
logits = torch.randn(1, N, dtype=torch.float32, device="cuda")
row = logits[0].float()
noisy = row + 0.8 * row.std() * torch.randn_like(row)
pre_mid = torch.topk(noisy, K).indices.int().view(1, K).contiguous()
seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
ref = torch.topk(row, K).values


def judge(name, out):
    o = out[0]
    n_neg = int((o < 0).sum())
    idx = o.clamp(min=0).long()
    v = row.gather(0, idx).sort(descending=True).values
    d = (v - ref).abs().max().item()
    nuniq = len(set(o.tolist()))
    oob = int((o >= N).sum())
    print(f"{name:24s} uniq={nuniq}/{K} neg={n_neg} oob={oob} "
          f"valdiff={d:.3e} {'OK' if d == 0.0 and nuniq == K and n_neg == 0 else 'FAIL'}")


from gvr_op26_r0mc_op import gvr_r0_mc_op26  # noqa: E402

judge("coop cs=4", gvr_r0_mc_op26(logits, pre_mid, seq_lens, K, crv,
                                  p4_coop=True))
judge("coop cs=2", gvr_r0_mc_op26(logits, pre_mid, seq_lens, K, crv,
                                  p4_coop=True, cluster_size=2))
judge("p4_rs cs=4", gvr_r0_mc_op26(logits, pre_mid, seq_lens, K, crv,
                                   p4_coop=False, p4_rs=True))
judge("snap cs=4", gvr_r0_mc_op26(logits, pre_mid, seq_lens, K, crv,
                                  p4_coop=False, p4_rs=False))
# hr1 sanity on same row
pre_hit = torch.topk(row, K).indices.int().view(1, K).contiguous()
judge("coop cs=4 hr1", gvr_r0_mc_op26(logits, pre_hit, seq_lens, K, crv,
                                      p4_coop=True))
torch.cuda.synchronize()

# wrong-index slice localization (cs=4 slices of the compressed row)
out = gvr_r0_mc_op26(logits, pre_mid, seq_lens, K, crv, p4_coop=True)
torch.cuda.synchronize()
true_idx = set(torch.topk(row, K).indices.tolist())
wrong = [i for i in out[0].tolist() if i not in true_idx]
print(f"wrong-selected count: {len(wrong)}")
slice_w = N // 4
from collections import Counter
c = Counter(min(3, i // slice_w) for i in wrong)
print("wrong by slice:", dict(sorted(c.items())))
missed = [i for i in true_idx if i not in set(out[0].tolist())]
cm = Counter(min(3, i // slice_w) for i in missed)
print("missed true-topK by slice:", dict(sorted(cm.items())))
