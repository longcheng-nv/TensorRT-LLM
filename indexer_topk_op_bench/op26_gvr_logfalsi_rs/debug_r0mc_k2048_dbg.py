# Single failing cell with kernel debug printfs (OP26_R0MC_DEBUG=1).
# Also a passing cell (N=131072) for contrast.
import os
import sys

os.environ["OP26_R0MC_DEBUG"] = "1"
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))

import torch

torch.manual_seed(0)
from gvr_op26_r0mc_op import gvr_r0_mc_op26  # noqa: E402

K, crv = 2048, 1
for N in (131072, 262144):
    logits = torch.randn(1, N, dtype=torch.float32, device="cuda")
    row = logits[0].float()
    pre_hit = torch.topk(row, K).indices.int().view(1, K).contiguous()
    seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
    print(f"---- N={N} cs=1 hr1 ----", flush=True)
    out = gvr_r0_mc_op26(logits, pre_hit, seq_lens, K, crv, cluster_size=1)
    torch.cuda.synchronize()
    idx = out[0].clamp(min=0).long()
    v = row.gather(0, idx).sort(descending=True).values
    ref = torch.topk(row, K).values
    d = (v - ref).abs().max().item()
    nuniq = len(set(out[0].tolist()))
    print(f"verdict N={N}: uniq={nuniq}/{K} valdiff={d:.3e}", flush=True)
