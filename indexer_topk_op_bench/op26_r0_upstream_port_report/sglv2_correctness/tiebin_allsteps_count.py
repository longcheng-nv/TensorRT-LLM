import sys, torch
from pathlib import Path
BENCH = Path("/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench")
sys.path.insert(0, str(BENCH / "harness"))
import real_data_v32 as RV32

def coarse_bin(x, bits):
    hx = x.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    key = torch.where(hx & 0x8000 != 0, (~hx) & 0xFFFF, hx | 0x8000)
    return key >> (16 - bits)

def tie_count(r, K, bits):
    hist = torch.bincount(coarse_bin(r, bits), minlength=1 << bits)
    ac = hist.flip(0).cumsum(0).flip(0); gc = ac - hist
    cand = ((gc < K) & (ac >= K)).nonzero()
    return int(hist[int(cand[-1])].item()) if cand.numel() else 0

for isl in ("128k", "256k"):
    tot = over = 0; overs = []
    for L in RV32.LAYERS_ALL:
        d = RV32._layer_dir(isl, L)
        lg = torch.load(d / "decode.logits.in.pt", map_location="cpu", weights_only=False)
        pk = torch.load(d / "decode.topk.out.pt", map_location="cpu", weights_only=False)
        for st in sorted(lg.keys()):
            Ns = int(pk[st].max()) + 1
            row = lg[st]
            r = (row[0] if row.dim() == 2 else row).float()[:Ns]
            c = tie_count(r, 2048, 10)
            tot += 1
            if c > 2048:
                over += 1; overs.append((L, st, c))
        del lg, pk
    print(f"v32 {isl}: {over}/{tot} (layer,step) cells over cap (10-bit): {overs}")
