# [p4f1] Silicon attribution: run the v2 flag-ON kernel (with the temporary
# nm printf) on the 25 bench cells using the BENCH ctor config (T=1024,
# 256-bit, mbpm=1, cs = N<65536 ? 1 : 4) and capture the printed need_more.
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/"
                   "TensorRT-LLM/indexer_topk_op_bench/harness")
import torch
import real_data_v4cap as RD4
import real_data_v32 as RDV
from gvrpkgf1.top_k.gvr_topk_decode import GvrTopKKernel as KF1

CELLS = (
    [("flash", 22, isl, RD4, 512, 4) for isl in RD4.ISLS]
    + [("pro", 30, isl, RD4, 1024, 4) for isl in RD4.ISLS]
    + [("v32", 34, isl, RDV, 2048, 1) for isl in RDV.ISLS]
)

for model, L, isl, RD, K, cr in CELLS:
    try:
        b = RD.get_bundle(model, isl, L, "fp32")
    except Exception as e:
        print(f"CELL {model}:{isl}:L{L:02d} LOAD-FAIL {type(e).__name__}",
              flush=True)
        continue
    lg = b["logits"].contiguous()
    pre = b["preIdx"].contiguous()
    N = b["N"]
    cs = 1 if N < 65536 else 4
    sl = torch.full((1,), N * cr, dtype=torch.int32, device="cuda")
    out = torch.empty((1, K), dtype=torch.int32, device="cuda")
    print(f"CELL {model}:{isl}:L{L:02d} N={N} cs={cs}", flush=True)
    KF1.launch(lg, pre, sl, out, K, compress_ratio=cr,
               p4_finebin_loop=True, cluster_size=cs, num_threads=1024,
               use_256bit_load=(lg.data_ptr() % 32 == 0),
               min_blocks_per_mp=1, enable_warp_parallel_reduce=True)
    torch.cuda.synchronize()
    # exactness sanity on the real row
    idx = out[0].long()
    ok = (idx >= 0).all() and idx.unique().numel() == K
    if ok:
        kv = lg[0, :N].gather(0, idx).sort().values
        rv = lg[0, :N].gather(0, b["ref"].long()).sort().values
        ok = torch.equal(kv, rv)
    print(f"  exact={bool(ok)}", flush=True)
