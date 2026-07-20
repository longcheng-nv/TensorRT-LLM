# [p4tt] PTX line-count delta: prod2 fast=True vs pristine, plus fast=False
# byte-identity re-check, at the two launch cfgs the battery exercises
# (T=512 small-N and T=1024/cs=4 N=65536-class).
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch

from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as KMOD
from gvrpkgprod2_pristine.top_k.gvr_topk_decode import GvrTopKKernel as KPRI

DEV = "cuda"
CR = 4

def ptx_of(kcls, n, k, **ov):
    lg = torch.randn((1, n), dtype=torch.float32, device=DEV)
    pre = torch.arange(k, dtype=torch.int32, device=DEV).view(1, k).contiguous()
    sl = torch.full((1,), n * CR, dtype=torch.int32, device=DEV)
    out = torch.empty((1, k), dtype=torch.int32, device=DEV)
    kcls.launch(lg, pre, sl, out, k, compress_ratio=CR, **ov)
    torch.cuda.synchronize()
    p = list(kcls._LAUNCH_CACHE.values())[-1].__ptx__
    if isinstance(p, bytes):
        p = p.decode()
    return re.sub(r"kernel_cutlass_gvr_topk_kernel_\w+", "KNAME", p)

for k in (512, 1024, 2048):
    for n in (4096, 65536):
        pf = ptx_of(KMOD, n, k, p4_tail_fast=True)
        ps = ptx_of(KMOD, n, k, p4_tail_fast=False)
        pp = ptx_of(KPRI, n, k)
        lf, ls, lp = (len(x.splitlines()) for x in (pf, ps, pp))
        biteq = "biteq" if ps == pp else "DIFF!"
        print(f"K={k} N={n}: fast={lf} pristine={lp} delta={lf-lp:+d} "
              f"({100.0*(lf-lp)/lp:+.1f}%) | slow-vs-pristine {biteq}")
