# [p4tt] K-gate proof: DEFAULT-kwargs kernels vs pristine.
#   K512  default -> gate OFF -> PTX byte-identical to pristine.
#   K1024/K2048 default -> gate ON -> PTX differs; exact; explicit False
#   stays byte-identical.
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from battery_p4tt import (KMOD, KPRI, make_preidx, run_kernel,
                          valueset_exact_rows, _norm_ptx)

n = 65536
fails = 0
for k in (512, 1024, 2048):
    torch.manual_seed(11)
    lg = torch.rand((2, n), dtype=torch.float32, device="cuda")
    pre = make_preidx(lg, k)
    out_p = run_kernel(KPRI, lg, pre, k)                       # pristine
    out_d = run_kernel(KMOD, lg, pre, k)                       # DEFAULT kwargs
    out_f = run_kernel(KMOD, lg, pre, k, p4_tail_fast=False)   # explicit off
    ptx_p = _norm_ptx(list(KPRI._LAUNCH_CACHE.values())[-1])
    caches = list(KMOD._LAUNCH_CACHE.values())
    ptx_f = _norm_ptx(caches[-1])
    ptx_d = _norm_ptx(caches[-2])
    ex = all(valueset_exact_rows(lg, out_d, k) + valueset_exact_rows(lg, out_f, k))
    gate_on = k >= 1024
    ok_d = (ptx_d != ptx_p) if gate_on else (ptx_d == ptx_p)
    ok_f = ptx_f == ptx_p
    tag = "PASS" if (ex and ok_d and ok_f) else "FAIL"
    fails += tag == "FAIL"
    print(f"[{tag}] K={k}: default gate {'ON' if gate_on else 'OFF'} "
          f"ptx_default{'!=' if gate_on else '=='}pristine {ok_d}, "
          f"explicit-False==pristine {ok_f}, exact {ex}")
sys.exit(1 if fails else 0)
