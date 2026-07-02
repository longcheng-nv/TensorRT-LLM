# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# BS sweep: gvr_mt_auto vs baseline at fixed (K,N), rows replicated
# (report convention: same data across BS). Checks high-BS occupancy impact
# of the extra M*num_threads smem. Usage: bs_sweep.py [K] [N]
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data  # noqa: E402
from gvr_cutedsl_op import gvr_cutedsl  # noqa: E402
from gvr_mt_op import gvr_mt_auto, pick_config  # noqa: E402
from ab_grid import cold_us, exact  # noqa: E402

if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
    cr_val = {512: 4, 1024: 4, 2048: 1}[K]
    b = synth_data.get_bundle(K, torch.float32, N)
    M, R, acc = pick_config(K, N)
    print(f"op18 BS sweep K={K} N={N} fp32 cfg=M{M}R{R}a{acc} — cold-L2 x3-median us")
    print(f"{'BS':>4} | base_us    mt_us  speedup  exact")
    for BS in (1, 4, 8, 16, 32, 64, 128):
        logits = b["logits"].cuda().repeat(BS, 1).contiguous()
        pre = b["preIdx"].cuda().repeat(BS, 1).contiguous()
        seq_lens = torch.full((BS,), b["Npad"] * cr_val, dtype=torch.int32, device="cuda")
        ob = torch.empty(BS, K, dtype=torch.int32, device="cuda")
        om = torch.empty(BS, K, dtype=torch.int32, device="cuda")
        cb = lambda: gvr_cutedsl(logits, pre, seq_lens, K, cr_val, out=ob)
        cm = lambda: gvr_mt_auto(logits, pre, seq_lens, K, cr_val, out=om)
        cb(); cm(); torch.cuda.synchronize()
        ok = exact(om, logits, K) and len(set(om[BS - 1].tolist())) == K
        tb = sorted(cold_us(cb) for _ in range(3))[1]
        tm = sorted(cold_us(cm) for _ in range(3))[1]
        print(f"{BS:>4} | {tb:7.1f}  {tm:7.1f}  {tb/tm:6.3f}x  {'OK' if ok else '**FAIL**'}", flush=True)
