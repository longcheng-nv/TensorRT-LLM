# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch-floor probe (feasibility prior): nsys-timed floors on this node.

Arms (20 cold-L2 NVTX'd launches each):
  fill    — trivial torch fill kernel (raw kernel-execution floor)
  gvr_id  — GVR baseline degenerate path N<=K (kernel prologue + identity emit)
  gvr_min — GVR baseline smallest real shape class (K512, N=1024, randn)

  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
      --capture-range-end=stop -f true -o results/floor/floor \
      python3 scripts/probe_floor.py
"""
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
OP40 = HERE.parent
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(HERE))

from ab40 import compile_arm, launch_cfg  # noqa: E402

DEV = "cuda"
REPS = 20
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)


def timed(tag, fire):
    fire()
    torch.cuda.synchronize()
    for _ in range(5):
        fire()
    torch.cuda.synchronize()
    for _ in range(REPS):
        _EVICT.random_()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"c|{tag}|{tag}|fp32")
        fire()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    print(f"  {tag} done", flush=True)


def main():
    prof.start()
    buf = torch.empty(1024, device=DEV)
    timed("fill", lambda: buf.fill_(1.0))

    K, cr = 512, 4
    g = torch.Generator(device=DEV).manual_seed(4242)
    # degenerate: N <= top_k -> identity emit path
    Nd = 384
    lg_d = torch.randn(1, Nd, generator=g, device=DEV)
    pre_d = torch.arange(K, dtype=torch.int32, device=DEV).reshape(1, K).contiguous()
    sl_d = torch.full((1,), Nd * cr, dtype=torch.int32, device=DEV)
    cfg = launch_cfg(lg_d, Nd)
    fn = compile_arm("base", K, cr, cfg)
    oi = torch.empty(1, K, dtype=torch.int32, device=DEV)
    timed("gvr_id", lambda: fn(lg_d, pre_d, sl_d, None, oi, None))

    # smallest real class: K512 N=1024
    Nm = 1024
    lg_m = torch.randn(1, Nm, generator=g, device=DEV)
    noisy = lg_m[0] + 0.5 * torch.randn(Nm, generator=g, device=DEV)
    pre_m = torch.topk(noisy, K).indices.to(torch.int32).reshape(1, K).contiguous()
    sl_m = torch.full((1,), Nm * cr, dtype=torch.int32, device=DEV)
    cfg_m = launch_cfg(lg_m, Nm)
    fn_m = compile_arm("base", K, cr, cfg_m)
    timed("gvr_min", lambda: fn_m(lg_m, pre_m, sl_m, None, oi, None))
    prof.stop()


if __name__ == "__main__":
    main()
