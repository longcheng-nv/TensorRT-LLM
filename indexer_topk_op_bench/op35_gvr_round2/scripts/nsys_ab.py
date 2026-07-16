# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op35 L2 ship arbiter — nsys paired A/B of base (PR HEAD) vs variant flags.

Per cell: NVTX 'c|<cell>|base' and 'c|<cell>|var' cold ranges (512MB evict
outside), REPS reps each, interleaved. Run 3 independent batches (x3 nsys reps)
for the ship verdict; single-GPU, paired same-process.

Usage: nsys_ab.py <shard_i> <shard_n> <var_flags_json> [family]
"""
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from ab_op35 import iter_cells, load_cell                        # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel as BaseK   # noqa: E402
from gvrpkg35.top_k.gvr_topk_decode import GvrTopKKernel as VarK  # noqa: E402

DEV = "cuda"
REPS = 10


def main():
    sh_i, sh_n = int(sys.argv[1]), int(sys.argv[2])
    vflags = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in json.loads(sys.argv[3]).items()}
    fam = sys.argv[4] if len(sys.argv) > 4 else "all"
    want = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else None
    evict = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
    prof.start()
    for ci, cell in enumerate(iter_cells(fam)):
        if ci % sh_n != sh_i:
            continue
        cid, lg_row, pre, K, cr, N = load_cell(cell)
        if want and cid not in want:
            continue
        lg = lg_row.unsqueeze(0).contiguous().to(DEV)
        pre = pre[:1].contiguous().to(DEV)
        sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        outi = torch.empty(1, K, dtype=torch.int32, device=DEV)
        vf = {k: v for k, v in vflags.items() if not k.endswith('_k2048')}
        if K == 2048:
            for k, v in vflags.items():
                if k.endswith('_k2048'):
                    vf[k[:-6]] = v
        calls = {
            "base": (lambda: BaseK.launch(lg, pre, sl, outi, K, compress_ratio=cr)),
            "var": (lambda: VarK.launch(lg, pre, sl, outi, K, compress_ratio=cr, **vf)),
        }
        for arm, call in calls.items():
            call()
        torch.cuda.synchronize()
        for rnd in (1, 2, 3):
            for r in range(REPS):
                for arm, call in calls.items():
                    evict.uniform_()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_push(f"c|{cid}|{arm}|r{rnd}")
                    call()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
        print(f"done {cid}", flush=True)
        del lg, pre, sl, outi
        torch.cuda.empty_cache()
    prof.stop()


if __name__ == "__main__":
    main()
