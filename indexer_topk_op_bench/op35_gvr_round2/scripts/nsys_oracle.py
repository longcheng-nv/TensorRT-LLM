# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op35 nsys oracle attribution sweep — L2 measurement of the phase blocks.

Arms per cell (all launched in one process; NVTX 'c|<cell>|<arm>' cold ranges,
512MB L2 evict OUTSIDE the range before each rep):
  base    : variant kernel, no flags   (== PR HEAD; variant pkg for jit parity)
  p3      : p3_oracle_frac=0.001       (P3 collect scan removed)
  p4      : p4_oracle_skip=True        (handoff2 + P4 + writeback removed)
  floor   : floor_oracle=True          (P1 + launch + identity emit only)

Run under: nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop
Usage: nsys_oracle.py <shard_i> <shard_n> <out_jsonl_prefix> [--family all]
Kernel times filled by scripts/parse_oracle.py from the .nsys-rep.
"""
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from ab_op35 import iter_cells, load_cell   # noqa: E402
from gvrpkg35.top_k.gvr_topk_decode import GvrTopKKernel as VarK  # noqa: E402

DEV = "cuda"
ARMS = {"base": {}, "p3": {"p3_oracle_frac": 0.001},
        "p4": {"p4_oracle_skip": True}, "floor": {"floor_oracle": True}}
REPS = 15


def main():
    sh_i, sh_n = int(sys.argv[1]), int(sys.argv[2])
    out = open(sys.argv[3], "a")
    fam = sys.argv[4] if len(sys.argv) > 4 else "all"
    evict = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
    cells = list(iter_cells(fam))
    prof.start()
    for ci, cell in enumerate(cells):
        if ci % sh_n != sh_i:
            continue
        cid, lg_row, pre, K, cr, N = load_cell(cell)
        lg = lg_row.unsqueeze(0).contiguous().to(DEV)
        pre = pre[:1].contiguous().to(DEV)
        sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        outi = torch.empty(1, K, dtype=torch.int32, device=DEV)
        for arm, flags in ARMS.items():
            call = lambda: VarK.launch(lg, pre, sl, outi, K, compress_ratio=cr, **flags)
            call()
            torch.cuda.synchronize()
            for r in range(REPS):
                evict.uniform_()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"c|{cid}|{arm}")
                call()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
            out.write(json.dumps(dict(cell=cid, arm=arm, N=N, K=K)) + "\n")
            out.flush()
        print(f"done {cid}", flush=True)
        del lg, pre, sl, outi
        torch.cuda.empty_cache()
    prof.stop()


if __name__ == "__main__":
    main()
