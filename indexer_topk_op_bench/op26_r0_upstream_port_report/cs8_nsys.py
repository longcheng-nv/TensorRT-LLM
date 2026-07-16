# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""nsys spot: is pick_config's cs=8 pick (BS<=4, N>=131072) sane with R0?

3 arms per cell, all through the NEW GvrTopKKernel.launch:
  auto  : no override -> pick_config decides (cs=8 at these cells)
  cs4   : cluster_size=4 forced (the pre-existing single-wave pick)
  op26  : op-bench op26_r0auto anchor (its mc arm resolves cs=4 at BS<=16)

op22-§env synth best, fp32, BS in {1,4}. Run under nsys (cuda,nvtx); parse
NVTX c| ranges with parse_nsys_full.parse_rep.
Usage: nsys profile -t cuda,nvtx -o <rep> python3 cs8_nsys.py <out.jsonl>
"""
import json
import os
import sys

os.environ.setdefault("SYNTH_POSITIONAL", "1")
RD = "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench"
sys.path.insert(0, "/tmp/gvrval1/pickcfg")          # gvrpkg with MODIFIED kernel
sys.path.insert(0, f"{RD}/op22_temporal_fixed_hr_bench")
sys.path.insert(0, f"{RD}/harness")

import torch  # noqa: E402
import bundle_data_env as B  # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
from sweep_nsys import build_call, measure_cell  # noqa: E402

DEV = "cuda"
CELLS = [(512, 131072), (1024, 131072), (1024, 262144), (2048, 131072)]
REPS_COLD, REPS_WARM = 30, 3


def main():
    f = open(sys.argv[1], "a")
    for K, N in CELLS:
        b = B.get_bundle("best", K, torch.float32, N)
        lg_row = b["logits"].contiguous()
        pre_row = b["preIdx"].contiguous()
        cr = b["cr"]; Np = lg_row.shape[1]
        for BS in (1, 4):
            lg = lg_row.expand(BS, -1).contiguous()
            pre = pre_row.expand(BS, -1).contiguous()
            sl = torch.full((BS,), Np * cr, dtype=torch.int32, device=DEV)
            for arm, ovr in (("auto", {}), ("cs4", {"cluster_size": 4})):
                out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                call = (lambda lg=lg, pre=pre, sl=sl, out=out, ovr=ovr:
                        GvrTopKKernel.launch(lg, pre, sl, out, K,
                                             compress_ratio=cr, **ovr))
                call(); torch.cuda.synchronize()
                cs = GvrTopKKernel.pick_config(torch.float32, BS, Np)["cluster_size"] \
                    if not ovr else ovr["cluster_size"]
                base = f"{arm}|{K}|{N}|{BS}"
                measure_cell(call, base, REPS_COLD, REPS_WARM)
                f.write(json.dumps({"arm": arm, "K": K, "N": N, "BS": BS,
                                    "cs": cs, "range_cold": f"c|{base}"}) + "\n")
                f.flush()
            base = f"op26|{K}|{N}|{BS}"
            call26, keep, extra = build_call("op26_r0auto", K, torch.float32,
                                             N, BS, cr, lg_row, pre_row)
            call26(); torch.cuda.synchronize()
            measure_cell(call26, base, REPS_COLD, REPS_WARM)
            f.write(json.dumps({"arm": "op26", "K": K, "N": N, "BS": BS,
                                "cs": extra.get("r0_arm", "?"),
                                "range_cold": f"c|{base}"}) + "\n")
            f.flush()
            print(f"done K{K} N{N} BS{BS}", file=sys.stderr, flush=True)
    f.close()


if __name__ == "__main__":
    main()
