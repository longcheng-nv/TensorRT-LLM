# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""nsys re-confirmation of ab_bigbs_runnercfg.py (big-BS 3-arm triage).

Same 3 arms / 20 cells, but timed via the report's canonical protocol:
NVTX cold-L2 ranges (measure_cell) inside ONE nsys process, pure-kernel
sum per range (nvtx_kern_sum, evict-filtered). Run under:
  nsys profile -t cuda,nvtx -o <rep> python3 bigbs_nsys.py <out.jsonl>
then parse with the same parse_rep as every prior report number.
"""
import json
import os
import sys

os.environ.setdefault("SYNTH_POSITIONAL", "1")
RD = "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench"
sys.path.insert(0, f"{RD}/op22_temporal_fixed_hr_bench")
sys.path.insert(0, f"{RD}/harness")
sys.path.insert(0, f"{RD}/op26_r0_upstream_port_report/gvrpkg_snapshot")
sys.path.insert(0, f"{RD}/op26_r0_upstream_port_report/harness")

import torch  # noqa: E402
import bundle_data_env as B  # noqa: E402
from sweep_nsys import build_call, measure_cell  # noqa: E402

# arm builders from the CUDA-event triage (same file => identical configs)
sys.path.insert(0, f"{RD}/op26_r0_upstream_port_report")
from ab_bigbs_runnercfg import pr_call, runner_policy, _valid  # noqa: E402

DEV = "cuda"
DTN = {torch.float32: "fp32", torch.bfloat16: "bf16"}

CELLS = [
    (512, N, "best", dt, bs)
    for dt in (torch.float32, torch.bfloat16)
    for N in (16384, 65536, 131072)
    for bs in (64, 256, 1024)
] + [(1024, 65536, "best", torch.bfloat16, 1024),
     (1024, 131072, "worst", torch.float32, 1024)]

REPS_COLD, REPS_WARM = 30, 3


def main():
    out_path = sys.argv[1]
    f = open(out_path, "a")
    for K, N, scen, tdt, BS in CELLS:
        b = B.get_bundle(scen, K, torch.float32, N)
        lg_row = b["logits"].to(tdt).contiguous()
        pre_row = b["preIdx"].contiguous()
        cr = b["cr"]; Np = lg_row.shape[1]
        lg = lg_row.expand(BS, -1).contiguous()
        pre = pre_row.expand(BS, -1).contiguous()
        sl = torch.full((BS,), Np * cr, dtype=torch.int32, device=DEV)
        dtn = DTN[tdt]

        fro = dict(cluster_size=1 if N < 65536 else 4, num_threads=1024,
                   use_256bit_load=True, min_blocks_per_mp=1,
                   enable_warp_parallel_reduce=True)
        run = runner_policy(tdt, BS, Np)
        for tag, cfg in (("pr_frozen", fro), ("pr_runner", run)):
            base = f"{tag}|{K}|{dtn}|{N}|{scen}|{BS}"
            out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
            key = (tag, tdt, K, cr) + tuple(sorted(cfg.items()))
            call = pr_call(key, cfg, tdt, K, cr, lg, pre, sl, out)
            call(); torch.cuda.synchronize()
            ex = _valid(out[0], lg[0], K, N)
            measure_cell(call, base, REPS_COLD, REPS_WARM)
            f.write(json.dumps({"arm": tag, "K": K, "dtype": dtn, "N": N,
                                "scen": scen, "BS": BS, "exact": ex,
                                "cfg": {k: v for k, v in cfg.items()},
                                "range_cold": f"c|{base}"}) + "\n")
            f.flush()
            del call, out
        base = f"op26|{K}|{dtn}|{N}|{scen}|{BS}"
        call26, keep, extra = build_call("op26_r0auto", K, tdt, N, BS,
                                         cr, lg_row, pre_row)
        call26(); torch.cuda.synchronize()
        out26 = next(t for t in reversed(keep)
                     if torch.is_tensor(t) and t.dtype == torch.int32
                     and t.dim() == 2 and t.shape[-1] == K)
        ex26 = _valid(out26[0], keep[0][0], K, N)
        measure_cell(call26, base, REPS_COLD, REPS_WARM)
        f.write(json.dumps({"arm": "op26", "K": K, "dtype": dtn, "N": N,
                            "scen": scen, "BS": BS, "exact": ex26,
                            "r0_arm": extra.get("r0_arm", "?"),
                            "range_cold": f"c|{base}"}) + "\n")
        f.flush()
        del call26, keep
        torch.cuda.empty_cache()
        print(f"done {K} {dtn} {N} {scen} BS{BS}", file=sys.stderr, flush=True)
    f.close()


if __name__ == "__main__":
    main()
