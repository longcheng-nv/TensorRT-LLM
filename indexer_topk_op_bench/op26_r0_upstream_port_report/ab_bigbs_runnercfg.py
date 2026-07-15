# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""BIG-BS TRIAGE: how much of the report-§7 large-BS PR-vs-op26 gap is the
harness's FROZEN kernel config (cs by N only, T=1024, mbpm=1) vs a genuine
algorithmic difference?

Three arms on the same op22-§env synthetic inputs replicated to BS rows:
  pr_frozen : GvrTopKKernel exactly as harness/perf_3arm_bs.py builds it
              (cs = N>=65536 ? 4 : 1, num_threads=1024, min_blocks_per_mp=1,
              use_256bit_load=True, enable_r0=True)  -> reproduces REPORT §7.
  pr_runner : SAME kernel, but cluster_size + tuning picked by a faithful
              replica of CuteDSLGvrTopKDecodeRunner.forward + _pick_tuning
              (the PR's actual production dispatch, untouched by the PR).
  op26      : op-bench op26_r0auto via build_call (report anchor arm).

CUDA-event cold-L2 (256MB L2 flush before each timed call), same-process
back-to-back => ratios are fair (op34 lesson: event tax hits all arms alike).
"""
import math
import os
import statistics
import sys

os.environ.setdefault("SYNTH_POSITIONAL", "1")
RD = "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench"
sys.path.insert(0, f"{RD}/op22_temporal_fixed_hr_bench")
sys.path.insert(0, f"{RD}/harness")
sys.path.insert(0, f"{RD}/op26_r0_upstream_port_report/gvrpkg_snapshot")
sys.path.insert(0, f"{RD}/op26_r0_upstream_port_report/harness")

import torch  # noqa: E402
import cutlass  # noqa: E402
import bundle_data_env as B  # noqa: E402
from exact import compile_kernel  # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
from sweep_nsys import build_call  # noqa: E402

DEV = "cuda"
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_EVICT = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=DEV)
_CDT = {torch.float32: cutlass.Float32, torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16}


def runner_policy(tdt, num_rows, N_row):
    """Faithful replica of CuteDSLGvrTopKDecodeRunner.forward cluster policy
    + _pick_tuning (single-CTA/sort path, max_seq_len=None, no LB)."""
    if N_row < 65536:
        cs = 1
    elif num_rows <= 4 and N_row >= 131072:
        cs = 8
    elif num_rows * 4 <= NUM_SMS:
        cs = 4
    elif num_rows * 2 <= NUM_SMS:
        cs = 2
    else:
        cs = 1
    n_per_cta = N_row // cs
    T = 1024 if (num_rows <= NUM_SMS and n_per_cta >= 65536) else 512
    v256 = tdt == torch.float32 and n_per_cta >= 16384
    wpr = T == 1024
    vec_w = (256 if v256 else 128) // (32 if tdt == torch.float32 else 16)
    n_vec_iters = max(1, n_per_cta // (T * vec_w))
    if tdt == torch.float32:
        if n_vec_iters < 4:
            mb = 0
        elif num_rows <= NUM_SMS:
            mb = 1
        elif NUM_SMS * 2 < num_rows <= NUM_SMS * 3 and n_per_cta <= 32768:
            mb = 3
        else:
            mb = 2
    else:
        if num_rows > NUM_SMS:
            mb = 3
        elif n_vec_iters < 4:
            mb = 0
        else:
            mb = 1
    return dict(cluster_size=cs, num_threads=T, use_256bit_load=v256,
                min_blocks_per_mp=mb, enable_warp_parallel_reduce=wpr)


_KCACHE = {}


def pr_call(cfg_key, cfg, tdt, K, cr, lg, pre, sl, out):
    f = _KCACHE.get(cfg_key)
    if f is None:
        k = GvrTopKKernel(dtype=_CDT[tdt], top_k=K, next_n=1,
                          compress_ratio=cr, return_output_values=False,
                          enable_r0=True, **cfg)
        f = compile_kernel(k, True)
        _KCACHE[cfg_key] = f
    return lambda: f(lg, pre, sl, None, out, None)


def _valid(idx, lg_row, K, N):
    u = torch.unique(idx[idx >= 0])
    if u.numel() != K or int(idx.min()) < 0 or int(idx.max()) >= N:
        return False
    v = lg_row[:N].float()
    kv = v.gather(0, idx.clamp(min=0).long()).sort().values
    rv = torch.topk(v, K).values.sort().values
    return bool(torch.equal(kv, rv))


def _time(call, reps=30, warm=10):
    for _ in range(warm):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        _EVICT.uniform_(0, 1)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); call(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def cell(K, N, scen, tdt, BS):
    b = B.get_bundle(scen, K, torch.float32, N)
    lg_row = b["logits"].to(tdt).contiguous()
    pre_row = b["preIdx"].contiguous()
    cr = b["cr"]; Np = lg_row.shape[1]
    lg = lg_row.expand(BS, -1).contiguous()
    pre = pre_row.expand(BS, -1).contiguous()
    sl = torch.full((BS,), Np * cr, dtype=torch.int32, device=DEV)

    rows = {}
    # frozen (harness/perf_3arm_bs.py config)
    fro = dict(cluster_size=1 if N < 65536 else 4, num_threads=1024,
               use_256bit_load=True, min_blocks_per_mp=1,
               enable_warp_parallel_reduce=True)
    # runner-faithful
    run = runner_policy(tdt, BS, Np)
    for tag, cfg in (("pr_frozen", fro), ("pr_runner", run)):
        out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
        key = (tag, tdt, K, cr) + tuple(sorted(cfg.items()))
        call = pr_call(key, cfg, tdt, K, cr, lg, pre, sl, out)
        call(); torch.cuda.synchronize()
        ex = _valid(out[0], lg[0], K, N)
        rows[tag] = (_time(call), ex, cfg["cluster_size"], cfg["num_threads"],
                     cfg["min_blocks_per_mp"])
    # op26_r0auto
    call26, keep, extra = build_call("op26_r0auto", K, tdt, N, BS,
                                     cr, lg_row, pre_row)
    call26(); torch.cuda.synchronize()
    out26 = next(t for t in reversed(keep)
                 if torch.is_tensor(t) and t.dtype == torch.int32
                 and t.dim() == 2 and t.shape[-1] == K)
    ex26 = _valid(out26[0], keep[0][0], K, N)
    rows["op26"] = (_time(call26), ex26, extra.get("r0_arm", "?"), "", "")
    return rows


if __name__ == "__main__":
    DTN = {torch.float32: "fp32", torch.bfloat16: "bf16"}
    CELLS = eval(sys.argv[1]) if len(sys.argv) > 1 else [
        (512, N, "best", dt, bs)
        for dt in (torch.float32, torch.bfloat16)
        for N in (16384, 65536, 131072)
        for bs in (64, 256, 1024)
    ] + [(1024, 65536, "best", torch.bfloat16, 1024),
         (1024, 131072, "worst", torch.float32, 1024)]
    hdr = (f"{'K':>5} {'N':>7} {'scen':>5} {'dt':>4} {'BS':>5} "
           f"{'frozen_us':>9} {'runner_us':>9} {'op26_us':>8} "
           f"{'fro/26':>6} {'run/26':>6} {'runner_cfg':>16} {'op26_arm':>8} ex(f/r/26)")
    print(hdr, flush=True)
    rr = []
    for K, N, scen, dt, bs in CELLS:
        r = cell(K, N, scen, dt, bs)
        tf, ef = r["pr_frozen"][0], r["pr_frozen"][1]
        tr, er = r["pr_runner"][0], r["pr_runner"][1]
        c26 = r["op26"]
        rcfg = f"cs{r['pr_runner'][2]}/T{r['pr_runner'][3]}/mb{r['pr_runner'][4]}"
        rr.append((tf / c26[0], tr / c26[0]))
        print(f"{K:>5} {N:>7} {scen:>5} {DTN[dt]:>4} {bs:>5} "
              f"{tf:>9.2f} {tr:>9.2f} {c26[0]:>8.2f} "
              f"{tf / c26[0]:>6.3f} {tr / c26[0]:>6.3f} {rcfg:>16} "
              f"{str(c26[2]):>8} {ef}/{er}/{c26[1]}", flush=True)
    gf = math.exp(sum(math.log(a) for a, _ in rr) / len(rr))
    gr = math.exp(sum(math.log(b) for _, b in rr) / len(rr))
    print(f"\ngeomean pr_frozen/op26 = {gf:.3f}   pr_runner/op26 = {gr:.3f}"
          f"   (>1 = PR arm slower than op26_r0auto)", flush=True)
