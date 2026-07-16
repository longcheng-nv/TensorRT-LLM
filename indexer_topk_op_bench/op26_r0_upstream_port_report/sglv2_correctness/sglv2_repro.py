#!/usr/bin/env python3
"""End-to-end repro: run the vendored sglang v2 kernel on (a) uniform [0,1)
@ N=128K (Slack repro) and (b) our op22-env synth 'best' bundle, then apply
the report's _exact gate (value-set == torch.topk). Expect (a) FAIL (b) PASS."""
import sys
from pathlib import Path
import torch

BENCH = Path("/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench")
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))
from sglang_v2_op import topk_v2, plan  # noqa: E402

DEV = "cuda"

def run_case(tag, row_f32, N, K):
    W = ((N + 63) // 64) * 64
    lg = torch.full((1, W), torch.finfo(torch.float32).min, dtype=torch.float32, device=DEV)
    lg[0, :N] = row_f32[:N].to(DEV)
    sl = torch.full((1,), N, dtype=torch.int32, device=DEV)
    out = torch.empty((1, K), dtype=torch.int32, device=DEV)
    md = plan(sl)
    torch.cuda.synchronize()
    topk_v2(lg, sl, K, out=out, metadata=md, max_seq_len=N)
    torch.cuda.synchronize()
    idx = out[0].long().cpu()
    row = row_f32[:N].float().cpu()
    ref = torch.topk(row, K).values.sort().values
    ok_range = bool(idx.min() >= 0 and idx.max() < N)
    uniq = int(idx.unique().numel())
    if ok_range:
        got = row[idx].sort().values
        exact = uniq == K and torch.equal(got, ref)
        nbad = int((got != ref).sum())
        maxerr = float((ref - got).abs().max()) if nbad else 0.0
        # how deep do wrong picks sit below the true K-th value?
        thr = float(ref[0])
        worst_pick = float(got.min())
    else:
        exact, nbad, maxerr, thr, worst_pick = False, K, float("nan"), float("nan"), float("nan")
    print(f"{tag:38s} N={N:<8d} K={K:<5d} exact={str(exact):5s} uniq={uniq:<5d} "
          f"mismatched_slots={nbad:<5d} max_val_err={maxerr:.6f} "
          f"true_thr={thr:.6f} worst_pick={worst_pick:.6f}")

g = torch.Generator().manual_seed(0)
u128 = torch.rand(131072, generator=g)
u64 = torch.rand(65536, generator=g)
u256 = torch.rand(262144, generator=g)
for K in (512, 1024, 2048):
    run_case("uniform[0,1) (Slack repro)", u128, 131072, K)
run_case("uniform[0,1)", u64, 65536, 2048)
run_case("uniform[0,1)", u256, 262144, 2048)

import bundle_data_env as SYNTH
for scen, K in (("best", 2048), ("worst", 2048)):
    b = SYNTH.get_bundle(scen, K, torch.float32, 131072, device="cpu")
    row = b["logits"][0].float()
    run_case(f"synth {scen} K{K}", row, 131072, K)

import real_data_v32 as RV32
b = RV32.get_bundle("v32", "128k", 34, "fp32")
run_case("real v32 128k L34", b["logits"][0].float().cpu(), b["N"], 2048)
