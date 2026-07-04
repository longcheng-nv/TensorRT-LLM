# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Full-grid exactness + no-fallback validation for the N-dispatched p2c op.

Two independent checks per cell:
  (A) REAL KERNEL value-equivalence: run src/gvr_p2c_op.gvr_cutedsl_p2c (the exact
      production launch path with the N-dispatched kCC/kFTarget override) and assert
      the gathered output values == torch.topk(logits, K).values (max |Δ| < 1e-3).
  (B) HOST-REPLAY no-fallback: replay the secant control flow with the SAME
      (kCC, kFTarget) the op dispatched and assert done==1 (converged, NO P2
      fallback to done==2) and cand ∈ [K, kCC]. The replay is validated 720/720 vs
      the real kernel (iter 0), so done==1 here == the kernel took no fallback.

Grid: dtype ∈ {fp32, bf16, fp16} × K ∈ {512, 1024, 2048} × all N × 3 beta cfgs ×
seeds. fp32 K512/K1024 at N≤65536 exercise the narrow params; everything else is
baseline (still validated, to prove the dispatch is a strict superset of baseline).
"""
import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parent / "src"))
from synth_data import get_bundle  # noqa: E402
from gvr_p2c_op import gvr_cutedsl_p2c, dispatch_params, NARROW_N_MAX  # noqa: E402
from p2_replay import replay_row, SecantCfg  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
CFGS = ["beta_shallow", "beta_moderate", "beta_deep"]
N_GRID = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


def kernel_exact(out, logits_row, K):
    idx = out[0].clamp(min=0).long()
    if len(set(idx.tolist())) != K:
        return False, float("inf")
    v = logits_row.float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits_row.float(), K).values
    d = (v - ref).abs().max().item()
    return d < 1e-3, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dts", default="fp32,bf16,fp16")
    ap.add_argument("--Ks", default="512,1024,2048")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--ns", default="")  # subset N
    args = ap.parse_args()

    dts = [DTYPES[x] for x in args.dts.split(",")]
    Ks = [int(x) for x in args.Ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    ns = [int(x) for x in args.ns.split(",")] if args.ns else N_GRID

    n_cells = 0
    n_fail_val = 0
    n_fail_fallback = 0
    narrow_cells = 0
    failures = []

    for dtype in dts:
        for K in Ks:
            cr_val = 4 if K in (512, 1024) else 1
            for N in ns:
                if N <= 2 * K:
                    continue
                kcc, kft = dispatch_params(dtype, K, N)
                is_narrow = kcc is not None
                for cfg in CFGS:
                    for seed in seeds:
                        b = get_bundle(K, dtype, N, cfg=cfg, seed=seed)
                        logits = b["logits"].to(dtype).contiguous()
                        pre = b["preIdx"].contiguous()
                        seq_lens = torch.full((1,), N * cr_val, dtype=torch.int32, device="cuda")
                        out = torch.empty(1, K, dtype=torch.int32, device="cuda")
                        # (A) real kernel value-equiv
                        gvr_cutedsl_p2c(logits, pre, seq_lens, K, cr_val, out=out)
                        torch.cuda.synchronize()
                        ok_val, d = kernel_exact(out, logits[0], K)
                        # (B) host-replay no-fallback (with dispatched params)
                        scfg = SecantCfg(kCC=kcc, kFTarget=kft)
                        rs = replay_row(logits[0], pre[0], N, K, cr_val, dtype, scfg)
                        ok_done = rs.converged and (K <= rs.cand_count <= (kcc or rs.cand_count + 1))

                        n_cells += 1
                        if is_narrow:
                            narrow_cells += 1
                        if not ok_val:
                            n_fail_val += 1
                            failures.append(f"VAL  {args.dts and str(dtype):>14} K={K} N={N} {cfg} s{seed} d={d:.2e}")
                        if not ok_done:
                            n_fail_fallback += 1
                            failures.append(
                                f"DONE {str(dtype):>14} K={K} N={N} {cfg} s{seed} "
                                f"done={'1' if rs.converged else '2'} cand={rs.cand_count} "
                                f"evals={rs.p2_evals} kcc={kcc}")

    print(f"\n=== SUMMARY ===")
    print(f"cells checked   : {n_cells}  (narrow={narrow_cells}, baseline={n_cells - narrow_cells})")
    print(f"value-equiv FAIL: {n_fail_val}")
    print(f"P2-fallback FAIL: {n_fail_fallback}")
    if failures:
        print("\nFAILURES:")
        for f in failures[:60]:
            print("  " + f)
    verdict = "PASS — all cells value-exact + no P2 fallback" if (n_fail_val == 0 and n_fail_fallback == 0) else "FAIL"
    print(f"\nVERDICT: {verdict}")
    sys.exit(0 if (n_fail_val == 0 and n_fail_fallback == 0) else 1)


if __name__ == "__main__":
    main()
