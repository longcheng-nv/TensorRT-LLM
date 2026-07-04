# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Full-grid exactness + no-fallback validation for the LOG-interp p2clog op.

Same two-check protocol as validate_exactness.py (iter7):
  (A) REAL KERNEL value-equivalence vs torch.topk (production launch path).
  (B) HOST-REPLAY (interp_mode="logcount") done==1 + cand in [K,kCC].
      NOTE: the replay uses np.log2 while the kernel uses fastmath lg2.approx;
      trajectories can differ in ulps, so (A) on the real kernel is the
      exactness authority and (B) is the no-fallback evidence on the replay's
      own (validated-faithful) control flow.

Variant portfolio (iter8a host winners, validated at ALL N — the ship dispatch
table is baked only after the iter8c nsys A/B picks per-regime winners):
  fp32: K512  (kcc=1024, kft=614)   narrow-log
        K1024 (kcc=2048, kft=1024)  narrow-log
        K1024 (kcc=None, kft=1024)  base-log (large-N candidate)
        K2048 (kcc=None, kft=2048)  base-log
        K2048 (kcc=4096, kft=2048)  kc2x-log
"""
import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parent / "src"))
from synth_data import get_bundle  # noqa: E402
from gvr_p2clog_op import gvr_cutedsl_p2clog  # noqa: E402
from p2_replay import replay_row, SecantCfg  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
CFGS = ["beta_shallow", "beta_moderate", "beta_deep"]
N_GRID = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

# (dtype, K) -> list of (kcc, kft) variants to validate at all N.
VARIANTS = {
    (torch.float32, 512): [(1024, 614)],
    (torch.float32, 1024): [(2048, 1024), (None, 1024)],
    (torch.float32, 2048): [(None, 2048), (4096, 2048)],
}


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
    ap.add_argument("--dts", default="fp32")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--ns", default="")
    args = ap.parse_args()

    dts = [DTYPES[x] for x in args.dts.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    ns = [int(x) for x in args.ns.split(",")] if args.ns else N_GRID

    n_cells = n_fail_val = n_fail_fallback = 0
    failures = []

    for (dtype, K), variants in VARIANTS.items():
        if dtype not in dts:
            continue
        cr_val = 4 if K in (512, 1024) else 1
        for kcc, kft in variants:
            for N in ns:
                if N <= 2 * K:
                    continue
                for cfg in CFGS:
                    for seed in seeds:
                        b = get_bundle(K, dtype, N, cfg=cfg, seed=seed)
                        logits = b["logits"].to(dtype).contiguous()
                        pre = b["preIdx"].contiguous()
                        seq_lens = torch.full((1,), N * cr_val, dtype=torch.int32, device="cuda")
                        out = torch.empty(1, K, dtype=torch.int32, device="cuda")
                        gvr_cutedsl_p2clog(logits, pre, seq_lens, K, cr_val,
                                           kcc=kcc, kft=kft, out=out)
                        torch.cuda.synchronize()
                        ok_val, d = kernel_exact(out, logits[0], K)
                        scfg = SecantCfg(kCC=kcc, kFTarget=kft, interp_mode="logcount")
                        rs = replay_row(logits[0], pre[0], N, K, cr_val, dtype, scfg)
                        ok_done = rs.converged and (K <= rs.cand_count <= (kcc or rs.cand_count + 1))

                        n_cells += 1
                        if not ok_val:
                            n_fail_val += 1
                            failures.append(f"VAL  {str(dtype):>14} K={K} kcc={kcc} kft={kft} N={N} {cfg} s{seed} d={d:.2e}")
                        if not ok_done:
                            n_fail_fallback += 1
                            failures.append(
                                f"DONE {str(dtype):>14} K={K} kcc={kcc} kft={kft} N={N} {cfg} s{seed} "
                                f"done={'1' if rs.converged else '2'} cand={rs.cand_count} evals={rs.p2_evals}")

    print("\n=== SUMMARY ===")
    print(f"cells checked   : {n_cells}")
    print(f"value-equiv FAIL: {n_fail_val}")
    print(f"P2-fallback FAIL: {n_fail_fallback}")
    if failures:
        print("\nFAILURES:")
        for f in failures[:60]:
            print("  " + f)
    ok = n_fail_val == 0 and n_fail_fallback == 0
    print(f"\nVERDICT: {'PASS — all cells value-exact + no P2 fallback' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
