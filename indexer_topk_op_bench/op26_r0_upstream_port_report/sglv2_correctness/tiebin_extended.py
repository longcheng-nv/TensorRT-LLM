#!/usr/bin/env python3
"""Extended sglang-v2 tie-bin occupancy sweep on REAL data.

Axes beyond the first probe:
  1. ALL layers (flash 21 / pro 30 / v32 58), not just the report's bench layer.
  2. ALL ISL rungs up to the capture max (V4 1M -> N=262144; V3.2 256k -> N=262144).
  3. v32 256k: ALL decode steps (temporal samples), not just the last.
  4. Synth extrapolation past the capture range: N = 512K / 1M (and 2M if the
     skill generates it) at K=2048 -- what tie-bin count does calibrated
     realistic data produce where no real capture exists?

Output: per (model, ISL): N, layer-max / median tie-bin count for the 10-bit
(cluster) and 12-bit (streaming) histograms, #layers over the 2048 cap.
"""
import sys, json
from pathlib import Path
import torch

BENCH = Path("/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench")
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))
import real_data_v4cap as RV4
import real_data_v32 as RV32

CAP = 2048

def coarse_bin(x, bits):
    hx = x.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    key = torch.where(hx & 0x8000 != 0, (~hx) & 0xFFFF, hx | 0x8000)
    return key >> (16 - bits)

def tie_count(row_f32, K, bits):
    hist = torch.bincount(coarse_bin(row_f32, bits), minlength=1 << bits)
    above_cum = hist.flip(0).cumsum(0).flip(0)
    gt_cum = above_cum - hist
    cand = ((gt_cum < K) & (above_cum >= K)).nonzero()
    bstar = int(cand[-1].item()) if cand.numel() else 0
    return int(hist[bstar].item())

def summarize(tag, N, K, ties10, ties12, extra=""):
    t10 = torch.tensor(ties10, dtype=torch.float64)
    t12 = torch.tensor(ties12, dtype=torch.float64)
    o10 = int((t10 > CAP).sum()); o12 = int((t12 > CAP).sum())
    flag = f"  << {o10} layers OVERFLOW (10-bit) >>" if o10 else ""
    print(f"{tag:22s} N={N:<8d} K={K:<5d} nrows={len(ties10):<4d} "
          f"tie10 max={int(t10.max()):<6d} med={int(t10.median()):<6d} "
          f"tie12 max={int(t12.max()):<6d} med={int(t12.median()):<5d} "
          f"margin10={CAP/max(1,int(t10.max())):.2f}x {extra}{flag}")

print("=== REAL V4 flash (K=512) / pro (K=1024): ALL layers x ALL ISL ===")
for model in ("flash", "pro"):
    K = RV4.MODELS[model]["K"]
    for isl in RV4.ISLS:
        s = RV4._slim(model, isl)
        N = s["N"]
        t10, t12 = [], []
        for L, row in s["cur"].items():
            r = row[:N].float()
            t10.append(tie_count(r, K, 10)); t12.append(tie_count(r, K, 12))
        summarize(f"real {model} {isl}", N, K, t10, t12)

print("\n=== REAL V3.2 (K=2048): ALL 58 layers x ALL ISL (last step) ===")
worst_cells = []
for isl in RV32.ISLS:
    s = RV32._slim(isl)
    N, s_last = s["N"], s["s_last"]
    t10, t12, per_layer = [], [], {}
    for L in RV32.LAYERS_ALL:
        d = RV32._layer_dir(isl, L)
        try:
            lg = torch.load(d / "decode.logits.in.pt", map_location="cpu",
                            weights_only=False)
        except Exception as e:
            print(f"  !! v32 {isl} L{L}: {e}"); continue
        row = lg[s_last]
        r = (row[0] if row.dim() == 2 else row).float()[:N]
        c10 = tie_count(r, 2048, 10)
        t10.append(c10); t12.append(tie_count(r, 2048, 12))
        per_layer[L] = c10
        del lg
    summarize(f"real v32 {isl}", N, 2048, t10, t12)
    top = sorted(per_layer.items(), key=lambda kv: -kv[1])[:3]
    print(f"    top layers (10-bit): {top}")
    worst_cells.append((isl, top[0][0], top[0][1]))

print("\n=== REAL V3.2 256k: ALL decode steps x all layers (temporal max) ===")
isl = "256k"
s = RV32._slim(isl)
best = (0, None, None)   # (tie10, layer, step)
for L in RV32.LAYERS_ALL:
    d = RV32._layer_dir(isl, L)
    lg = torch.load(d / "decode.logits.in.pt", map_location="cpu", weights_only=False)
    pk = torch.load(d / "decode.topk.out.pt", map_location="cpu", weights_only=False)
    for st in sorted(lg.keys()):
        Ns = int(pk[st].max()) + 1          # per-layer valid bound (safe slice)
        row = lg[st]
        r = (row[0] if row.dim() == 2 else row).float()[:Ns]
        c = tie_count(r, 2048, 10)
        if c > best[0]:
            best = (c, L, st)
    del lg, pk
print(f"  v32 256k all-steps max tie10 = {best[0]} (layer {best[1]}, step {best[2]}) "
      f"margin={CAP/max(1,best[0]):.2f}x")

print("\n=== SYNTH extrapolation past capture range (calibrated skill) ===")
import bundle_data_env as SYNTH
for K, scens in ((2048, ("best", "worst", "slowbase")),
                 (1024, ("best", "worst")), (512, ("best", "worst"))):
    for scen in scens:
        for N in (262144, 524288, 1048576, 2097152):
            try:
                b = SYNTH.get_bundle(scen, K, torch.float32, N, device="cpu")
            except Exception as e:
                print(f"  synth {scen} K{K} N={N}: unavailable ({type(e).__name__})")
                continue
            r = b["logits"][0][:N].float()
            summarize(f"synth {scen} K{K}", N, K,
                      [tie_count(r, K, 10)], [tie_count(r, K, 12)])

print("\nWORST_REAL_CELLS =", json.dumps(worst_cells))
