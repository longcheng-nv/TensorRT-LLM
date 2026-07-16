#!/usr/bin/env python3
"""Why did the op26 report's synth+real sweeps not catch the sglang-v2 topk
precision issue reported in Slack (uniform [0,1) @ 128K)?

Quantitative probe: sglang v2 builds a coarse histogram over the TOP kHistBits
bits of the fp16(ordered) representation (TopKRegister: 12 bits, N<=8192;
TopKStreaming: 12 bits; TopKCluster: 10 bits). Elements strictly above the
threshold bin are emitted; elements INSIDE the threshold bin go to a tie
buffer capped at kMaxNumTie=2048 — overflow candidates are silently DROPPED
(topk_impl.cuh L673-674). So the kernel is exact iff
    count(threshold bin) <= 2048.
This script computes that count for (a) uniform [0,1) — the Slack repro,
(b) our op22-env synth bundles, (c) our real V4 flash/pro + V3.2 captures,
at the exact (K, N) cells the report swept.
"""
import sys, json
from pathlib import Path
import torch

BENCH = Path("/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench")
sys.path.insert(0, str(BENCH / "op26_r0_upstream_port_report" / "rival_harness"))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))

def coarse_bin(x, bits):
    """extract_coarse_bin<bits>(fp32 x) — torch replica (topk_impl.cuh L78)."""
    hx = x.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    key = torch.where(hx & 0x8000 != 0, (~hx) & 0xFFFF, hx | 0x8000)
    return key >> (16 - bits)

def tiebin_count(row_f32, K, bits):
    """(#above, #in-threshold-bin) under the coarse-bin invariant:
    count(bins > b*) < K <= count(bins >= b*)."""
    b = coarse_bin(row_f32, bits)
    hist = torch.bincount(b, minlength=1 << bits)
    above_cum = hist.flip(0).cumsum(0).flip(0)          # count(bins >= i)
    gt_cum = above_cum - hist                           # count(bins >  i)
    bstar = int((gt_cum < K).nonzero()[-1].item()) if (gt_cum < K).any() else 0
    # threshold bin = HIGHEST bin with count(bins > b) < K and count(bins>=b)>=K
    cand = ((gt_cum < K) & (above_cum >= K)).nonzero()
    bstar = int(cand[-1].item()) if cand.numel() else bstar
    return int(gt_cum[bstar].item()), int(hist[bstar].item()), bstar

def report(tag, row, K, N):
    row = row[:N].float().cpu()
    out = {}
    for bits, path in ((10, "cluster(N large)"), (12, "reg/streaming")):
        ab, tie, bstar = tiebin_count(row, K, bits)
        out[bits] = (ab, tie)
        flag = "  << OVERFLOW (>2048, silently dropped) >>" if tie > 2048 else ""
        print(f"  {tag:44s} K={K:<5d} N={N:<8d} bits={bits:2d} [{path:16s}] "
              f"above={ab:<6d} tie_bin={tie:<7d}{flag}")
    return out

print("=== (a) Slack repro: uniform [0,1) ===")
g = torch.Generator().manual_seed(0)
for N in (65536, 131072, 262144):
    u = torch.rand(N, generator=g)
    for K in (512, 1024, 2048):
        report("uniform[0,1)", u, K, N)

print("\n=== (b) op22-env SYNTH bundles (as swept in report §8/§3) ===")
import bundle_data_env as SYNTH
for scen, K in (("best", 512), ("best", 1024), ("best", 2048),
                ("worst", 2048), ("slowbase", 2048)):
    b = SYNTH.get_bundle(scen, K, torch.float32, 131072, device="cpu")
    lg = b["logits"]
    row = lg[0] if lg.dim() == 2 else lg
    report(f"synth {scen} ({b.get('cfg','')})"[:44], row.float().cpu(), K, 131072)

print("\n=== (c) REAL decode captures (as swept in report §4/§8) ===")
import real_data_v4cap as RV4
import real_data_v32 as RV32
for model, K, L in (("flash", 512, 22), ("pro", 1024, 30)):
    for isl in ("64k", "128k", "256k"):
        b = RV4.get_bundle(model, isl, L, "fp32")
        row = b["logits"][0] if b["logits"].dim() == 2 else b["logits"]
        report(f"real {model} {isl} L{L}", row.float().cpu(), K, b["N"])
for isl in ("64k", "128k"):
    b = RV32.get_bundle("v32", isl, 34, "fp32")
    row = b["logits"][0] if b["logits"].dim() == 2 else b["logits"]
    report(f"real v32 {isl} L34", row.float().cpu(), K=2048, N=b["N"])
