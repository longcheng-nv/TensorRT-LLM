#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op37 D3 probe — does the tight_bracket admission FIRE on the cold
flash/512k win cell (hit .057), or does tb=on win via tightened falsi
seeds? Runs the cell once with tb_debug=True (leader-CTA printf of
fired/lo_m/hi_m/cnt_hi/band) at BS=1 (one row, one printf) and BS=8.

Battery (43a92cc70c) says the bracket fell to refine on this cell;
phase data (phase_lj.csv) shows band-P3 running. One of the two is
mis-read — this settles it. Usage: CUDA_VISIBLE_DEVICES=<gpu> \
ARM=tb_on|tb_thin|esc python3 src/probe_tb_debug.py [model isl layer BS ...]
"""
import os
import sys
from pathlib import Path

import torch

ARM = os.environ.get("ARM", "tb_on")
ARM_KW = {
    "base": {},
    "base_t512": dict(num_threads=512, enable_warp_parallel_reduce=False),
    "tb_on": dict(tight_bracket=True, tb_debug=True),
    "tb_on_t512": dict(tight_bracket=True, tb_debug=True, num_threads=512,
                       enable_warp_parallel_reduce=False),
    "tb_thin": dict(tight_bracket=True, tb_qfracs=(0.85, 0.35, 0.05), tb_debug=True),
    "esc": dict(tb_escape=True, tb_debug=True),
    "esc_t512": dict(tb_escape=True, tb_debug=True, num_threads=512,
                     enable_warp_parallel_reduce=False),
}[ARM]

HERE = Path(__file__).resolve().parent
OP37 = HERE.parent
OPBENCH = OP37.parent
sys.path.insert(0, str(OPBENCH / "op26_r0_upstream_port_report" / "p4f1_harness"))
sys.path.insert(0, str(OPBENCH / "harness"))
sys.path.insert(0, str(OP37 / "variant"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0"), cutlass.__version__

from gvrpkg37.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
import real_data_v4cap as RV4  # noqa: E402

DEV = "cuda"


def run(model, isl, layer, BS):
    b = RV4.get_bundle(model, isl, layer, "fp32")
    # bundle logits may be [1, L] — flatten to the 1D row (the earlier 2D
    # lg[:N] sliced ROWS and made the exact-check gather assert, NOT a
    # kernel defect: the base arm "failed" identically).
    lg = b["logits"].to(torch.float32).contiguous().reshape(-1)
    pre = b["preIdx"].contiguous().reshape(-1)
    N, K, cr, ref = b["N"], b["K"], b["cr"], b["ref"]
    logits = lg.expand(BS, -1).contiguous()
    pre_b = pre.expand(BS, -1).contiguous()
    seq_lens = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    cfg = GvrTopKKernel.pick_config(torch.float32, BS, N, max_seq_len=N * cr)
    print(f"[{ARM} {model}/{isl} L{layer} BS={BS}] K={K} N={N} pick=cs{cfg['cluster_size']}/"
          f"T{cfg['num_threads']} hit={b['hit_rate']:.3f}", flush=True)
    # Production launch() contract — same pick/compile path as the nsys
    # verdict sweep's _build_lj/_build_esc arms.
    out = torch.full((BS, K), -1, dtype=torch.int32, device=DEV)
    GvrTopKKernel.launch(logits, pre_b, seq_lens, out, K, compress_ratio=cr,
                         **ARM_KW)
    torch.cuda.synchronize()
    idx = out[0].to(torch.int64)
    lgr = lg[:N].float()
    n_bad = int((idx < 0).sum() + (idx >= N).sum())
    uniq = int(torch.unique(idx.clamp(0, N - 1)).numel())
    if n_bad or uniq != K:
        print(f"  exact=False (bad_idx={n_bad} uniq={uniq}/{K})", flush=True)
        return
    exact = bool(torch.equal(lgr[idx].sort().values,
                             lgr[ref.to(torch.int64)].sort().values))
    print(f"  exact={exact}", flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    args = sys.argv[1:]
    cells = []
    while args:
        cells.append((args[0], args[1], int(args[2]), int(args[3])))
        args = args[4:]
    if not cells:
        cells = [("flash", "512k", 22, 1), ("flash", "512k", 22, 8)]
    for c in cells:
        run(*c)
