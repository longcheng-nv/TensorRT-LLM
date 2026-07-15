# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op34 env + anchor check on the CURRENT node (node changed 074->048).

Verifies imports/JIT builds and re-establishes the anchor: op26_r0auto vs
sglang_v2 on a fixed anchor cell (pro/256k/mid-GVR-layer, BS=1 fp32) using
cold-L2 wallclock median (L1 screen — NOT a ship verdict, just an anchor for
absolute-us transfer across the node change).
"""
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OPBENCH = HERE.parents[1]
sys.path.insert(0, str(OPBENCH / "harness"))
sys.path.insert(0, str(OPBENCH / "op26_gvr_logfalsi_rs" / "src"))

import real_data_v4cap as RD4
from sglang_v2_op import topk_v2, plan as sglv2_plan
from gvr_op26_r0_op import gvr_r0_op26

DEV = "cuda"
print(f"[env] torch {torch.__version__} cap {torch.cuda.get_device_capability()} "
      f"{torch.cuda.get_device_name()}")


def cold_median(fn, evict, iters=30, warmup=5):
    for _ in range(warmup):
        evict(); fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        evict()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    model, isl = "pro", "256k"
    b = None
    # mid GVR layer
    layers = RD4.MODELS[model]["layers"]
    L = layers[len(layers) // 2]
    print(f"[anchor] loading {model} ISL={isl} layer={L} ...")
    b = RD4.get_bundle(model, isl, L, "fp32")
    logits_row = b["logits"]; pre = b["preIdx"]; N = b["N"]; K = b["K"]; cr = b["cr"]
    print(f"[anchor] N={N} Npad={b['Npad']} K={K} hit_rate={b['hit_rate']:.3f}")

    evbuf = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device=DEV)  # 512MB
    def evict():
        evbuf.uniform_()

    out = torch.empty((1, K), dtype=torch.int32, device=DEV)
    # GVR seq_lens = UNCOMPRESSED length; kernel divides by cr internally.
    seq_div = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)

    sl = torch.full((1,), N, dtype=torch.int32, device=DEV)
    md = sglv2_plan(sl)
    torch.cuda.synchronize()
    topk_v2(logits_row, sl, K, out=out, metadata=md, max_seq_len=N)
    t_sgl = cold_median(lambda: topk_v2(logits_row, sl, K, out=out, metadata=md,
                                        max_seq_len=N), evict)

    gvr_r0_op26(logits_row, pre, seq_div, K, compress_ratio=cr, out=out)
    t_r0 = cold_median(lambda: gvr_r0_op26(logits_row, pre, seq_div, K,
                                           compress_ratio=cr, out=out), evict)

    # exactness sanity
    gvr_r0_op26(logits_row, pre, seq_div, K, compress_ratio=cr, out=out)
    torch.cuda.synchronize()
    vdiff, recall, nneg = RD4.value_metrics(out[0], logits_row, b["ref"], K)
    print(f"[anchor 048] cold-L2 median us: sglang={t_sgl:.2f}  op26_r0auto={t_r0:.2f}  "
          f"r0/sgl={t_r0/t_sgl:.3f}  (r0 exact vdiff={vdiff:.2e} recall={recall:.3f})")
    print("[env] OK — imports/JIT/loader/anchor all functional")


if __name__ == "__main__":
    main()
