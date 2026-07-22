"""Minimal GVR top-K workload for APIC capture.

Runs the pinned-head (PR#16457 @04a0900ff7) GVR kernel on one real cell
(pro_512k_L30: K=1024, N=131075, low hint 0.23), 8 warmup launches
(incl. cuteDSL JIT) + 5 steady-state launches, then exits.
"""
import os
import sys
from pathlib import Path

import torch

KFC = Path("/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/"
           "TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/"
           "kf_campaign")
sys.path.insert(0, str(KFC / os.environ.get("GVRPKG_DIR", "gvrpkg_04a0")))
sys.path.insert(0, str(KFC.parent.parent / "harness"))

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
import real_data_v4cap as v4  # noqa: E402

b = v4.get_bundle("pro", "512k", 30, "fp32")
K, cr, N = b["K"], b["cr"], b["N"]
lg = b["logits"].contiguous()
pre = b["preIdx"].contiguous()
sl = torch.full((1,), N * cr, dtype=torch.int32, device="cuda")
out = torch.empty(1, K, dtype=torch.int32, device="cuda")

for _ in range(8):  # warmup (first launch triggers cuteDSL JIT)
    GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr)
torch.cuda.synchronize()

for _ in range(5):  # steady-state launches (capture targets)
    GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr)
torch.cuda.synchronize()

idx = out.flatten().to(torch.int64)
ref = b["ref"].to(torch.int64)
sel = b["logits"][0, :N].float()[idx].sort().values
rv = b["logits"][0, :N].float()[ref].sort().values
print("EXACT:", bool(torch.equal(sel, rv)), "| K,N,cr =", K, N, cr)
print("DRIVER_DONE")
