# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""[op37-dp4] real-data spot check: dist_p4=True on production captures.

One cell per source, BS=4 (BS=1 capture row replicated), production launch
contract (GvrTopKKernel.launch -> pick_config, cs>1 by shape), exact vs
torch.topk (index validity + uniqueness + value multiset):
  - flash  K=512  cr=4  ISL 512k  (harness/real_data_v4cap.py)
  - pro    K=1024 cr=4  ISL 512k  (harness/real_data_v4cap.py)
  - v32    K=2048 cr=1  ISL 128k  (harness/real_data_v32.py)
NOTE: the task path op26_r0_upstream_port_report/harness/real_data_v32.py
does not exist; the canonical loader lives at harness/real_data_v32.py.
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH / "harness"))

from gvrpkg37.top_k.gvr_topk_decode import GvrTopKKernel as Gvr37  # noqa: E402
import real_data_v4cap as v4cap  # noqa: E402
import real_data_v32 as v32  # noqa: E402

DEV = "cuda"
BS = 4


def check_exact(logits, out, K):
    BS_, N = logits.shape
    ref = torch.sort(torch.topk(logits.float(), K, dim=1).values,
                     dim=1, descending=True).values
    for b in range(BS_):
        idx = out[b].long()
        if (idx < 0).any() or (idx >= N).any():
            return False, f"row{b}: invalid index"
        if idx.unique().numel() != K:
            return False, f"row{b}: duplicate indices"
        got = torch.sort(logits[b, idx].float(), descending=True).values
        if not torch.equal(got, ref[b]):
            return False, f"row{b}: multiset mismatch ({(got != ref[b]).sum().item()})"
    return True, ""


def run_cell(name, bundle):
    K, cr, N = bundle["K"], bundle["cr"], bundle["N"]
    logits = bundle["logits"].float().repeat(BS, 1).contiguous()
    pre = bundle["preIdx"].repeat(BS, 1).contiguous()
    sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    cfg = Gvr37.pick_config(torch.float32, BS, logits.shape[1])
    assert cfg["cluster_size"] > 1, (name, cfg)
    Gvr37.launch(logits, pre, sl, out, K, compress_ratio=cr, dist_p4=True)
    torch.cuda.synchronize()
    ok, why = check_exact(logits, out, K)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} K={K} cr={cr} N={N} BS={BS} "
          f"cs={cfg['cluster_size']} T={cfg['num_threads']} "
          f"hit={bundle['hit_rate']:.3f} layer={bundle['layer']} | "
          f"{'OK' if ok else 'FAIL:' + why}", flush=True)
    return ok


def main():
    torch.cuda.init()
    npass = ntot = 0
    for name, bundle in (
        ("flash-512k", v4cap.get_bundle("flash", "512k", 22, "fp32")),
        ("pro-512k", v4cap.get_bundle("pro", "512k", 30, "fp32")),
        ("v32-128k", v32.get_bundle("v32", "128k", 31, "fp32")),
    ):
        ntot += 1
        npass += int(run_cell(name, bundle))
    print(f"REALDATA_DP4: {npass}/{ntot} PASS", flush=True)
    return 0 if npass == ntot else 1


if __name__ == "__main__":
    sys.exit(main())
