# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""d1a (p4_peer_push) DSMEM checklist fixtures — NOTES.md ship item 3.

Explicit unit coverage BEFORE PR packaging:
  - cs in {2, 4, 8, 16} (compile-forced, not just pick_config's choices)
  - forced-hit (preIdx == exact top-K, hit=1.0) / forced-miss (bottom-K)
  - generic randn + noise preIdx
  - short-row degrade: compile a cs>1 config, pass sl small enough that
    seq_len fits one CTA slice -> do_cluster_sync=False path with the
    p4_peer_push flag STILL ON (peers must not push, leader must not wait)
Arms: base / d1a / all. Exactness = tie-robust value-multiset vs torch.topk.

  PYTHONNOUSERSITE=1 PYTHONPATH=<cutlass450> python3 validate_d1a_fixtures.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "variant"))

import cutlass  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

from gvrpkgpr.top_k.gvr_topk_decode import GvrTopKKernel as K37  # noqa: E402
from validate_op37 import compile_arm, exact_set  # noqa: E402

DEV = "cuda"
ARMS = [("base", {}),
        ("d1a", dict(p4_peer_push=True)),
        ("all", dict(p4_rs_rw_search=True, p4_fine_skip=True,
                     p4_peer_push=True))]
KCRS = [(512, 4), (1024, 4), (2048, 1)]
CSS = [2, 4, 8, 16]
NPAD = 262144        # padded/compile row width for all fixtures


def cfg_for(cs, logits):
    cfg = K37.pick_config(torch.float32, 1, NPAD)
    cfg["cluster_size"] = cs
    if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
        cfg["use_256bit_load"] = False
    return cfg


def run_one(fn, logits, pre, n_valid, cr, K):
    sl = torch.full((1,), n_valid * cr, dtype=torch.int32, device=DEV)
    oi = torch.empty(1, K, dtype=torch.int32, device=DEV)
    fn(logits, pre, sl, None, oi, None)
    torch.cuda.synchronize()
    return exact_set(oi, logits[0], K, n_valid)


def main():
    g = torch.Generator(device=DEV).manual_seed(20260722)
    fails, total = [], 0
    for K, cr in KCRS:
        base_row = torch.randn(1, NPAD, generator=g, device=DEV)
        topk = torch.topk(base_row[0], K)
        pre_hit = topk.indices.to(torch.int32).reshape(1, K).contiguous()
        pre_miss = (torch.topk(-base_row[0], K).indices
                    .to(torch.int32).reshape(1, K).contiguous())
        noisy = base_row[0] + 0.5 * torch.randn(NPAD, generator=g, device=DEV)
        pre_noise = (torch.topk(noisy, K).indices
                     .to(torch.int32).reshape(1, K).contiguous())
        for cs in CSS:
            cfg = cfg_for(cs, base_row)
            n_short = max(2 * K + 64, (NPAD // cs) // 2)   # fits one CTA slice
            short_row = base_row.clone()
            short_row[0, n_short:] = float("-inf")         # only [0,n_short) valid
            pre_short = (torch.topk(short_row[0, :n_short], K).indices
                         .to(torch.int32).reshape(1, K).contiguous())
            cases = [
                ("hit1.0", base_row, pre_hit, NPAD),
                ("miss", base_row, pre_miss, NPAD),
                ("noise", base_row, pre_noise, NPAD),
                ("degrade", short_row, pre_short, n_short),
            ]
            for arm, flags in ARMS:
                try:
                    fn = compile_arm(K, cr, cfg, flags)
                except Exception as e:
                    fails.append(f"K{K}/cs{cs}/{arm}: COMPILE {e!r}")
                    total += len(cases)
                    continue
                for name, lg, pre, nv in cases:
                    total += 1
                    try:
                        ok = run_one(fn, lg.contiguous(), pre, nv, cr, K)
                    except Exception as e:
                        ok = False
                        name += f" EXC {e!r}"
                    tag = f"K{K}/cs{cs}/{arm}/{name}"
                    print(f"  {tag}: {'OK' if ok else 'FAIL'}", flush=True)
                    if not ok:
                        fails.append(tag)
    print(f"\n[d1a-fixtures] {total - len(fails)}/{total} OK; "
          f"FAILS: {fails or 'none'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
