# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41: order-stat ladder one-pass coverage over ALL layers (not just the 3
bench layers) of every hint-path (model, isl) group — the straggler layers
found by v3_pass_probe_bs live here. kC matches v3: 8192 for K>=2048 else
6144."""
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle  # noqa: E402
from bs38_nsys import all_cells  # noqa: E402

RANK_FRACS = [1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 7 / 8, 1.0]
LAYERS = {"flash": list(range(2, 43, 2)), "pro": list(range(2, 61, 2)),
          "v32": list(range(3, 61))}


def main():
    groups = sorted({(m, i) for m, i, _ in all_cells()})
    tot = Counter()
    print("model,isl,L,K,npad,h,onepass6,onepass8,best_c")
    for model, isl in groups:
        for L in LAYERS[model]:
            try:
                b = bundle(model, isl, L)
            except Exception:
                continue
            K = b["K"]
            lg = b["logits"][0].float().cuda()
            npad = lg.numel()
            if npad <= 12288:
                continue
            kC = 8192 if K >= 2048 else 6144
            pre = b["preIdx"][0].to(torch.int64).cuda()
            pre = pre[(pre >= 0) & (pre < npad)]
            g = lg[pre]
            gs = g.sort(descending=True).values
            vK = lg.sort(descending=True).values[K - 1].item()
            h = (g >= vK).sum().item() / K
            cs = []
            for f in RANK_FRACS:
                m = max(1, min(gs.numel(), int(round(f * K))))
                t = gs[m - 1].item()
                cs.append(int((lg >= t).sum().item()))
            ok6 = any(K <= c <= kC
                      for c, f in zip(cs, RANK_FRACS)
                      if f in (1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8, 1.0))
            ok8 = any(K <= c <= kC for c in cs)
            bc = min((c for c in cs if c >= K), default=-1)
            tot[("6", ok6)] += 1
            tot[("8", ok8)] += 1
            print(f"{model},{isl},{L},{K},{npad},{h:.3f},{int(ok6)},"
                  f"{int(ok8)},{bc}", flush=True)
    print("\n== one-pass coverage over ALL hint-path layers ==",
          file=sys.stderr)
    for k in sorted(tot):
        print(f"  ladder{k[0]} ok={k[1]}: {tot[k]}", file=sys.stderr)


if __name__ == "__main__":
    main()
