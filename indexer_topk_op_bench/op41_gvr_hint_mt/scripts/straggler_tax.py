# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41: bound the straggler tax on STOCK v3 — paired event timing of
(a) hetero batch over ALL layers vs (b) batch cycled over p0-only layers
(passes==0 per the DBG probe at BS=n). The (a)-(b) gap is the max attainable
win from a perfect one-pass ladder. Same batch size, same npad, same K."""
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, timeit, build  # noqa: E402
from v3_pass_probe import build_dbg  # noqa: E402

GROUPS = [("v32", "64k"), ("v32", "16k"), ("v32", "256k"), ("v32", "128k"),
          ("pro", "512k"), ("pro", "1024k"), ("pro", "256k"),
          ("flash", "512k")]
BS_LIST = [16, 64, 256, 1024]
LAYERS = {"flash": list(range(2, 43, 2)), "pro": list(range(2, 61, 2)),
          "v32": list(range(3, 61))}


def main():
    dbg = build_dbg()
    v3 = build("kernel_bs")
    print("model,isl,BS,n_layers,n_p0,hetero_us,p0only_us,tax")
    for model, isl in GROUPS:
        rows, npad_ref = [], None
        for L in LAYERS[model]:
            try:
                b = bundle(model, isl, L)
            except Exception:
                continue
            npad = b["logits"].shape[1]
            if npad_ref is None:
                npad_ref = npad
            if npad != npad_ref:
                continue
            rows.append((b["logits"][0], b["preIdx"][0], b["N"]))
        n = len(rows)
        lg_all = torch.stack([r[0] for r in rows]).cuda()
        pre_all = torch.stack([r[1] for r in rows]).cuda()
        n_valid = max(r[2] for r in rows)
        K = pre_all.shape[1]
        # per-layer pass count at BS=n (each layer once)
        out = torch.full((n, K), -7, dtype=torch.int32, device="cuda")
        dbg.run(lg_all, pre_all, n_valid, out)
        torch.cuda.synchronize()
        passes = [int(x) // 10 for x in out[:, 0].cpu()]
        p0 = [i for i, p in enumerate(passes) if p == 0]
        if not p0:
            print(f"{model},{isl},-,{n},0,,,no-p0-layers", flush=True)
            continue
        for bs in BS_LIST:
            ih = (torch.arange(bs) % n)
            ip = torch.tensor(p0)[torch.arange(bs) % len(p0)]
            th = {}
            for tag, idx in (("h", ih), ("p", ip)):
                lg = lg_all[idx].contiguous()
                pre = pre_all[idx].contiguous()
                o = torch.empty((bs, K), dtype=torch.int32, device="cuda")
                v3.run(lg, pre, n_valid, o)
                torch.cuda.synchronize()
                best = None
                for _ in range(5):
                    us = timeit(lambda: v3.run(lg, pre, n_valid, o), reps=7)
                    best = us if best is None or us < best else best
                th[tag] = best
            print(f"{model},{isl},{bs},{n},{len(p0)},{th['h']:.1f},"
                  f"{th['p']:.1f},{th['h'] / th['p']:.3f}", flush=True)


if __name__ == "__main__":
    main()
