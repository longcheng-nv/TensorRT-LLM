# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41: saturation-verdict verification on BS>1 HETEROGENEOUS batches.

Batch = all GVR-active layers of the same (model, isl) real capture (distinct
rows, distinct h / hint distributions), cycled to BS. DBG_PASSES build emits
per-row: out[row,0] = passes*10 + fell_to_exact_descent.

Reports the per-row pass histogram over every hint-path (model, isl) group x
BS in {16, 64, 256, 1024}. A nonzero tail here reopens the ladder campaign."""
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
from v3_pass_probe import build_dbg  # noqa: E402

BS_LIST = [16, 64, 256, 1024]
LAYERS = {"flash": list(range(2, 43, 2)), "pro": list(range(2, 61, 2)),
          "v32": list(range(3, 61))}


def main():
    mod = build_dbg()
    groups = sorted({(m, i) for m, i, _ in all_cells()})
    print("model,isl,BS,nrows_distinct,npad,pass_hist,descent_rows")
    total = Counter()
    for model, isl in groups:
        rows = []
        npad_ref = None
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
        if npad_ref is None or npad_ref <= 12288:
            print(f"{model},{isl},-,{len(rows)},{npad_ref},direct-path,skip")
            continue
        n = len(rows)
        lg_all = torch.stack([r[0] for r in rows]).cuda()
        pre_all = torch.stack([r[1] for r in rows]).cuda()
        n_valid = max(r[2] for r in rows)
        for bs in BS_LIST:
            idx = torch.arange(bs) % n
            lg = lg_all[idx].contiguous()
            pre = pre_all[idx].contiguous()
            K = pre.shape[1]
            out = torch.full((bs, K), -7, dtype=torch.int32, device="cuda")
            mod.run(lg, pre, n_valid, out)
            torch.cuda.synchronize()
            v = out[:, 0].cpu()
            hist = Counter((int(x) // 10) for x in v)
            desc = int(sum(1 for x in v if int(x) % 10 == 1))
            total.update(hist)
            hs = " ".join(f"p{k}:{c}" for k, c in sorted(hist.items()))
            print(f"{model},{isl},{bs},{n},{npad_ref},{hs},{desc}",
                  flush=True)
    print("\n== TOTAL per-row pass histogram (hint-path groups, all BS) ==",
          file=sys.stderr)
    for k, c in sorted(total.items()):
        print(f"  passes={k}: {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
