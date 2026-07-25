# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""i10-H offline: no-sample prior pick — for each (K, npad-band), find the
fixed rung index minimizing re-streams across all 865 cells; compare vs the
sampled pick's 12/865."""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_patho import phase1, AR, SS  # noqa: E402
from ab import parse_cell, all_cells  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402


def cellinfo(cell):
    model, isl, L = parse_cell(cell)
    mod = v32 if model == "v32" else v4
    b = mod.get_bundle(model, isl, L, "fp32")
    K, Npad = b["K"], b["Npad"]
    row = b["logits"][0].float().cpu().numpy()[:Npad]
    pre = b["preIdx"][0].cpu().numpy()[:K]
    v4._bundle_cache.clear(); v32._bundle_cache.clear()
    kC = 8192 if K >= 2048 else 6144
    hints = row[np.clip(pre, 0, Npad - 1)]
    rungs, hmin, hmax = phase1(hints, K)
    cnt = np.array([(row >= t).sum() for t in rungs], np.int64)
    return K, Npad, kC, cnt


def band(npad):
    for b in (12288, 20480, 32832, 65600, 131136, 262208, 1 << 30):
        if npad <= b:
            return b
    return 1 << 30


def main():
    cells = all_cells()
    rows = []
    for c in cells:
        try:
            K, Npad, kC, cnt = cellinfo(c)
        except Exception as e:
            print("[skip]", c, e); continue
        rows.append((c, K, Npad, kC, cnt))
    groups = defaultdict(list)
    for c, K, Npad, kC, cnt in rows:
        groups[(K, band(Npad))].append((c, kC, cnt))
    tot_rs = 0
    for (K, b), lst in sorted(groups.items()):
        best_j, best_rs = None, 1 << 30
        for j in range(AR):
            rs = sum(1 for _, kC, cnt in lst if not (K <= cnt[j] <= kC))
            if rs < best_rs:
                best_j, best_rs = j, rs
        tot_rs += best_rs
        print(f"K={K:5d} npad<={b:8d}: cells={len(lst):3d} best_rung={best_j} "
              f"restreams={best_rs}")
    print(f"\nTOTAL restreams with per-(K,band) fixed rung: {tot_rs}/{len(rows)}"
          f"  (sampled pick: 12/865)")


if __name__ == "__main__":
    main()
