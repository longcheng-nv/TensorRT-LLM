# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""i10-D offline: replay pivot pick (band-first + 2-sigma escape, iter8b
semantics) with parametric tgt multiplier; report re-streams + candidate-count
(P4 cost proxy) across all 82 grid cells."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_patho import phase1, AR, SS, MAXPASS  # noqa: E402
from ab import parse_cell, all_cells  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402


def pick(est, K, kC, tmul):
    lo, hi = (3 * K) // 2, (6 * kC) // 10
    tgt = max(min(int(tmul * K), hi), lo)
    best, bestd = AR - 1, 1 << 60
    for j in range(AR):
        if lo <= est[j] <= hi:
            d = abs(int(est[j]) - tgt)
            if d < bestd:
                best, bestd = j, d
    if bestd == 1 << 60:  # iter8b 2-sigma escape
        for j in range(AR):
            e = int(est[j])
            if e <= 0:
                continue
            g = 2.0 * np.sqrt(SS * e)
            if e - g >= K and e + g <= kC:
                d = abs(e - tgt)
                if d < bestd:
                    best, bestd = j, d
    return best


def run(cell, tmuls):
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
    f4 = row[: (Npad // 4) * 4].reshape(-1, 4)
    samp = f4[::SS].ravel()
    est = np.array([SS * (samp >= t).sum() for t in rungs], np.int64)
    out = {}
    for tm in tmuls:
        j = pick(est, K, kC, tm)
        C = int(cnt[j])
        out[tm] = (j, C, K <= C <= kC)
    return K, kC, out


def main():
    tmuls = [3.0, 2.5, 2.0]
    cells = [c for c in all_cells()]
    stats = {tm: dict(restream=0, csum=0, cmax=0, changed=0) for tm in tmuls}
    bad = []
    for cell in cells:
        try:
            K, kC, r = run(cell, tmuls)
        except Exception as e:
            print(f"[skip] {cell}: {e}")
            continue
        j0, C0, ok0 = r[3.0]
        for tm in tmuls:
            j, C, ok = r[tm]
            s = stats[tm]
            s["csum"] += C
            s["cmax"] = max(s["cmax"], C)
            if not ok:
                s["restream"] += 1
                if tm != 3.0:
                    bad.append((cell, tm, j, C, K, kC))
            if j != j0:
                s["changed"] += 1
    n = len(cells)
    for tm in tmuls:
        s = stats[tm]
        print(f"tgt={tm}K: restreams={s['restream']}/{n} picks-changed="
              f"{s['changed']} avgC={s['csum']//n} maxC={s['cmax']}")
    for b_ in bad[:15]:
        print("  RESTREAM", b_)


if __name__ == "__main__":
    main()
