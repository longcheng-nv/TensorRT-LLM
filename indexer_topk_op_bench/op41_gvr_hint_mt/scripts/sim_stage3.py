# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41: offline simulation of v3 P1 (stage1 trim -> stage2 64-bin rungs)
WITH and WITHOUT a stage-3 32-sub-bin refinement, over ALL hint-path layers.
Metric: one-pass success = any rung count in [K, kC] on the first P2 pass.
Also simulates HS (hint subsampling) 1 and 2."""
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

LAYERS = {"flash": list(range(2, 43, 2)), "pro": list(range(2, 61, 2)),
          "v32": list(range(3, 61))}
QT8 = [10, 25, 45, 65, 82, 94]   # AR8 quantile percents (of Ks)
QT6 = [15, 40, 70, 92]           # AR6


def sim_rungs(g, K, qt_pct, hs, stage3):
    """Replicate P1 rung construction. g = hint values (torch 1D cuda)."""
    gsub = g[::hs]
    Ks = gsub.numel()
    hmin, hmax = gsub.min().item(), gsub.max().item()
    if not (hmax - hmin > 0):
        return [hmin]
    # stage 1: 64-bin trim at 97%
    binw = (hmax - hmin) / 64.0
    b1 = torch.clamp(((gsub - hmin) / binw).long(), 0, 63)
    h1 = torch.bincount(b1, minlength=64)
    suf = torch.flip(torch.cumsum(torch.flip(h1, [0]), 0), [0])
    qtrim = (Ks * 97) // 100
    tlow = hmin
    for b in range(63, -1, -1):
        if suf[b].item() >= qtrim:
            tlow = hmin + binw * b
            break
    if not (hmax - tlow > 0):
        return [hmax, hmin]
    # stage 2: 64-bin over [tlow, hmax], rung = lower bin edge at CCDF >= qt
    binw2 = (hmax - tlow) / 64.0
    sel = gsub[gsub >= tlow]
    b2 = torch.clamp(((sel - tlow) / binw2).long(), 0, 63)
    h2 = torch.bincount(b2, minlength=64)
    suf2 = torch.flip(torch.cumsum(torch.flip(h2, [0]), 0), [0])
    rungs = [hmax + (hmax - tlow)]
    for pct in qt_pct:
        target = (Ks * pct) // 100
        rung = tlow  # target below trim range fallback
        for b in range(63, -1, -1):
            if suf2[b].item() >= target:
                rung = tlow + binw2 * b
                if stage3:  # 32 sub-bins within [rung, rung+binw2)
                    sub = sel[(sel >= rung) & (sel < rung + binw2)]
                    above = suf2[b + 1].item() if b < 63 else 0
                    need = target - above
                    if sub.numel() and need > 0:
                        sbw = binw2 / 32.0
                        b3 = torch.clamp(((sub - rung) / sbw).long(), 0, 31)
                        h3 = torch.bincount(b3, minlength=32)
                        suf3 = torch.flip(torch.cumsum(torch.flip(h3, [0]),
                                                       0), [0])
                        for sb in range(31, -1, -1):
                            if suf3[sb].item() >= need:
                                rung = rung + sbw * sb
                                break
                break
        rungs.append(rung)
    rungs.append(hmin)
    return rungs


def main():
    tot = Counter()
    print("model,isl,L,K,onepass_s2_hs1,onepass_s3_hs1,onepass_s3_hs2")
    for model, isl in sorted({(m, i) for m, i, _ in all_cells()}):
        for L in LAYERS[model]:
            try:
                b = bundle(model, isl, L)
            except Exception:
                continue
            K = b["K"]
            lg = b["logits"][0].float().cuda()
            if lg.numel() <= 12288:
                continue
            kC = 8192 if K >= 2048 else 6144
            pre = b["preIdx"][0].to(torch.int64).cuda()
            pre = pre[(pre >= 0) & (pre < lg.numel())]
            g = lg[pre]
            srt = lg.sort(descending=True).values
            qt = QT8 if K != -1 else QT6  # envelope tiers mostly AR8/AR6;
            # AR8 is the denser/common ladder — simulate it
            res = {}
            for tag, hs, s3 in (("s2_hs1", 1, False), ("s3_hs1", 1, True),
                                ("s3_hs2", 2, True)):
                rungs = sim_rungs(g, K, qt, hs, s3)
                ok = 0
                for t in rungs:
                    c = int((lg >= t).sum().item())
                    if K <= c <= kC:
                        ok = 1
                        break
                res[tag] = ok
                tot[(tag, ok)] += 1
            print(f"{model},{isl},{L},{K},{res['s2_hs1']},{res['s3_hs1']},"
                  f"{res['s3_hs2']}", flush=True)
    print("\n== one-pass coverage ==", file=sys.stderr)
    for k in sorted(tot):
        print(f"  {k[0]} ok={k[1]}: {tot[k]}", file=sys.stderr)


if __name__ == "__main__":
    main()
