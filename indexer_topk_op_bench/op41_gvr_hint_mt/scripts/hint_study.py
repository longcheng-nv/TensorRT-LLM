# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41 phase-0 crux: what do preIdx hint ORDER STATISTICS know about the
true K-th value, per envelope cell?

Per cell (75 real-capture cells of the op39 envelope):
  h        = hit rate  |preIdx ∩ topK| / K   (offline truth, NOT a kernel input)
  v_K      = true K-th largest logit
  c(t_m)   = row count >= t_m for hint-rank thresholds m in RANKS
             (t_m = m-th largest of logits[preIdx]; m=K == min-hint)
  c_cur    = count at the op39 production threshold policy
             max(min-hint, sample-quantile@2K target) -- approximated here by
             min(2K-target count, c(min-hint)) reference columns.
Key questions:
  Q1 fixed rank: is there an m with c(t_m) in [K, CAP] across cells? (expect NO)
  Q2 ladder+iterate: given the exact count feedback after one collect at
     t_{m0}, how many cells need 0/1/2 extra rounds to land in [K, CAP]?
  Q3 domain split: which (cell, BS) are L2-resident (npad*BS*4 <= 100MB)?
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle  # noqa: E402
from bs38_nsys import all_cells  # noqa: E402

CAP = 8192
RANK_FRACS = [1 / 8, 1 / 4, 1 / 2, 3 / 4, 7 / 8, 1.0]  # of K; 1.0 == min-hint


def main():
    cells = all_cells()
    print("cell,K,npad,h,vK,c_minhint," +
          ",".join(f"c_m{int(f * 8)}d8" for f in RANK_FRACS[:-1]) +
          ",rank_hK_count")  # count at t = (hK)-th hint (oracle rank)
    rows = []
    for model, isl, L in cells:
        b = bundle(model, isl, L)
        K, N = b["K"], b["N"]
        lg = b["logits"][0].float().cuda()
        npad = lg.numel()
        pre = b["preIdx"][0].to(torch.int64).cuda()
        pre = pre[(pre >= 0) & (pre < npad)]
        g = lg[pre]
        gs = g.sort(descending=True).values
        srt = lg.sort(descending=True).values
        vK = srt[K - 1].item()
        h = (g >= vK).sum().item() / K
        cnt = {}
        for f in RANK_FRACS:
            m = max(1, min(gs.numel(), int(round(f * K))))
            t = gs[m - 1].item()
            # count via searchsorted on descending sort
            c = int((lg >= t).sum().item())
            cnt[f] = c
        mh = max(1, min(gs.numel(), int(round(h * K))))
        c_hk = int((lg >= gs[mh - 1].item()).sum().item())
        out = (f"{model}_{isl}_L{L:02d},{K},{npad},{h:.3f},{vK:.5f},"
               f"{cnt[1.0]}," +
               ",".join(str(cnt[f]) for f in RANK_FRACS[:-1]) +
               f",{c_hk}")
        print(out, flush=True)
        rows.append((model, isl, L, K, npad, h, cnt))

    # ---- aggregate verdicts ----
    print("\n== Q1 fixed-rank viability: cells with c in [K, CAP] ==",
          file=sys.stderr)
    n = len(rows)
    for f in RANK_FRACS:
        ok = sum(1 for *_, K, np_, h, c in rows if K <= c[f] <= CAP)
        under = sum(1 for *_, K, np_, h, c in rows if c[f] < K)
        over = sum(1 for *_, K, np_, h, c in rows if c[f] > CAP)
        print(f"  m={f:.3f}K: ok {ok}/{n}  undershoot {under}  "
              f"overflow {over}", file=sys.stderr)
    print("\n== Q2 2-round ladder: round1 m=K/2; if under -> m=K, if over "
          "-> exact t2 from stored (existing machinery) ==", file=sys.stderr)
    r1ok = sum(1 for *_, K, np_, h, c in rows if K <= c[0.5] <= CAP)
    r1under = sum(1 for *_, K, np_, h, c in rows if c[0.5] < K)
    r1over = sum(1 for *_, K, np_, h, c in rows if c[0.5] > CAP)
    minfix = sum(1 for *_, K, np_, h, c in rows
                 if c[0.5] < K and c[1.0] <= CAP)
    print(f"  round1 ok {r1ok}/{n}; under {r1under} (min-hint fixes "
          f"{minfix}); over {r1over}", file=sys.stderr)
    print("\n== h distribution ==", file=sys.stderr)
    hs = sorted(h for *_, h, _ in [(r[5], r[6]) for r in rows])
    import statistics
    print(f"  min {hs[0]:.3f} p25 {hs[n // 4]:.3f} med "
          f"{statistics.median(hs):.3f} p75 {hs[3 * n // 4]:.3f} "
          f"max {hs[-1]:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
