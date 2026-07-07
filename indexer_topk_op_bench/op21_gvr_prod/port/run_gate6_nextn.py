#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Gate 6 standalone: next_n (MTP rows) + request-level varlen seq_lens on
the ASSEMBLED kernel. The contract code was inherited verbatim from the
production kernel body, but the op21 bench never exercised next_n > 1 —
this closes UPSTREAM_ASSESSMENT §5 item 3's validation gap.

Contract (kernel body): rows = n_req * next_n; per row r,
req = r // next_n, kv = seq_lens[req] - next_n + r % next_n + 1,
N_eff = kv (cr=1) or kv // cr (cr=4); cr=1 additionally offsets every
preIdx pointer by r % next_n + 1 (diagonal convention).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from validate_port import ms_port, msc_port  # noqa: E402
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
from real_data_v2 import value_metrics  # noqa: E402


def main():
    ok = bad = 0
    g = torch.Generator(device="cpu").manual_seed(123)
    for K, cr in ((512, 4), (1024, 4), (2048, 1)):
        for next_n in (2, 4):
            n_req = 3
            rows = n_req * next_n
            N_max = 131072
            lg = torch.randn(rows, N_max, generator=g).float().cuda()
            kv_lens = torch.tensor(
                [N_max * cr - 5, N_max * cr - cr * 1000 - 3,
                 N_max * cr // 2 + 7][:n_req], dtype=torch.int32)
            sl = kv_lens.cuda()
            pre = torch.randint(0, N_max // 2 - next_n - 2, (n_req, K),
                                generator=g, dtype=torch.int32).cuda()
            for tag in ("ms", "C4"):
                if tag == "ms":
                    o = ms_port(lg, pre, sl, K, cr, next_n=next_n)
                else:
                    o = msc_port(lg, pre, sl, K, cr, C=4, next_n=next_n)
                torch.cuda.synchronize()
                row_bad = 0
                for r in range(rows):
                    req, nn_i = r // next_n, r % next_n
                    kv = int(kv_lens[req]) - next_n + nn_i + 1
                    N_eff = kv if cr == 1 else kv // cr
                    ref = torch.topk(lg[r, :N_eff].float(), K).indices
                    vd, _rc, ng = value_metrics(
                        o[r:r + 1], lg[r:r + 1, :N_eff].float(), ref, K)
                    u = torch.unique(o[r][o[r] >= 0]).numel()
                    if not (vd == 0 and ng == 0 and u == K):
                        row_bad += 1
                        print(f"FAIL {tag} K{K} cr{cr} nn{next_n} row{r}: "
                              f"vd={vd:.2e} u={u} N_eff={N_eff}")
                ok += row_bad == 0
                bad += row_bad != 0
    print(f"[gate6] next_n varlen: {ok} ok / {bad} fail")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
