#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Re-run the upstream test_cute_dsl_gvr_topk_decode main grid against the
ASSEMBLED gvr_topk_decode_ms.py (UPSTREAM_ASSESSMENT §5 PR-1 acceptance:
"re-run the upstream 30-case adversarial suite" — this drives the FULL
main-test grid, which contains those cases).

Inputs + tie-aware reference are the upstream helpers VERBATIM
(_upstream_test_helpers.py). Grid mirrors the upstream parametrize axes,
minus seqlen_sorted=True (sort-indirect mode is deliberately NOT in the
PR-1 first step — the ms runner path is non-LB / non-sort; see
UPSTREAM_ASSESSMENT §5 item 3). cluster_size axis: 1 -> GvrMsKernel,
4 -> GvrMsClusterKernel C4 (forced, even at N=4096 where the production
dispatch would pick single-CTA — exactness must hold anyway).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _upstream_test_helpers import _make_inputs, _tie_aware_check  # noqa: E402
from validate_port import ms_port, msc_port  # noqa: E402


def main():
    quick = "--quick" in sys.argv
    PAIRS = [(torch.bfloat16, 512), (torch.bfloat16, 1024),
             (torch.float16, 1024), (torch.float32, 2048)]
    ok = bad = skipped = 0
    fails = []
    for dtype, top_k in PAIRS:
        for N in (4096, 65536):
            for varlen in (False, True):
                for next_n in (1, 2):
                    for batch_size in (1, 32):
                        for cr in (1, 4):
                            for hit in ((0.5,) if quick else (0.0, 0.5)):
                                for cs in (1, 4):
                                    if N - next_n + 1 < top_k:
                                        skipped += 1
                                        continue
                                    if varlen and batch_size < 2:
                                        skipped += 1
                                        continue
                                    num_rows = batch_size * next_n
                                    lg, pre, sl = _make_inputs(
                                        num_rows, N, top_k, dtype, next_n,
                                        seed=42, compress_ratio=cr,
                                        preidx_hit_rate=hit, varlen=varlen)
                                    out = torch.empty(num_rows, top_k,
                                                      dtype=torch.int32,
                                                      device="cuda")
                                    if cs == 1:
                                        ms_port(lg, pre, sl, top_k, cr,
                                                next_n=next_n, out=out)
                                    else:
                                        msc_port(lg, pre, sl, top_k, cr, C=4,
                                                 next_n=next_n, out=out)
                                    torch.cuda.synchronize()
                                    tag = (f"{dtype} K{top_k} N{N} vl{int(varlen)} "
                                           f"nn{next_n} bs{batch_size} cr{cr} "
                                           f"hit{hit} cs{cs}")
                                    try:
                                        _tie_aware_check(out, lg, sl, top_k,
                                                         next_n,
                                                         compress_ratio=cr)
                                        ok += 1
                                    except AssertionError as e:
                                        bad += 1
                                        fails.append(tag)
                                        print(f"FAIL {tag}: {e}")
    print(f"upstream-grid vs ported ms kernel: {ok} ok / {bad} fail "
          f"({skipped} skipped as upstream does)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
