# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op33 conditional dispatch on op27_hls (D2 M-reduction). *** RETRACTED / NO-SHIP (iter6) ***
Clean single-GPU paired A/B (2026-07-14) shows M=3 REGRESSES the WORST scenario (K1024 N32768
worst 0.787; K512 N262144 worst 0.727) — the iter5 "+9%" was a real-only/N<=65536 measurement
artifact + 8-GPU contention. FAILS the ship rule. Kept only as a falsified reference. DO NOT SHIP.

Original (WRONG) rationale below.

Verdict (nsys, BS=1 fp32): qfracs M=3 (0.85,0.35) beats op27_hls default M=4
for K in {512,1024} (+2.6..14.3%, geomean ~1.09) but LOSES for K2048
(0.89..0.996×, which needs the op27 tail ladder). So:

    dispatch_op33(K) = M=3 (0.85,0.35)  iff K < 2048  else  op27_hls default

ONE dispatch rule. Baseline byte-identical for K2048. Both the single-CTA
sandwich (gvr_ms) and cluster (gvr_msc) paths read _qfracs_for(K) which honors
OP25_QFRACS, so the override is applied via a scoped env set at build time
(compiled kernels are cached by qfracs; the env read happens only at compile).
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OPB = _HERE.parents[1]  # indexer_topk_op_bench
sys.path.insert(0, str(_OPB / "op21_gvr_prod" / "src"))
sys.path.insert(0, str(_OPB / "harness"))
from gvr_msc_op import gvr_ms_auto  # noqa: E402

_M3_QFRACS = "0.85,0.35"
_M3_KS = (512, 1024)


def gvr_ms_auto_op33(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                     out=None):
    if int(index_topk) in _M3_KS:
        old = os.environ.get("OP25_QFRACS")
        os.environ["OP25_QFRACS"] = _M3_QFRACS
        try:
            return gvr_ms_auto(logits, pre_idx, seq_lens, index_topk,
                               compress_ratio, out=out)
        finally:
            if old is None:
                os.environ.pop("OP25_QFRACS", None)
            else:
                os.environ["OP25_QFRACS"] = old
    return gvr_ms_auto(logits, pre_idx, seq_lens, index_topk, compress_ratio,
                       out=out)
