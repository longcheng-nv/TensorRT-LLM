#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op36 sweep — sweep_rival with the op36 arm registry monkeypatched in.
Everything else (NVTX c|/w| ranges, cold-L2 evict, 20/50 reps, cell-resumable
jsonl, folded exactness) is the rival harness verbatim. CLI == sweep_rival."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH / "op26_r0_upstream_port_report" / "rival_harness"))

import ops_op36                                   # noqa: E402 (also sets paths)
import sweep_rival as SR                          # noqa: E402

SR.build_call_rival = ops_op36.build_call_op36
SR.ops_for_rival = ops_op36.ops_for_op36

if __name__ == "__main__":
    sys.exit(SR.main())
