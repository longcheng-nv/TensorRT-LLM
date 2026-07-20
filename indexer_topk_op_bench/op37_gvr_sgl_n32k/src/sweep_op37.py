#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op37 sweep — sweep_rival with the op37 arm registry monkeypatched in.
CLI == sweep_rival (cell-resumable jsonl, NVTX c|/w|, cold-L2, exactness)."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH / "op26_r0_upstream_port_report" / "rival_harness"))

import ops_op37                                   # noqa: E402
import sweep_rival as SR                          # noqa: E402

SR.build_call_rival = ops_op37.build_call_op37
SR.ops_for_rival = ops_op37.ops_for_op37

if __name__ == "__main__":
    sys.exit(SR.main())
