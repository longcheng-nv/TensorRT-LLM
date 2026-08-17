# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op46 DPS entry shim for the op44 harness build_candidate contract.

quick_ab_op44.build_candidate() loads this file standalone via
importlib.util.spec_from_file_location (mangled module name, arbitrary cwd),
then calls the module-level run(logits, pre_idx, n_valid, indices) --
destination-passing style, k = pre_idx.size(1), result written into
`indices`.

The shim bootstraps sys.path so the sibling src/ modules import by absolute
location, and defers ALL heavy imports (torch/cutlass via ct_op's lazy
per-family imports) to first call: kernels being written concurrently
(ct_clus / ct_regclus) only fail if a shape actually routes to them.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_ct_op = None


def _op():
    global _ct_op
    if _ct_op is None:
        import ct_op
        _ct_op = ct_op
    return _ct_op


def run(logits, pre_idx, n_valid, indices):
    """GVR exact top-k decode (CuTeDSL, prod-hardened) — main.cpp run()."""
    _op().run(logits, pre_idx, n_valid, indices)
    return indices


def run_ws(logits, pre_idx, n_valid, indices, workspace):
    """Explicit-workspace form for multi-stream callers — main.cpp run_ws()."""
    _op().run_ws(logits, pre_idx, n_valid, indices, workspace)
    return indices


def workspace_bytes():
    """Slab workspace size in bytes (per concurrent stream)."""
    return _op().workspace_bytes()


__all__ = ["run", "run_ws", "workspace_bytes"]
