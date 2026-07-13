# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rung-0 crux probe (PLAN P1/P2): a pure bf16 copy with the same read+write
traffic as RMSNorm = the effective-bandwidth ceiling for any candidate.
NOT a candidate; never gated/shipped."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_rmsnorm import get_inputs  # noqa: F401,E402

_out = {}


def kernel_fn(x, w):
    key = x.shape
    if key not in _out:
        _out[key] = torch.empty_like(x)
    torch.mul(x, 1.0, out=_out[key])   # real elementwise kernel (copy_ lowers to a DtoD memcpy)
    return _out[key]
