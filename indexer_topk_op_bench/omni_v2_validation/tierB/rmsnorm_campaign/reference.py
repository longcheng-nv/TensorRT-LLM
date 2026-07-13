# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Campaign reference implementation (SKILL Phase 0.1) — doubles as the
'eager torch RMSNorm (fp32 upcast)' rival: kernel_fn == reference_fn."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common_rmsnorm import (  # noqa: F401,E402
    EPS,
    HIDDEN,
    TOKEN_GRID,
    get_adversarial_inputs,
    get_inputs,
    make_inputs,
    reference_fn,
)

kernel_fn = reference_fn
