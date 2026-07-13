# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incumbent: flashinfer.norm.rmsnorm (flashinfer 0.6.11) — TRT-LLM production default.

A/B is ALWAYS against this module (SKILL 0.2). Out-of-place, CUDA-graph compatible.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_rmsnorm import EPS, get_adversarial_inputs, get_inputs, reference_fn  # noqa: F401,E402

import flashinfer  # noqa: E402


def kernel_fn(x, w):
    return flashinfer.norm.rmsnorm(x, w, eps=EPS)
