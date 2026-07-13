# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iter3 ship candidate: regime dispatch (1 rule, budget <= 3).

  tokens <= 512  -> single-pass Triton kernel (1 CTA/row, fp32 accum, autotuned
                    num_warps) — wins the latency/occupancy regime (iter1:
                    1.047/0.995/1.025 at T=1/16/256)
  tokens  > 512  -> flashinfer.norm.rmsnorm — owns the BW-saturated regime
                    (WALLS: flashinfer large-T BW-efficiency edge; iter2
                    falsified all Triton config repairs there)

Threshold rationale: crossover lies in the (256, 4096] grid gap; 512 keeps a
2x safety margin below the first losing measured cell. Out-of-place,
CUDA-graph compatible (dispatch is host-side on shape, resolved at capture).
No incumbent source edits — flashinfer is called, never modified.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_rmsnorm import EPS, get_adversarial_inputs, get_inputs, reference_fn  # noqa: F401,E402

import flashinfer  # noqa: E402

from candidate_triton import kernel_fn as _triton_kernel_fn  # noqa: E402

DISPATCH_TOKENS_MAX_TRITON = 512  # rule 1 of 1


def kernel_fn(x, w):
    if x.shape[0] <= DISPATCH_TOKENS_MAX_TRITON:
        return _triton_kernel_fn(x, w)
    return flashinfer.norm.rmsnorm(x, w, eps=EPS)
