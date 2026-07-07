# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Validation shim: stands in for tensorrt_llm/_torch/cute_dsl_kernels/
blackwell/utils.py so the assembled gvr_topk_decode_ms.py can be imported
and exactness-gated against the bench's vendored modules (no tensorrt_llm
install needed). Requires <bench>/ops on sys.path."""
from cute_vendored.blackwell.utils import (  # noqa: F401
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)
