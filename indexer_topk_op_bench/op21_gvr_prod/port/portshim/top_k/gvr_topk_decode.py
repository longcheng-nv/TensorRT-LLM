# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Validation shim for blackwell/top_k/gvr_topk_decode.py (see ../utils.py).

Upstream main has the #15198 cluster helpers unified into gvr_topk_decode.py;
the bench vendored tree keeps them in gvr_topk_decode_cluster.py — re-export
both so the assembled MS kernel's imports resolve identically.
"""
from cute_vendored.blackwell.top_k.gvr_topk_decode import (  # noqa: F401
    GvrParams,
    _fmin_f32_inline,
    float_as_uint32,
)
from cute_vendored.blackwell.top_k.gvr_topk_decode_cluster import (  # noqa: F401
    ld_shared_cluster_f32,
    ld_shared_cluster_i32,
    mapa_shared_cluster,
)
