# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op46 workspace mirror of src_cuda/main.cpp B2 (L15-37) + run_ws checks
(L107-114) and kernel.h workspace_bytes contract.

B2 semantics mirrored exactly:
  * ONE zero-initialised slab workspace per device, lazily allocated through
    the torch caching allocator (main.cpp:32-33 `at::zeros(..., kByte)`);
  * keep-alive store (`ws_keep[GVR_MAX_DEV]`) -> module dict `_ws_keep`
    (tensor refcount = keep-alive, same as the C static array);
  * double-checked locking: lock-free hot-path load (a GIL-atomic dict get
    plays the `std::memory_order_acquire` load, main.cpp:26-27), slow path
    re-checks under a mutex (main.cpp:28-31);
  * device index bounds `0 <= d < GVR_MAX_DEV` (main.cpp:24-25) -- checked
    BEFORE the CUDA-ness of the tensor, exactly like the C binding (run()
    resolves the default workspace before run_impl's B1 checks, so a CPU
    logits tensor dies here with "device index out of range: -1").

Concurrent STREAMS on one device that may both take the multi-CTA SPLIT path
must pass their own workspace via run_ws() (main.cpp:16-17).

Size: gvr_topk_workspace_bytes() = GVR_WS_BUF_OFF + MAXC*GCAP*sizeof(int2)
    = 2048 + 160*16384*8 = 20,973,568 B (kernel.cu L44-46).

Kernel-facing view: ct_main's compiled signature takes the workspace as a
1-D contiguous int32 tensor (fake tensor dtype Int32, assumed_align=16 --
torch caching-allocator bases are 256B-aligned so the default slab always
satisfies it).  `kernel_view()` reproduces the C binding's raw
`workspace.data_ptr()` semantics for arbitrary user tensors by aliasing the
underlying storage at the tensor's byte offset.
"""

import threading

import torch

GVR_MAX_DEV = 64                      # kernel.cu L19 / main.cpp:19
_MAXC = 160                           # kernel.cu L17
_GCAP = 16384                         # kernel.cu L18
_GVR_WS_BUF_OFF = 2048                # kernel.cu L43
WS_BYTES = _GVR_WS_BUF_OFF + _MAXC * _GCAP * 8   # 20,973,568 (kernel.cu L44-46)
assert WS_BYTES == 20_973_568

_mu = threading.Lock()                # main.cpp:28 slow-path mutex
_ws_keep = {}                         # device index -> keep-alive int32 view


def workspace_bytes() -> int:
    """kernel.h:12 gvr_topk_workspace_bytes()."""
    return WS_BYTES


def default_workspace(ref: torch.Tensor) -> torch.Tensor:
    """main.cpp:23-37 default_workspace(ref) -> per-device cached slab.

    Returns the kernel-facing 1-D int32 view (zero-initialised on first use;
    the kernel restores the zeros it consumes, so one zeroing suffices for
    the lifetime of the cache entry)."""
    d = ref.get_device()
    if not (0 <= d < GVR_MAX_DEV):
        raise RuntimeError(f"device index out of range: {d}")
    ws = _ws_keep.get(d)              # hot path: one (GIL-atomic) load
    if ws is not None:
        return ws
    with _mu:                         # slow path: double-checked
        ws = _ws_keep.get(d)
        if ws is not None:
            return ws
        # lazy zeros via the torch caching allocator (at::zeros kByte,
        # main.cpp:32-33), viewed int32 for the DSL launch signature.
        buf = torch.zeros(WS_BYTES, dtype=torch.uint8, device=ref.device)
        ws = buf.view(torch.int32)
        _ws_keep[d] = ws              # keep-alive (ws_keep[d] = tensor)
        return ws


def validate_run_ws(workspace: torch.Tensor, logits: torch.Tensor) -> None:
    """main.cpp:107-114 run_ws() workspace hardening, same predicate order:
    CUDA + same device as logits; numel*element_size >= workspace_bytes();
    base 8-byte aligned."""
    if not (workspace.is_cuda and workspace.get_device() == logits.get_device()):
        raise RuntimeError("workspace must be a CUDA tensor on the same device")
    if workspace.numel() * workspace.element_size() < WS_BYTES:
        raise RuntimeError(f"workspace too small: need {WS_BYTES} bytes")
    if workspace.data_ptr() & 7:
        raise RuntimeError("workspace must be 8-byte aligned")


def kernel_view(workspace: torch.Tensor) -> torch.Tensor:
    """Raw-pointer semantics of the C binding (main.cpp:115 passes
    workspace.data_ptr() and nothing else): alias the first WS_BYTES bytes at
    the tensor's data_ptr() as int32[WS_BYTES/4], ignoring dtype/shape.

    NOTE: the DSL-side fake tensor declares assumed_align=16; a workspace at
    8-but-not-16-byte alignment passes the C-contract check above but is
    rejected by the DSL at conversion -- surfaced as a launch failure with
    shape context by ct_op (documented in notes/ct_op_NOTES.md)."""
    if (workspace.dtype is torch.int32 and workspace.dim() == 1
            and workspace.is_contiguous()
            and workspace.storage_offset() == 0
            and workspace.numel() == WS_BYTES // 4):
        return workspace              # already the canonical view
    off_bytes = workspace.storage_offset() * workspace.element_size()
    if off_bytes & 3:
        # unreachable past the 8B-alignment check for allocator-backed
        # storages; kept as a hard error rather than silent misalias.
        raise RuntimeError("workspace storage offset must be 4-byte aligned")
    t = torch.empty(0, dtype=torch.int32, device=workspace.device)
    t.set_(workspace.untyped_storage(), off_bytes // 4, (WS_BYTES // 4,))
    return t


def _reset_for_tests() -> None:
    """Drop cached slabs (tests only; NOT part of the C contract)."""
    with _mu:
        _ws_keep.clear()


__all__ = ["GVR_MAX_DEV", "WS_BYTES", "workspace_bytes", "default_workspace",
           "validate_run_ws", "kernel_view"]
