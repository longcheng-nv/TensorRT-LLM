# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op46 operator entry: CuTeDSL mirror of src_cuda/main.cpp run()/run_ws()/
workspace_bytes() (spec section 1).

B1 hardening checks run in the SAME ORDER with the SAME PREDICATES as
main.cpp:43-88 (run_impl):
  1. all three tensors CUDA (main.cpp:43-44)
  2. dtypes: logits f32, pre_idx i32, indices i32 (45-47)
  3. all 2-D (48-49)
  4. all contiguous (50-51)
  5. n_valid unwrap (57-67): python-int fast path (strict integral cast, like
     pybind cast<int64_t>); Tensor path checks
     torch.cuda.is_current_stream_capturing() FIRST and fails loudly (B1d),
     else .item() (the D2H sync)
  6. b/npad from logits, k = pre_idx.size(1) (68-70)
  7. b == 0 -> early no-op (71, B1f)
  8. npad % 4 == 0 (74-75, B1e float4 row loads)
  9. logits base 16-byte aligned (76-78)
 10. pre_idx/indices batch dims match (79-81)
 11. indices width >= k (84-85)
 12. n_valid >= 0 (86)
 13. n = min(nv, npad) clamped in unbounded ints BEFORE any narrowing (88)

Dispatch: ct_dispatch.route(b, n, npad, k) -> compile-cache keyed on
(kernel family, constexpr tuple) inside each family module -> BIND-ONCE
launch cache keyed on the shape key (b, n, npad, k): caches the compiled
callable + the prebuilt runtime-scalar arg pack as plain Python ints (probe
P12: plain ints, never pre-wrapped cutlass.Int32; pre-binding removes only
route()/marshal-prep work -- the tvm-ffi per-argument cost is paid every
call).  Hot enqueue target ~3-6 us (P12 arg-width tax); measured numbers in
notes/ct_op_NOTES.md.

Error contract (spec 1.4): launch failures surface as exceptions WITH
(b, n, npad, k) context, mirroring main.cpp:94-95.

All four family modules are imported LAZILY (first shape that routes to
them), so a missing/broken sibling only fails when actually routed to, with
(b, n, npad, k) context.  Wired compiled ABIs (verified against each
module's __call__ signature):
  ct_reg     (logits, pre_idx, out, n, CMP, QC, smem_bytes)
  ct_main    (logits, pre_idx, out, ws, n, npad, k, SCAP_, CMP_, R, SMP,
              TGT, Q, SS2, TGT2)             [only family taking workspace]
  ct_clus    (logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP, TGT, Q,
              SS2, TGT2)                      [get_compiled keyed +scap/cmp_]
  ct_regclus (logits, pre_idx, out, n)
"""

import operator

import torch

import ct_workspace
from ct_dispatch import route

WS_BYTES = ct_workspace.WS_BYTES

# shape key (b, n, npad, k) -> (fn, args tuple of python ints, needs_ws)
_LAUNCH_CACHE = {}

# hot-path local bindings (each torch.<attr> lookup costs ~0.1 us; the B1
# battery runs on EVERY call — mirror of main.cpp's "sub-100ns predicted
# branches" intent within Python's reach; measured in notes/ct_op_NOTES.md)
_F32 = torch.float32
_I32 = torch.int32
_TENSOR = torch.Tensor
_is_capturing = torch.cuda.is_current_stream_capturing
_index = operator.index
_ws_hot = ct_workspace._ws_keep          # shared dict object (hot-path load)
_GVR_MAX_DEV = ct_workspace.GVR_MAX_DEV


def workspace_bytes() -> int:
    """main.cpp:122-123 workspace_bytes() binding."""
    return ct_workspace.workspace_bytes()


# ---------------------------------------------------------------------------
# per-family launcher builders (cold path: once per distinct shape key)
# ---------------------------------------------------------------------------
def _build_launcher(b, n, npad, k):
    rd = route(b, n, npad, k)
    fam = rd['kernel']
    tpl = tuple(rd['tpl'])
    rt = rd['rt']
    if fam in ('reg', 'regimg'):
        import ct_reg
        fn = ct_reg.get_compiled(tpl)
        # compiled ABI: (logits, pre_idx, out, n, CMP, QC, smem_total)
        args = (rt['n'], rt['CMP'], rt['QC'],
                ct_reg.STATIC_BYTES + rd['smem'])
        return (fn, args, False)
    if fam == 'main':
        import ct_main
        fn = ct_main.get_compiled(tpl)
        # compiled ABI: (logits, pre_idx, out, ws, n, npad, k, SCAP_, CMP_,
        #                R, SMP, TGT, Q, SS2, TGT2)  [SCAP_/CMP_ dead, ABI parity]
        args = (rt['n'], rt['npad'], rt['k'], rt['SCAP_'], rt['CMP_'],
                rt['R'], rt['SMP'], rt['TGT'], rt['Q'], rt['SS2'], rt['TGT2'])
        return (fn, args, True)
    if fam == 'clus':
        import ct_clus
        # compile key carries the smem-extent scalars (scap/cmp_); compiled
        # ABI: (logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP, TGT, Q,
        #       SS2, TGT2) -- NO workspace (spec §4c)
        fn = ct_clus.get_compiled(tpl, scap=rt['SCAP'], cmp_=rt['CMP'])
        args = (rt['n'], rt['npad'], rt['k'], rt['SCAP'], rt['CMP'],
                rt['SMP'], rt['TGT'], rt['Q'], rt['SS2'], rt['TGT2'])
        return (fn, args, False)
    if fam == 'reg_clus':
        import ct_regclus
        # compiled ABI: (logits, pre_idx, out, n) -- smem/k derived in-module
        fn = ct_regclus.get_compiled(tpl)
        return (fn, (rt['n'],), False)
    # unreachable: route() only emits the five families above
    raise RuntimeError(f"unknown dispatch family {fam!r}")


# ---------------------------------------------------------------------------
# run_impl mirror (main.cpp:39-96)
# ---------------------------------------------------------------------------
def _run_impl(logits, pre_idx, n_valid, indices, ws):
    if not (logits.is_cuda and pre_idx.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if logits.dtype is not _F32:
        raise RuntimeError("logits must be float32")
    if pre_idx.dtype is not _I32:
        raise RuntimeError("pre_idx must be int32")
    if indices.dtype is not _I32:
        raise RuntimeError("indices must be int32")
    lsh, psh, ish = logits.shape, pre_idx.shape, indices.shape
    if not (len(lsh) == 2 and len(psh) == 2 and len(ish) == 2):
        raise RuntimeError("logits/pre_idx/indices must be 2-D")
    if not (logits.is_contiguous() and pre_idx.is_contiguous()
            and indices.is_contiguous()):
        raise RuntimeError("tensors must be contiguous")

    # n_valid unwrap (main.cpp:57-67): tensor path = D2H sync, illegal under
    # CUDA graph capture -- fail loudly instead of crashing the capture (B1d).
    if isinstance(n_valid, _TENSOR):
        if _is_capturing():
            raise RuntimeError(
                "tensor n_valid requires a D2H sync, illegal under CUDA "
                "graph capture — pass n_valid as a python int")
        nv = int(n_valid.item())
    else:
        # strict integral cast (pybind cast<int64_t> rejects floats/strings)
        nv = _index(n_valid)

    b, npad = lsh
    k = psh[1]
    if b == 0:                       # empty batch: no-op (main.cpp:71, B1f)
        return
    if npad & 3:
        raise RuntimeError(
            f"npad (logits stride) must be a multiple of 4, got {npad}")
    if logits.data_ptr() & 15:
        raise RuntimeError(
            "logits base must be 16-byte aligned (storage-offset views "
            "break the float4 row loads)")
    if psh[0] != b or ish[0] != b:
        raise RuntimeError(
            f"batch dims must match: logits {b} pre_idx {psh[0]} "
            f"indices {ish[0]}")
    if ish[1] < k:
        raise RuntimeError(
            f"indices width {ish[1]} < k={k} (k is pre_idx.size(1))")
    if nv < 0:
        raise RuntimeError(f"n_valid must be non-negative, got {nv}")
    # clamp BEFORE any narrowing (main.cpp:87-88; python ints are unbounded,
    # so min() is the exact 64-bit clamp)
    n = nv if nv < npad else npad

    # CUDA out-indexing mirror: every kernel derives O = out + row*k
    # (kernel.cu L475/L1309 etc.) -- flat PACKED rows, ignoring the actual
    # indices width.  The DSL kernels index out[row, :] with the tensor's own
    # row stride, so a wider `indices` must be re-viewed packed (pure view,
    # no copy; contiguity already checked).
    if ish[1] != k:
        indices = indices.reshape(-1)[:b * k].view(b, k)

    key = (b, n, npad, k)
    lc = _LAUNCH_CACHE.get(key)
    try:
        if lc is None:
            lc = _build_launcher(b, n, npad, k)
            _LAUNCH_CACHE[key] = lc
        fn, args, needs_ws = lc
        if needs_ws:
            fn(logits, pre_idx, indices, ws, *args)
        else:
            fn(logits, pre_idx, indices, *args)
    except Exception as e:
        raise RuntimeError(
            f"gvr_topk launch failed (b={b} n={n} npad={npad} k={k}): "
            f"{e}") from e


# ---------------------------------------------------------------------------
# exports (main.cpp:98-124)
# ---------------------------------------------------------------------------
def run(logits, pre_idx, n_valid, indices):
    """Fast 4-arg form: signature-identical to the original candidate.
    Default per-device slab workspace resolved FIRST (main.cpp:99-102 --
    a CPU logits tensor therefore dies with 'device index out of range').
    Hot path inlines the C binding's check + atomic-load + cache-hit
    (main.cpp:24-27); the slow path allocates under ct_workspace's lock."""
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:            # main.cpp:25, EVERY call
        raise RuntimeError(f"device index out of range: {d}")
    ws = _ws_hot.get(d)
    if ws is None:
        ws = ct_workspace.default_workspace(logits)
    _run_impl(logits, pre_idx, n_valid, indices, ws)


def run_ws(logits, pre_idx, n_valid, indices, workspace):
    """Explicit-workspace form for multi-stream callers (main.cpp:105-116)."""
    ct_workspace.validate_run_ws(workspace, logits)
    _run_impl(logits, pre_idx, n_valid, indices,
              ct_workspace.kernel_view(workspace))


__all__ = ["run", "run_ws", "workspace_bytes", "WS_BYTES"]
