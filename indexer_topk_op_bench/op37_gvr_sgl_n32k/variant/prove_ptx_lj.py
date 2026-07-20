# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""[op37-lj] default-off PTX byte-identity proof.

Compiles the SAME launch-shape variant (fp32, K=1024, cs=4, T=1024,
256-bit loads, mbpm from pick_config at BS=2 N=131072) from
  arm A: the PRISTINE gvrpkgprod2 (op26 p4f1_harness snapshot)
  arm B: gvrpkg37 with tight_bracket left DEFAULT OFF, imported under a
         module ALIAS directory named `gvrpkgprod2` (symlink) so the CUDA
         kernel entry name — which embeds the module path — matches arm A
         and the PTX is byte-comparable (same trick as the dp4 proof,
         logs/ptx_gvrpkg37_defaultoff_aliased.ptx).

Usage: python3 prove_ptx_lj.py A|B <out.ptx>
Then md5-compare the two outputs.
"""
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent.parent

os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")

arm, out_path = sys.argv[1], sys.argv[2]
if arm == "A":
    sys.path.insert(0, str(_BENCH / "op26_r0_upstream_port_report" / "p4f1_harness"))
else:
    alias_root = Path(tempfile.mkdtemp(prefix="ljalias_"))
    (alias_root / "gvrpkgprod2").symlink_to(_HERE / "gvrpkg37")
    sys.path.insert(0, str(alias_root))

import torch  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as _crt  # noqa: E402

from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402

torch.cuda.init()
TOP_K = 1024
cfg = GvrTopKKernel.pick_config(torch.float32, 2, 131072)
cfg["cluster_size"] = 4  # contract shape: fp32 K1024 N=131072 BS=2 cs=4
kernel = GvrTopKKernel(
    dtype=cutlass.Float32,
    top_k=TOP_K,
    next_n=1,
    compress_ratio=1,
    return_output_values=False,
    **cfg,
)
n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
in_align = 32 if cfg["use_256bit_load"] else 16
input_fake = _crt.make_fake_compact_tensor(
    kernel.dtype, (n_rows, n_cols), stride_order=(1, 0), assumed_align=in_align
)
pre_idx_fake = _crt.make_fake_compact_tensor(
    cutlass.Int32, (n_batch, TOP_K), stride_order=(1, 0), assumed_align=16
)
seq_lens_fake = _crt.make_fake_compact_tensor(cutlass.Int32, (n_batch,), stride_order=(0,))
out_indices_fake = _crt.make_fake_compact_tensor(
    cutlass.Int32, (n_rows, TOP_K), stride_order=(1, 0), assumed_align=16
)
fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
compiled = cute.compile(
    kernel,
    input_fake,
    pre_idx_fake,
    seq_lens_fake,
    None,
    out_indices_fake,
    None,
    stream=fake_stream,
    options="--enable-tvm-ffi",
)
ptx = getattr(compiled, "__ptx__", None)
if ptx is None:
    raise SystemExit("no __ptx__ attribute; inspect CUTE_DSL_KEEP_PTX dump dir")
if not isinstance(ptx, str):
    ptx = str(ptx)
Path(out_path).write_text(ptx)
print(f"arm {arm}: cfg={cfg} ptx_bytes={len(ptx)} -> {out_path}")
