# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Splice the D2b probe into gvrpkgp4probe (a copy of the [p4sub] twin).

Packs (cnt[b*] << 32) | (K - rank_above) into the unused phase_ts slot 15,
stamped by thread0 right after the coarse 3-step search publishes
b*/rank_above. Degenerate branches leave slot 15 at 0 (probe decodes 0 as
'pipeline not reached'). One extra smem read + shift + store on thread0 —
timing-irrelevant, and this package is probe-only anyway.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "gvrpkgp4probe" / "top_k" / "gvr_topk_decode.py"
src = TARGET.read_text()

if "[d2bprobe]" in src:
    raise SystemExit("[d2bprobe] already spliced")
assert "[p4sub]" in src, "expected the [p4sub] twin as the base"

OLD = """            cute.arch.barrier()
            # [p4sub] s12: after coarse 3-step high->low bin search
            if tidx == cutlass.Int32(0):
                phase_ts_row[12] = cute.arch.clock64()
            b_star = s_iscalars[3]
            rank_above = s_iscalars[2]
"""
NEW = """            cute.arch.barrier()
            # [p4sub] s12: after coarse 3-step high->low bin search
            if tidx == cutlass.Int32(0):
                phase_ts_row[12] = cute.arch.clock64()
            b_star = s_iscalars[3]
            rank_above = s_iscalars[2]
            # [d2bprobe] slot 15 = (cnt[b*] << 32) | (K - rank_above)
            if tidx == cutlass.Int32(0):
                cntb_p = smem_hist[b_star]
                need_p = cutlass.Int32(kK) - rank_above
                phase_ts_row[15] = (
                    (cutlass.Int64(cntb_p) << cutlass.Int64(32))
                    | cutlass.Int64(need_p)
                )
"""
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)
TARGET.write_text(src)
print("[d2bprobe] spliced")
