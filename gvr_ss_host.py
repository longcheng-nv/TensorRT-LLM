# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-sampling GVR top-K decode — host side (dispatch, workspace, entry).

Companion to ``gvr_topk_decode_self_sampling.py`` (the device module).
Three sections:

1. dispatch — the CUDA host dispatch as a pure function
   ``route(b, n, npad, k)``;
2. workspace — one zero-initialised per-device slab (20,973,568 B) via the
   torch caching allocator, with keep-alive + double-checked locking;
3. operator entry — ``run(logits, pre_idx, n_valid, indices)`` /
   ``run_ws(..., workspace)`` DPS forms with input hardening and a
   bind-once launch cache keyed on ``(b, n, npad, k)``.

OPERATOR CONTRACT (batch-uniform entries): ``n_valid`` is one host python
int for the whole batch — every row shares the same valid prefix, in
COMPRESSED index space (the caller applies any ``compressRatio`` division).
``pre_idx`` is consumed as-is — raw prev-step top-K indices, uniformly for
DSv3.2 / DSv4 Flash / Pro. The +1 temporal shift ``heuristicTopKDecode.cu``
applies for cr==1 is deliberately dropped: hints only steer the sampling
ladder (exactness never depends on them), and raw prev-step hints overlap
the current top-K at least as well as +1-shifted ones on real decode data,
so one offset-free hint convention serves all three models. The production
per-row contract (per-request ``kv_lens`` read on-device, per-row MTP
offsets — sync-free and CUDA-graph-replay safe with growing KV) is
implemented by ``run_varlen``, which is the entry the opt-in DSA dispatch
seam calls. The batch-uniform ``run``/``run_ws`` entries keep the simpler
contract (one host-side ``n_valid`` for the whole batch), are exercised for
unit tests and benchmarking only, and must not be substituted for
``run_varlen`` under continuous batching, MTP (``next_n > 1``), or
CUDA-graph capture.
"""

import math
import operator
import threading
from collections.abc import Sequence

import torch

_dev_mod = None


def _device():
    """Lazy import of the merged device module (first routed shape compiles;
    a broken/absent device module only fails when actually reached)."""
    global _dev_mod
    if _dev_mod is None:
        try:
            from . import gvr_ss as _m  # in-tree
        except ImportError:  # standalone dir
            import gvr_ss as _m
        _dev_mod = _m
    return _dev_mod


_dev_mod_v4 = None


def _device_v4():
    """Lazy import of the 8-byte-vector (bf16x4) device module twin. Same
    kernel families and fp32 arm as the primary module; its bf16 arm keeps the
    fp32 tile geometry (4 elements per vector), which the bf16 route table
    selects for deep-split streaming rows whose per-CTA chunk would leave a
    16-byte tile half idle."""
    global _dev_mod_v4
    if _dev_mod_v4 is None:
        try:
            from . import gvr_ss_v4 as _m4  # in-tree
        except ImportError:  # standalone dir
            import gvr_ss_v4 as _m4
        _dev_mod_v4 = _m4
    return _dev_mod_v4


# ===========================================================================
# ==== dispatch =============================================================
# ===========================================================================
"""Pure-Python mirror of the GVR CUDA host dispatch (gvr_topk_launch).

route(b, n, npad, k) is a PURE function of its four ints -- no env knobs, no
GPU, stdlib only.  It returns the kernel family, its compile-time template
tuple, the runtime scalar pack `rt`, grid/cluster/block geometry, smem size,
and whether the family needs the workspace.

rt carries the FULL runtime scalar list each kernel receives, in signature
order, always starting with (n, npad, k).

Dead ABI-parity args: gvr_main's `int SCAP_, int CMP_` params are NEVER read
by the kernel body -- it recomputes them as constexprs that mirror the host
formulas bit-identically.  They are kept in rt purely for ABI parity.
gvr_clus's SCAP/CMP are LIVE runtime args.  `aim` and `SFAC` are host-side
intermediates only (never cross the ABI), so they do not appear in rt.

C-semantics notes encoded here:
  * every `/` on ints is C truncating division -> Python `//` (all operands
    are non-negative on every reachable path);
  * `sel = (long long)SFAC * n / aim` and the TGT/TGT2 products are 64-bit in C;
    Python ints are exact, so `//` reproduces them;
  * `int r = (int)(0.5 + sqrt((double)(6LL*n)))` truncates toward zero after
    the +0.5 -> `int(0.5 + math.sqrt(float(6*n)))`;
  * `IMGW = (n + 3) & ~3` four-element float4 round-up;
  * the reg-block CMP (possibly widened to n by DEGE) is scoped to the
    register-resident block; the streaming path re-derives its own CMP.
"""


# ---- dispatch constants (must match the device kernels) ---------------------
NB = 1024  # register-path histogram bins
QUADC = 96  # crossing-bin O(mc^2) rank gate (streaming/reg paths)
SNB = 256  # streaming-path bin count
CMPC = 4096  # crossing-bin slots per CTA, clustered register path
BLKC = 1024  # CTA size of the clustered register path


def route(b: int, n: int, npad: int, k: int) -> dict[str, object]:
    """Mirror of the CUDA gvr_topk_launch dispatch. Pure. See module doc."""
    if b < 1:
        raise RuntimeError(f"route requires b >= 1, got {b}")
    wide = b <= 148

    # ======================= register-resident block ========================
    n4 = n >> 2
    CMP = n if n < 2560 else 2560
    QC = 1024 if b > 148 else QUADC
    CURE = not (n < 2 * k and b > 148)
    DEGE = (n <= 3 * k) or (n <= 4 * k + 64)
    if DEGE and CMP < n:
        CMP = n
    NBSEL = (2 * NB) if (n4 > 512 and not (n4 <= 1024 and not wide)) else NB
    IMGOFF = NBSEL
    smem_reg = (NBSEL + 2 * CMP) * 4

    def _reg(BLK, VPT, MINB, NBH):
        # DEG wins over the CUR flag; DEG forces KPT=1, else KPT ladder 1/2/4.
        if DEGE:
            tpl = (BLK, VPT, MINB, 1, CURE, True, False, NBH)
        else:
            kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else 4)
            tpl = (BLK, VPT, MINB, kpt, CURE, False, False, NBH)
        return {
            "kernel": "reg",
            "tpl": tpl,
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # full ABI
                "CMP": CMP,
                "IMGOFF": IMGOFF,
                "QC": QC,
            },
            "grid": (b, 1),
            "cluster": 1,
            "block": BLK,
            "smem": smem_reg,
            "ws": False,
        }

    IMGW = (n + 3) & ~3
    smi = (NBSEL + (2 * CMP if 2 * CMP > IMGW else IMGW)) * 4
    IMGE = wide and (not DEGE) and k <= 1024

    if n4 <= 256:
        return _reg(256, 1, 8, NB)
    if n4 <= 512:
        return _reg(512, 1, 4, NB)
    if n4 <= 1024:
        if wide:
            if IMGE:
                # regimg launch: gvr_topk_reg<1024,1,2,1,true,false,true,2048>
                return {
                    "kernel": "regimg",
                    "tpl": (1024, 1, 2, 1, True, False, True, 2 * NB),
                    "rt": {
                        "n": n,
                        "npad": npad,
                        "k": k,  # full ABI
                        "CMP": CMP,
                        "IMGOFF": IMGOFF,
                        "QC": QC,
                    },
                    "grid": (b, 1),
                    "cluster": 1,
                    "block": 1024,
                    "smem": smi,
                    "ws": False,
                }
            return _reg(1024, 1, 2, 2 * NB)
        return _reg(512, 2, 4, NB)

    # ---- clustered register-resident path ----
    if n4 > 4096 and n4 <= 8 * BLKC * 4 and k <= BLKC:
        av = 148 // (b if b > 0 else 1)  # truncating
        amax = 1
        while (amax << 1) <= av and amax < 8:
            amax <<= 1
        vsel = 0
        cs = 0
        if amax >= 2:
            # cs=8 co-residency veto: an 8-CTA cluster with b > 15 exceeds
            # GPC packing; such shapes fall through to the streaming path.
            for v in (1, 2, 4):
                c = 1  # 64-bit product in C
                while c * BLKC * v < n4:
                    c <<= 1
                if c == 8 and b > 15:  # the veto
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:
            smc = (3 * NB + 2 * CMPC) * 4
            return {
                "kernel": "reg_clus",
                "tpl": (BLKC, vsel, cs),
                "rt": {"n": n, "npad": npad, "k": k},  # dims only
                "grid": (cs, b),
                "cluster": cs,
                "block": BLKC,
                "smem": smc,
                "ws": False,
            }

    if n4 <= 4096 and wide:
        return _reg(1024, 4, 1, 2 * NB)

    # ====================== streaming / collect path ========================
    R = 1
    if b <= 32:
        r1 = 148 // b
        if r1 < 1:
            r1 = 1
        r2 = ((n >> 2) + 1023) // 1024
        if r2 < 1:
            r2 = 1
        R = r1 if r1 < r2 else r2
        if R < 1:
            R = 1
    elif b <= 74 and (n >> 2) >= 16384 and k <= 1024:  # shallow R=2 split
        R = 2

    useclus = False
    if 2 <= R <= 8 and k <= 1024:
        p2 = 1
        while (p2 << 1) <= R:
            p2 <<= 1
        # gvr_clus cs=8 hits the same GPC packing wall as the clustered
        # register path; same veto, same b > 15 threshold.
        if p2 == 8 and b > 15:
            p2 = 4
        R = p2
        useclus = True

    big = b * R <= 148
    SCAP = (16384 if R == 1 else 8192) if big else (8192 if k > 1024 else 4096)
    CMP = (4096 if k > 1024 else 2048) if big else 1024

    aim = (
        ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    q = 6 * n  # 6LL * n
    r = int(0.5 + math.sqrt(float(q)))  # C cast trunc
    if r > aim:
        aim = r
    SFAC = (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 else (64 if k >= 1024 else 32)
    amin = 3 * k if R == 2 else (7 * k) // 2
    if R > 1 and aim < amin:
        aim = amin
    if aim > (SCAP >> 1):
        aim = SCAP >> 1
    if aim < k:
        aim = k

    n4s = n >> 2
    SMP, SS2, TGT, TGT2 = 0, 1, 0, 0
    small_dense = (k > 1024) and (not big) and n <= SCAP and n > 2 * k
    if (n > SCAP or small_dense) and n4s >= 4:  # PAIR sample
        sel = SFAC * n // aim  # 64-bit
        if sel < 256:
            sel = 256
        if sel > n // 2:
            sel = n // 2
        pairs = sel >> 3
        if pairs < 1:
            pairs = 1
        half = n4s >> 1
        if half < 1:
            half = 1
        if pairs > half:
            pairs = half
        SS2 = half // pairs
        if SS2 < 1:
            SS2 = 1
        SMP = half // SS2
        if SMP < 1:
            SMP = 1
        TGT = (aim * (SMP * 8)) // n  # 64-bit
        if TGT < 1:
            TGT = 1
        TGT2 = (k * (SMP * 8)) // n  # 64-bit
        if TGT2 < 1:
            TGT2 = 1
    Q = (n4s + R - 1) // R

    if useclus:
        if n > SCAP and n4s >= 4:  # QUAD override
            sel = SFAC * n // aim
            if sel < 256:
                sel = 256
            if sel > n // 2:
                sel = n // 2
            quads = sel >> 4
            if quads < 1:
                quads = 1
            quarter = n4s >> 2
            if quarter < 1:
                quarter = 1
            if quads > quarter:
                quads = quarter
            SS2 = quarter // quads
            if SS2 < 1:
                SS2 = 1
            SMP = quarter // SS2
            if SMP < 1:
                SMP = 1
            TGT = (aim * (SMP * 16)) // n
            if TGT < 1:
                TGT = 1
            TGT2 = (k * (SMP * 16)) // n
            if TGT2 < 1:
                TGT2 = 1
        smc = SNB * 8 + (SCAP + 4) * 8 + CMP * 8
        per = Q >> 10
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        CS = 2 if R == 2 else (4 if R == 4 else 8)
        return {
            "kernel": "clus",
            "tpl": (1024, U, 1, SNB, CS),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # ABI (live)
                "SCAP": SCAP,
                "CMP": CMP,
                "SMP": SMP,
                "TGT": TGT,
                "Q": Q,
                "SS2": SS2,
                "TGT2": TGT2,
            },
            "grid": (CS, b),
            "cluster": CS,
            "block": 1024,
            "smem": smc,
            "ws": False,
        }

    smem_main = (SCAP + 4) * (8 if (R > 1 or b <= 296) else 4) + (CMP + 1) * 8

    def _main(BLK, MINB, U, SPLIT):
        # KPT ladder 1/2/4/8; grid = (R, b).
        kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else (4 if k <= 4 * BLK else 8))
        # TSH-floor staging gate.  The CUDA form is a grid-uniform RUNTIME
        # gate (gridDim.y > 15 && k <= 1024 && (n >> 2) <= 32768); here it
        # is a compile-time key -- per-launch semantics are identical
        # because the gate is uniform over the grid.
        tshg = bool(SPLIT) and b > 15 and k <= 1024 and (n >> 2) <= 32768
        return {
            "kernel": "main",
            "tpl": (BLK, U, MINB, SNB, kpt, SPLIT, tshg),
            # SCAP_/CMP_ are dead ABI-parity args: gvr_main never reads them
            # (it recomputes them as constexprs).
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,  # full ABI
                "SCAP_": SCAP,
                "CMP_": CMP,
                "R": R,
                "SMP": SMP,
                "TGT": TGT,
                "Q": Q,
                "SS2": SS2,
                "TGT2": TGT2,
            },
            "grid": (R, b),
            "cluster": 1,
            "block": BLK,
            "smem": smem_main,
            "ws": True,
        }

    if big:
        per = Q >> 10
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        return _main(1024, 1, U, R > 1)  # SPLIT iff R>1
    if b <= 296:
        return _main(512, 2, 8, False)
    return _main(256, 4, 8, False)


if __name__ == "__main__":
    smoke = [
        # (b, n, npad, k)                          expected family
        (64, 1024, 1024, 512),  # reg   n4<=256 rung (DEG: n<=3k)
        (64, 2048, 2048, 512),  # reg   n4<=512 rung
        (1024, 4096, 4096, 1024),  # reg   n4<=1024, b>148 -> (512,2,4)
        (64, 4096, 4096, 512),  # regimg wide !DEGE k<=1024
        (64, 4096, 4096, 1024),  # reg   wide but DEGE (n<=4k+64)
        (8, 65536, 65536, 1024),  # reg_clus (vsel=2, cs=8; b<=15 no veto)
        (16, 131072, 131072, 512),  # main  cs=8 veto fall-through -> SPLIT slab, tshg=True
        (64, 16384, 16384, 1024),  # reg   wide 4k fallback (1024,4,1)
        (64, 262144, 262144, 1024),  # clus  R=2 shallow cluster split
        (1, 1048576, 1048576, 1024),  # main  deep slab SPLIT R=148
        (20, 262144, 262144, 2048),  # main  k>1024 split (no useclus)
        (512, 131072, 131072, 1024),  # main  b>296 BLK=256
        (256, 6144, 6144, 2048),  # main  small_dense sample gate
        (256, 262144, 262144, 2048),  # main  KBIG-domain (k>1024), BLK=512 KPT=4
    ]
    for shp in smoke:
        print(shp, "->", route(*shp))


# ---------------------------------------------------------------------------
# two-time-scale dispatch split (per-row varlen / CUDA-graph groundwork)
# ---------------------------------------------------------------------------
# route(b, n, npad, k) factored into
#   route_static(b, n, npad, k)  — everything that must be frozen per launch:
#       family, compile tuple, grid, cluster, block, and the rt scalars that
#       change only at discrete n-thresholds;
#   route_dynamic(static, n)     — the n-continuous scalars a per-row kernel
#       recomputes from its own row length (the device code will mirror these
#       formulas): n, CMP (reg families), the sampling ladder
#       SMP/TGT/SS2/TGT2/Q (streaming families), and the reg-family smem
#       footprint.
# INVARIANT: merging route_dynamic back into route_static reproduces
# route() EXACTLY for every n. The policy of which n to freeze the static
# half at (e.g. max_seq_len) is a perf-only choice — the factorization
# itself is lossless.

_DYN_RT = {
    "reg": ("n", "CMP"),
    "regimg": ("n", "CMP"),
    "reg_clus": ("n",),
    "clus": ("n", "SMP", "TGT", "Q", "SS2", "TGT2"),
    "main": ("n", "SMP", "TGT", "Q", "SS2", "TGT2"),
}
_DYN_SMEM = ("reg", "regimg")  # smem depends on CMP/IMGW -> recomputed per n


def route_static(b: int, n: int, npad: int, k: int) -> dict[str, object]:
    """route() with the n-continuous fields redacted (see _DYN_RT/_DYN_SMEM).
    Constant on maximal n-intervals ("bands"); every redacted field is
    reconstructible from (static, n) by route_dynamic."""
    plan = route(b, n, npad, k)
    st = {key: (dict(val) if isinstance(val, dict) else val) for key, val in plan.items()}
    for f in _DYN_RT[st["kernel"]]:
        st["rt"].pop(f)
    if st["kernel"] in _DYN_SMEM:
        st.pop("smem")
    return st


def route_dynamic(static: dict[str, object], n: int) -> tuple[dict[str, object], int]:
    """Recompute the redacted n-continuous scalars from (static, n).
    Returns (rt_updates, smem). Must stay equivalent to route(); the
    device-side per-row engine mirrors exactly these formulas."""
    fam = static["kernel"]
    k = static["rt"]["k"]
    if fam in ("reg", "regimg"):
        dege = static["tpl"][5]
        cmp_ = n if dege else (n if n < 2560 else 2560)
        nbsel = static["rt"]["IMGOFF"]
        if fam == "regimg":
            imgw = (n + 3) & ~3
            smem = (nbsel + (2 * cmp_ if 2 * cmp_ > imgw else imgw)) * 4
        else:
            smem = (nbsel + 2 * cmp_) * 4
        return {"n": n, "CMP": cmp_}, smem
    if fam == "reg_clus":
        return {"n": n}, static["smem"]

    # streaming families (main / clus): the sampling-ladder scalars
    b = static["grid"][1]
    if fam == "clus":
        R = static["cluster"]
        scap = static["rt"]["SCAP"]
    else:
        R = static["rt"]["R"]
        scap = static["rt"]["SCAP_"]
    big = b * R <= 148
    aim = (
        ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    r_ = int(0.5 + math.sqrt(float(6 * n)))
    if r_ > aim:
        aim = r_
    sfac = (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 else (64 if k >= 1024 else 32)
    amin = 3 * k if R == 2 else (7 * k) // 2
    if R > 1 and aim < amin:
        aim = amin
    if aim > (scap >> 1):
        aim = scap >> 1
    if aim < k:
        aim = k

    n4s = n >> 2
    smp, ss2, tgt, tgt2 = 0, 1, 0, 0
    small_dense = (k > 1024) and (not big) and n <= scap and n > 2 * k
    if (n > scap or small_dense) and n4s >= 4:
        sel = sfac * n // aim
        sel = 256 if sel < 256 else sel
        sel = n // 2 if sel > n // 2 else sel
        pairs = max(sel >> 3, 1)
        half = max(n4s >> 1, 1)
        pairs = half if pairs > half else pairs
        ss2 = max(half // pairs, 1)
        smp = max(half // ss2, 1)
        tgt = max((aim * (smp * 8)) // n, 1)
        tgt2 = max((k * (smp * 8)) // n, 1)
    q_ = (n4s + R - 1) // R
    if fam == "clus" and n > scap and n4s >= 4:
        sel = sfac * n // aim
        sel = 256 if sel < 256 else sel
        sel = n // 2 if sel > n // 2 else sel
        quads = max(sel >> 4, 1)
        quarter = max(n4s >> 2, 1)
        quads = quarter if quads > quarter else quads
        ss2 = max(quarter // quads, 1)
        smp = max(quarter // ss2, 1)
        tgt = max((aim * (smp * 16)) // n, 1)
        tgt2 = max((k * (smp * 16)) // n, 1)
    return (
        {"n": n, "SMP": smp, "TGT": tgt, "Q": q_, "SS2": ss2, "TGT2": tgt2},
        static["smem"],
    )


def route_split(b: int, n: int, npad: int, k: int) -> dict[str, object]:
    """route_static + route_dynamic recombined — must equal route() exactly
    (the factorization fuzz in the unit tests asserts this)."""
    st = route_static(b, n, npad, k)
    dyn, smem = route_dynamic(st, n)
    plan = {key: (dict(val) if isinstance(val, dict) else val) for key, val in st.items()}
    plan["rt"].update(dyn)
    plan["smem"] = smem
    return plan


def route_streaming(
    b: int, n: int, npad: int, k: int, force_main: bool = False
) -> dict[str, object]:
    """route() restricted to its STREAMING half (main / clus) — the varlen
    capture policy: per-row kernels must be picked from the families that are
    correct for ANY row length, so the register-resident specialists are
    skipped even when the envelope n would normally land on them.  Where
    route() itself lands on main/clus this is IDENTICAL to route().
    force_main additionally skips the clus rounding, so the raw
    min(r1, r2) R matches the CUDA else-branch exactly."""
    if b < 1:
        raise RuntimeError(f"route_streaming requires b >= 1, got {b}")
    R = 1
    if b <= 32:
        r1 = max(148 // b, 1)
        r2 = max(((n >> 2) + 1023) // 1024, 1)
        R = max(min(r1, r2), 1)
    elif b <= 74 and (n >> 2) >= 16384 and k <= 1024:
        R = 2
    useclus = False
    if not force_main and 2 <= R <= 8 and k <= 1024:
        p2 = 1
        while (p2 << 1) <= R:
            p2 <<= 1
        if p2 == 8 and b > 15:
            p2 = 4
        R = p2
        useclus = True
    big = b * R <= 148
    scap = (16384 if R == 1 else 8192) if big else (8192 if k > 1024 else 4096)
    cmp_ = (4096 if k > 1024 else 2048) if big else 1024
    aim = (
        ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    r_ = int(0.5 + math.sqrt(float(6 * n)))
    if r_ > aim:
        aim = r_
    sfac = (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 else (64 if k >= 1024 else 32)
    amin = 3 * k if R == 2 else (7 * k) // 2
    if R > 1 and aim < amin:
        aim = amin
    if aim > (scap >> 1):
        aim = scap >> 1
    if aim < k:
        aim = k
    n4s = n >> 2
    smp, ss2, tgt, tgt2 = 0, 1, 0, 0
    small_dense = (k > 1024) and (not big) and n <= scap and n > 2 * k
    if (n > scap or small_dense) and n4s >= 4:
        sel = min(max(sfac * n // aim, 256), n // 2)
        pairs = min(max(sel >> 3, 1), max(n4s >> 1, 1))
        half = max(n4s >> 1, 1)
        ss2 = max(half // pairs, 1)
        smp = max(half // ss2, 1)
        tgt = max((aim * (smp * 8)) // n, 1)
        tgt2 = max((k * (smp * 8)) // n, 1)
    q_ = (n4s + R - 1) // R
    if useclus:
        if n > scap and n4s >= 4:
            sel = min(max(sfac * n // aim, 256), n // 2)
            quads = min(max(sel >> 4, 1), max(n4s >> 2, 1))
            quarter = max(n4s >> 2, 1)
            ss2 = max(quarter // quads, 1)
            smp = max(quarter // ss2, 1)
            tgt = max((aim * (smp * 16)) // n, 1)
            tgt2 = max((k * (smp * 16)) // n, 1)
        smc = SNB * 8 + (scap + 4) * 8 + cmp_ * 8
        per = q_ >> 10
        u_ = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        cs = 2 if R == 2 else (4 if R == 4 else 8)
        return {
            "kernel": "clus",
            "tpl": (1024, u_, 1, SNB, cs),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,
                "SCAP": scap,
                "CMP": cmp_,
                "SMP": smp,
                "TGT": tgt,
                "Q": q_,
                "SS2": ss2,
                "TGT2": tgt2,
            },
            "grid": (cs, b),
            "cluster": cs,
            "block": 1024,
            "smem": smc,
            "ws": False,
        }
    smem_main = (scap + 4) * (8 if (R > 1 or b <= 296) else 4) + (cmp_ + 1) * 8

    def _main(blk_, minb_, u_, split_):
        kpt = 1 if k <= blk_ else (2 if k <= 2 * blk_ else (4 if k <= 4 * blk_ else 8))
        tshg = bool(split_) and b > 15 and k <= 1024 and (n >> 2) <= 32768
        return {
            "kernel": "main",
            "tpl": (blk_, u_, minb_, SNB, kpt, split_, tshg),
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,
                "SCAP_": scap,
                "CMP_": cmp_,
                "R": R,
                "SMP": smp,
                "TGT": tgt,
                "Q": q_,
                "SS2": ss2,
                "TGT2": tgt2,
            },
            "grid": (R, b),
            "cluster": 1,
            "block": blk_,
            "smem": smem_main,
            "ws": True,
        }

    if big:
        per = q_ >> 10
        u_ = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        return _main(1024, 1, u_, R > 1)
    if b <= 296:
        return _main(512, 2, 8, False)
    return _main(256, 4, 8, False)


_VARLEN_CACHE = {}

# ---- prefill launcher cache ------------------------------------------------
# Prefill forces R==1 (route_streaming gives R>1 only for b<=74). The compiled
# launcher depends only on the row tier, k and the envelope bucket — never on the
# exact row count or npad — so the cache stays bounded on a long-running server.
_PREFILL_CACHE = {}
_PREFILL_ROW_SLAB = 32768  # gridDim.y <= 65535; slab so keys stay bounded
_PREFILL_TIER_ROWS = (75, 149, 297)  # (rows<=148, 149..296, >296) band reps


def _prefill_tier(rows: int) -> int:
    return 0 if rows <= 148 else 1 if rows <= 296 else 2


def _prefill_bucket(n_env: int) -> int:
    # pow2-quantize the envelope so a growing envelope reuses one plan; cap at
    # 32768 because U=8 for every n>=32768 on the tier-0 arm.
    return min(1 << max(int(n_env) - 1, 1).bit_length(), 32768)


def _prefill_cache_key(tier: int, k: int, n_bucket: int):
    # tiers 1/2 fix U, so the bucket does not change their engine — collapse it
    # to one key so warmup covers them with a single launch.
    return (tier, k, n_bucket if tier == 0 else 0)


def _prefill_launcher(tier: int, k: int, n_bucket: int) -> tuple:
    """Prefill plan + compiled launcher: ``_varlen_launcher``'s main branch with
    r_const=1, split=False and the prefill compile flag. SCAP_/CMP_ are envelope
    upper bounds; npad is filled per call in ``run_prefill``."""
    key = _prefill_cache_key(tier, k, n_bucket)
    hit = _PREFILL_CACHE.get(key)
    if hit is not None:
        return hit
    b_route = _PREFILL_TIER_ROWS[tier]
    n_route = max(n_bucket, k + 1)
    plan = route_streaming(b_route, n_route, n_route, k, force_main=True)
    if plan["kernel"] != "main":
        raise RuntimeError(f"prefill route did not land on gvr_main: {plan['kernel']}")
    rt = plan["rt"]
    if rt["R"] != 1:
        raise RuntimeError(f"prefill requires R==1 (got {rt['R']})")
    tpl = tuple(plan["tpl"])
    dev = _device()
    fn = dev.get_compiled(tpl[:6] + (False,) + (1, 0, 1), hint_free=True, prefill=True)
    big = tier == 0
    # r_const==1 branch of the _varlen_launcher tuning scalars
    aim_base = (
        (4 * k if k >= 1024 else 2 * k) if big else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    sfac = 64 if k >= 1024 else 32
    amin = (7 * k) // 2
    sd_en = 1 if (k > 1024 and not big) else 0
    tail = (aim_base, sfac, amin, sd_en, 0)  # tsh_en=0 (split=False)
    lc = ("main", fn, (rt["SCAP_"], rt["CMP_"]), tail)
    _PREFILL_CACHE[key] = lc
    return lc


def _varlen_launcher(
    num_rows: int,
    npad: int,
    k: int,
    n_env: int,
    next_n: int,
    cr: int,
) -> tuple:
    """Capture-time varlen plan + compiled launcher.  The gvr_main port is
    the universally correct fallback; specialist family tiers below.  Every
    choice here is a function of capture-stable quantities only — mirroring
    the in-tree runner's pick_tuning(graph_capture=...) discipline."""
    key = (num_rows, npad, k, n_env, next_n, cr)
    hit = _VARLEN_CACHE.get(key)
    if hit is not None:
        return hit
    # Two envelopes: the kernel gets the PHYSICAL bound (never past the row
    # stride, so a row whose kv length exceeds the logits width clamps and
    # takes the short path instead of reading into the next row); the router
    # gets the k+1 floor it needs to pick a non-degenerate family.
    n_kernel = min(n_env, npad)
    n_route = max(n_kernel, k + 1)
    cr_shift = 0 if cr == 1 else 2
    dev = _device()
    # ---- route() parity, family tier 1: clustered register-resident --------
    # Admit reg_clus exactly where the free route picks it; its whole
    # admission window (n4 <= 32768) fits capture-frozen envelopes. The
    # choice is a pure function of this cache key, so CUDA-graph replay
    # safety is unchanged; per-row n / short-row handling lives in-kernel.
    plan_free = route(num_rows, n_route, npad, k)
    if plan_free["kernel"] == "reg_clus":
        fn = dev.get_compiled__regclus(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
        )
        lc = ("reg_clus", fn, n_kernel)
        _VARLEN_CACHE[key] = lc
        return lc
    # ---- route() parity, family tier 2: register-resident (+img flavor) ----
    # Same admission rule as tier 1: exactly where the free route picks
    # reg/regimg (the whole small/mid-N band across all row counts). CMP/QC/
    # smem are envelope-derived launch constants -- in-kernel they are pure
    # capacity clamps (CMP), a fast-path threshold (QC) and the launch smem
    # size, all safe upper bounds for every per-row n <= envelope; per-row n
    # / short-row handling lives in-kernel.
    if plan_free["kernel"] in ("reg", "regimg"):
        fn = dev.get_compiled__reg(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
        )
        rt_f = plan_free["rt"]
        lc = (
            "reg",
            fn,
            (n_kernel, rt_f["CMP"], rt_f["QC"], dev.STATIC_BYTES + plan_free["smem"]),
        )
        _VARLEN_CACHE[key] = lc
        return lc
    # ---- route() parity, family tier 3: cluster split (clus) ---------------
    # Same admission rule: exactly where the free route picks clus (the
    # large-N mid-rows band). SCAP/CMP are launch-stable (pure functions of
    # rows/CS/k — never of n) so the envelope values are the per-row values;
    # the sampling-ladder scalars (SMP/TGT/Q/SS2/TGT2) are dead launch slots,
    # re-derived per row in-kernel by the route_dynamic clus mirror.
    # Per-row n / short-row handling in-kernel.
    if plan_free["kernel"] == "clus":
        rt_f = plan_free["rt"]
        fn = dev.get_compiled__clus(
            tuple(plan_free["tpl"]),
            scap=rt_f["SCAP"],
            cmp_=rt_f["CMP"],
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
        )
        lc = (
            "clus",
            fn,
            (n_kernel, npad, k, rt_f["SCAP"], rt_f["CMP"], 0, 0, 0, 0, 0),
        )
        _VARLEN_CACHE[key] = lc
        return lc
    plan = route_streaming(num_rows, n_route, npad, k, force_main=True)
    tpl = tuple(plan["tpl"])  # (BLK, U, MINB, SNB, KPT, SPLIT, TSHG)
    rt = plan["rt"]
    r_const = rt["R"]
    # TSHG (tpl[6]) is dead under varlen (the ctor compiles the TSH
    # machinery in whenever SPLIT); normalize it out of the compile key so
    # row counts differing only in that slot share one engine
    fn = dev.get_compiled(tpl[:6] + (False,) + (next_n, cr_shift, r_const), hint_free=True)
    big = num_rows * r_const <= 148
    aim_base = (
        ((4 * k if k >= 1024 else 2 * k) if r_const == 1 else 2 * k)
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    sfac = (
        (32 if r_const == 2 else (48 if k > 1024 else 16))
        if r_const > 1
        else (64 if k >= 1024 else 32)
    )
    amin = 3 * k if r_const == 2 else (7 * k) // 2
    sd_en = 1 if (k > 1024 and not big) else 0
    # TSH-floor staging: gate on SPLIT and K only. Gating additionally on
    # num_rows > 15 would strand small batches in SPLIT-main without the
    # staged floor (a distribution-dependent tail regression); the kernel
    # gates TSH per row at runtime anyway.
    tsh_en = 1 if (tpl[5] and k <= 1024) else 0
    pre = (0, npad, k, rt["SCAP_"], rt["CMP_"], r_const, 0, 0, 0, 0, 0)
    tail = (aim_base, sfac, amin, sd_en, tsh_en)
    lc = ("main", fn, pre, tail)
    _VARLEN_CACHE[key] = lc
    return lc


# ===========================================================================
# ==== native bf16 arm (separate route table + launcher + entry) ============
# ===========================================================================
# The fp32 dispatch above is untouched. The bf16 arm mirrors it through its
# OWN route table (route_bf16 / route_streaming_bf16 — initially the fp32
# table, re-tunable for bf16 tie hazards without touching the fp32 tables)
# and its own launcher cache, compiling the device module's BFloat16
# constexpr variants (native 16-bit loads widened in registers; 4 elements
# per 8-byte vector, so every element-indexed capacity in the fp32 table
# stays valid).


def _bf16_halve_u(plan: dict) -> dict:
    """bf16 streaming arms read 8-element 16B vectors, so one vector covers
    what two fp32 float4s did: halve the per-thread vector batch U (tuple slot
    1) to keep per-tile ELEMENT coverage — and the 32-bit classify mask —
    identical while halving the load-instruction count."""
    tpl = list(plan["tpl"])
    tpl[1] = max(int(tpl[1]) // 2, 1)
    plan["tpl"] = tuple(tpl)
    return plan


def route_bf16(b: int, n: int, npad: int, k: int) -> dict[str, object]:
    """bf16 dispatch table. route() is a pure function of shape, so the fp32
    table is the correct starting point; bf16-specific re-tunes (16B-vector U
    halving, register-family capacity fitting, bin-count halving) are applied
    to the returned copy only."""
    plan = route(b, n, npad, k)
    if (
        plan["kernel"] in ("reg", "regimg")
        and k == 512
        and n <= 1280
        and b > 148
    ):
        # The 1088-column bf16 envelope fits 1024 vectorized elements plus a
        # 256-element scalar tail. One 256-thread, VPT=1 CTA therefore covers
        # the row without classifying synthetic -inf lanes.
        plan["tpl"] = (256, 1, 8, 1, True, True, False, 512)
        plan["block"] = 256
    # Packed bf16 crossing words are legal when every candidate index fits in
    # 16 bits.  Restrict them to wave-rich register grids where halving the
    # shared crossing window can improve residency and hide the extra packing
    # ALU; small-batch grids retain the original pair representation.
    pk16 = npad <= 65536 and n <= 65536 and b > 15
    if plan["kernel"] == "main" and not plan["tpl"][5] and b > 148:
        # bf16-only register-band extension. The fp32 b>148 register rung tops
        # out at n4 <= 1024 quads; rows a hair above it (the 4K+pad envelopes:
        # n4 in (1024, 1152]) fall to the BLK=256 nosplit streaming arm, whose
        # sample/collect/refine phases dominate at these sizes (NCU on that
        # arm: 39% occupancy, DRAM <10%, no dominant phase). The register
        # engine's PROVEN wide-band (1024, VPT=2) bf16 tuples hold 2048
        # bf16x4 vector slots -- enough for these rows -- and the same compiled
        # variants already serve the b <= 148 shapes of the same envelopes
        # (v32_4k / flash_16k), so this only widens their grid. fp32 routing
        # is untouched.
        n4q = n >> 2
        if 1024 < n4q <= 1152:
            CMP = n if n < 2560 else 2560
            QC = 1024  # b > 148 rank gate, as in route()
            CURE = not (n < 2 * k and b > 148)
            DEGE = (n <= 3 * k) or (n <= 4 * k + 64)
            if DEGE and CMP < n:
                CMP = n
            # BLK=512/MINB=4 keeps 4 CTAs per SM (592 co-resident slots, the
            # same occupancy shape as the proven fp32 (512,2,4) rung and the
            # streaming arm it replaces); a 1024-thread MINB=1 tuple measured
            # 40.1us on v32_4k_bs1024 (6.9 serial waves) vs 31.5us baseline.
            # VPT=4 bf16x4 vectors carry the same 8 data registers as fp32
            # VPT=2 float4s, so the 32-register launch bound still fits.
            # 512-bin histograms per the BLK512 register tier.
            # Tail-aware VPT fit (see the reg branch below): the bf16 reg
            # kernel's capped vector window plus <= BLK scalar-tail overflow
            # lets VPT=2 (1024 slots + tail) cover these 4K+pad envelopes with
            # half the value registers and half the classify work of VPT=4.
            vptx = 4
            while vptx > 1 and (
                512 * (vptx // 2) >= n4q or 512 * (vptx // 2) * 4 + 512 >= n
            ):
                vptx //= 2
            if DEGE:
                tpl = (512, vptx, 4, 1, CURE, True, False, 512)
            else:
                kpt = 1 if k <= 512 else 2
                tpl = (512, vptx, 4, kpt, CURE, False, False, 512)
            # ONE-WAVE reshape (512, V, 4) -> (256, 2V, 8): same 512*V slots
            # (so the scalar-tail overflow is unchanged), same 32-reg budget,
            # but 1184 co-resident row-CTAs vs 592 — a B=1024 grid drains in
            # one wave, not 1.73. Gated k <= 1024: at K=512/1024 the halved
            # wave count dominates (flash_16k_bs1024 15.84->14.88us), but at
            # K=2048 (v32_4k_bs1024) doubling VPT back to 4 restores classify
            # work that the tail-aware VPT halving removed, and the per-CTA
            # K=2048 collect/emit dominates the wave count — measured 19.46->
            # 21.73us, so K=2048 keeps the leader's BLK=512 halved-VPT rung.
            if 2 * vptx <= 4 and k <= 1024:
                tpl = (256, 2 * vptx, 8) + tuple(tpl[3:])
            return {
                "kernel": "reg",
                "tpl": tpl,
                "rt": {
                    "n": n,
                    "npad": npad,
                    "k": k,
                    "CMP": CMP,
                    "IMGOFF": 2 * NB,
                    "QC": QC,
                },
                "grid": (b, 1),
                "cluster": 1,
                "block": tpl[0],
                "smem": (tpl[7] + (CMP if pk16 else 2 * CMP)) * 4,
                "pk16": pk16,
                "ws": False,
            }
    if (
        plan["kernel"] == "main"
        and b <= 148
        and 1024 * 4 * 4 + 1024 >= n
        and (n >> 2) > 1152
        # A two-rank packed register cluster covers the K-heavy low-B tail
        # with more row parallelism; let the existing refit below select it.
        and not (b <= 2 and k > BLKC and 2 * BLKC * 8 + BLKC >= n)
    ):
        # Second bf16 register band, unlocked by the tail-clamped window: a
        # (1024, VPT=4) batch plus the <= BLK scalar tail holds envelopes up
        # to 17408 elements — rows whose fp32 quad count (4112..4352) misses
        # the 4096-slot capacity by the 64-pad tail and therefore fell to
        # streaming. One CTA per row: no split sample/merge phases, no
        # cluster sync, no DSMEM merge. Measured on the 16K envelopes:
        # v32_16k_bs2 11.62->8.78 us, v32_16k_bs128 13.76->10.54 us. Kept off
        # reg_clus rows (flash_64k: the (1024,1,2) cluster measured faster at
        # k=512) and off b > 148 (multi-wave MINB=1 grids measured +34..55%
        # slower than the streaming arms on bs256/512).
        n4q = n >> 2
        CMP = n if n < 2560 else 2560
        QC = 1024
        CURE = not (n < 2 * k and b > 148)
        DEGE = (n <= 3 * k) or (n <= 4 * k + 64)
        if DEGE and CMP < n:
            CMP = n
        kpt = 1 if k <= 1024 else 2
        vptx = 4
        while vptx > 1 and (
            1024 * (vptx // 2) >= n4q or 1024 * (vptx // 2) * 4 + 1024 >= n
        ):
            vptx //= 2
        tpl = (1024, vptx, 1, kpt, CURE, DEGE, False, 1024)
        return {
            "kernel": "reg",
            "tpl": tpl,
            "rt": {
                "n": n,
                "npad": npad,
                "k": k,
                "CMP": CMP,
                "IMGOFF": 2 * NB,
                "QC": QC,
            },
            "grid": (b, 1),
            "cluster": 1,
            "block": tpl[0],
            "smem": (tpl[7] + (CMP if pk16 else 2 * CMP)) * 4,
            "pk16": pk16,
            "ws": False,
        }
    if (
        plan["kernel"] == "main"
        and plan["tpl"][5]
        and b <= 32
        and k <= 2 * BLKC
    ):
        # bf16 reg_clus refit of the low-B deep-split band. fp32 register
        # capacity (cs * BLKC * v <= 32768 quads = 131072 elems) never fit
        # these rows, so route() fell to the R-part split main engine (global
        # slab, gpu-scope fences, last-CTA merge scan) — measured ~parity
        # with fp32 (pro_512k_bs1 12.03us vs 11.79 fp32) because the phases,
        # not the bytes, dominate at b*R << 148 CTAs. bf16 doubles elements
        # per register: rows up to 262144 elems fit an (8-CTA, v<=4) cluster
        # whose DSMEM merge already measured 8.5us at n=65600
        # (flash_256k_bs1 reg_clus (1024,1,8)) vs ~12us for the equivalent
        # splits. Same (v, cs) derivation as the reg_clus refit below (same
        # amax ladder, same cs=8 b>15 GPC veto); no-fit shapes fall through
        # to the split arms unchanged. v <= 2 only: (4, 8) tuples measured
        # 19-29us vs the 11-12us splits on the 262144-elem envelopes — the
        # per-CTA ladder classify at 32 elems/thread over 8-16 SMs loses to
        # the split arm's 33-128-CTA parallelism (packed fragments did not
        # rescue it), while v <= 2 keeps 16 elems/thread and wins ~15-17%.
        # K=2048 is admitted through B<=32 after changing the bf16 P0 bracket
        # from a first-K prefix to a whole-row strided sample. That prevents
        # the upper tail from collapsing into one huge crossing bin while the
        # count/escape/emit phases remain unchanged and exact.
        # cs=16 widening: the 163776/262144-elem envelopes only fit v=2 at a
        # 16-CTA cluster (fp32 never emits cs > 8; the DSMEM merge phases are
        # CS-parametric). A 16-CTA cluster fills most of one GPC (B200: 8 GPCs
        # x 18-19 SMs), so at most 8 such clusters are co-resident -> b <= 8
        # veto keeps every cluster in the first scheduling wave; larger b
        # keeps the split-main arm, whose 33-128-CTA slab parallelism wins
        # once the row count covers the machine anyway.
        av = 148 // (b if b > 0 else 1)
        amax = 1
        while (amax << 1) <= av and amax < 16:
            amax <<= 1
        vsel = 0
        cs = 0
        if amax >= 2:
            for v in (1, 2):
                c = 1
                while c * BLKC * v * 8 + BLKC < n:
                    c <<= 1
                if c == 8 and b > 15:  # same GPC-packing veto as route()
                    continue
                # cs=16 admission: one resident wave (a 16-CTA cluster fills a
                # GPC, so <= 8 co-resident), the proven 16-elems/thread shape
                # only, and K >= 1024 only — the K=512 1024-bin variant
                # measured 13.2us vs the 10.0us split on the 262144-elem
                # envelope (flash_1024k_bs1): the doubled merge fan-in and
                # cluster rendezvous outweigh its light emit.
                # Re-admit B=1 K>=1024 at CS16/VPT2: these envelopes cannot
                # fit CS8, and DSMEM avoids the split-main global slab plus
                # LAST-CTA serial merge. Keep K512 and VPT1 excluded.
                if c == 16 and (b > 8 or v < 2 or k < 1024):
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:
            rank_span = BLKC * vsel * 8
            needed_cs = max((n - BLKC + rank_span - 1) // rank_span, 1)
            if (
                b == 1
                and k > BLKC
                and vsel == 2
                and cs == 16
                and needed_cs == 10
            ):
                # Narrow the cluster only when the route's capacity formula
                # proves that exactly ten VPT2 ranks are required.
                cs = 10
            return {
                "kernel": "reg_clus",
                "tpl": (BLKC, vsel, cs),
                "rt": {"n": n, "npad": npad, "k": k},  # dims only
                "grid": (cs, b),
                "cluster": cs,
                "block": BLKC,
                "smem": (3 * NB + 2 * CMPC) * 4,
                "ws": False,
            }
    if plan["kernel"] in ("main", "clus"):
        plan = _bf16_halve_u(plan)
    elif plan["kernel"] == "reg_clus":
        # Re-derive (v, cs) with the SAME dispatch formula route() uses, but in
        # bf16 8-element-vector units (n8): naive VPT halving strands ranks
        # idle when VPT == 1 (the per-rank element span doubles), while the
        # re-derivation keeps ranks balanced and can shrink the cluster.
        # Tail-aware coverage: the bf16 reg_clus kernel caps its vector window
        # at cs*span and hands the (<= BLKC, enforced here) element overflow
        # to the rank-0 scalar tail, so a 64-pad envelope just over a pow2
        # capacity no longer doubles v (or the cluster width).
        n8 = n >> 3
        av = 148 // (b if b > 0 else 1)
        amax = 1
        while (amax << 1) <= av and amax < 8:
            amax <<= 1
        vsel = 0
        cs = 0
        if amax >= 2:
            for v in (1, 2, 4):
                c = 1
                while c * BLKC * v * 8 + BLKC < n:
                    c <<= 1
                if c == 8 and b > 15:  # same GPC-packing veto as route()
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:
            plan["tpl"] = (BLKC, vsel, cs)
            plan["grid"] = (cs, b)
            plan["cluster"] = cs
        else:
            plan = _bf16_halve_u(plan)
    elif plan["kernel"] in ("reg", "regimg"):
        # (BLK, VPT, MINB, KPT, CUR, DEG, IMG, NBH)
        tpl = list(plan["tpl"])
        if b == 1 and tpl[2] > 1:
            # BS=1 register-budget lift (allowed BS=1-specific configuration
            # selection): MINB is a min_blocks_per_mp co-residency floor whose
            # only effect at one CTA per launch is capping the kernel to
            # 65536/(BLK*MINB) registers per thread (32 at (512,.,4) and
            # (1024,.,2)). A single-row launch has nothing to co-schedule, so
            # MINB=1 restores the full 64-register budget on the one live CTA.
            tpl[2] = 1
        n4r = n >> 2
        # capacity-fit VPT: the fp32 table over-provisions the register batch
        # for the envelope; dead -inf lanes still classify into the histogram,
        # so halve VPT while the halved batch still covers the envelope.
        # Tail-aware fit: the bf16 reg kernel caps its vector window at
        # BLK*VPT and hands the (<= BLK, enforced here) element overflow to
        # the value-generic scalar tail, so the 64-pad envelopes just over a
        # pow2 slot count no longer force a 2x over-provisioned batch.
        while tpl[1] > 1 and (
            tpl[0] * (tpl[1] // 2) >= n4r
            or tpl[0] * (tpl[1] // 2) * 4 + tpl[0] >= n
        ):
            tpl[1] //= 2
        if (
            b == 1
            and tpl[0] == 1024
            and not tpl[6]
            and k <= 1024
            and 512 * 4 + 512 >= n
        ):
            # A single BLK512/VPT1 CTA covers these BS=1 envelopes through the
            # existing tail-aware window while reducing barrier and scan work.
            # The wider VPT2 envelope is excluded after repeated regression.
            tpl[0] = 512
            tpl[1] = 1
            plan["block"] = 512
        elif (
            b == 1
            and tpl[0] == 1024
            and tpl[1] == 1
            and not tpl[6]
            and k > 1024
            and 512 * 2 * 4 + 512 >= n
        ):
            # K-heavy 4K envelopes retain the same vector capacity and value
            # payload as BLK1024/VPT1, but BLK512/VPT2 halves the number of
            # warps participating in each CTA-wide barrier.
            tpl[0] = 512
            tpl[1] = 2
            plan["block"] = 512
        # Match the histogram width to the register engine: BLK512 needs
        # only 512 bf16 bins, while BLK256/1024 retain 1024. This preserves
        # the ladder and escape descent while reducing clear/scan work and
        # dynamic shared memory on the BLK512 rungs.
        target_nbh = 512 if tpl[0] == 512 else 1024
        if tpl[7] > target_nbh:
            tpl[7] = target_nbh
        if b > 148 and k <= 1024 and tpl[0] == 512 and tpl[2] == 4 and tpl[1] <= 2:
            # ONE-WAVE reshape of the b > 148 register rung: (512, V, 4) ->
            # (256, 2V, 8). SAME 512*V vector slots (so the scalar-tail
            # overflow, if any, is unchanged) and the same 32-reg launch
            # budget (2V packed u32 pairs carry what V float4-equivalents
            # did), but 1184 co-resident row-CTAs instead of 592: a B=1024
            # grid drains in ONE wave instead of 1.73, and intra-row barriers
            # sync 8 warps instead of 16. This composes with the tail-aware
            # VPT halving above (which minimises registers/classify work at
            # BLK=512, 1.73 waves); here we trade the halved VPT back for
            # BLK=256's doubled co-residency, which the B=1024 grids need
            # more. VPT stays within the reg kernel's (1, 2, 4) bound.
            tpl[0] = 256
            tpl[1] *= 2
            tpl[2] = 8
            if tpl[1] <= 2 and k > 512:
                # Low-VPT one-wave rungs use 256 bins for K>=1024, where
                # halving the clear/scan work wins.  Keep 1024 bins for
                # K=512: its shorter ladder makes the coarser crossing bin's
                # reclassification cost larger than the scan savings.
                tpl[7] = 256
            plan["block"] = 256
        plan["tpl"] = tuple(tpl)
        # launcher smem arg must track the (possibly smaller) bin count; the
        # packed representation uses one key16|idx16 word per crossing
        # candidate; regimg keeps the original ck/ci pair layout.
        if pk16 and plan["kernel"] == "reg":
            plan["smem"] = (tpl[7] + plan["rt"]["CMP"]) * 4
            plan["pk16"] = True
        else:
            plan["smem"] = (tpl[7] + 2 * plan["rt"]["CMP"]) * 4
        if b == 1 and k > 1024:
            # The BS=1 K2048 register buckets have crossing populations at
            # most 141 on the real bf16 captures, and every crossing set is
            # a single value class.  Admit them to the existing exact
            # one-class emitter instead of sending m in (96, 192] through
            # the barrier-separated key-space narrowing fallback.
            plan["rt"]["QC"] = 192
    return plan


def _degen_gate(blk: int, kpt: int, split: bool) -> tuple[int, int]:
    """(CMPB, SCPB) of a main-family tuple — the SAME constexpr formulas the
    kernel derives (gvr_ss GvrMainKernel ctor): the capacity pair that
    decides whether a row falls into the degen narrowing path. Used by bf16
    route rungs to guarantee no rung shrinks the degen gate."""
    kbig = (kpt >= 2) and (kpt * blk >= 2048)
    scpb = (8192 if split else 16384) if blk >= 1024 else (8192 if kbig else 4096)
    cmpb = (4096 if kbig else 2048) if blk >= 1024 else 1024
    return cmpb, scpb


def route_streaming_bf16(
    b: int, n: int, npad: int, k: int, force_main: bool = False
) -> dict[str, object]:
    """bf16 twin of route_streaming (see route_bf16).

    Split-aware vector-width pick: the fp32 table derives the split count R so
    each CTA's chunk (~n/R) fills one fp32 tile of BLK*U*4 elements. When
    U == 1 the 16-byte bf16 tile cannot shrink with U and spans BLK*8 -- a
    ~BLK*4 chunk then idles half the threads and halves the outstanding-load
    parallelism exactly in the latency-bound deep-split regime. Route those
    plans to the 8-byte-vector engine (fp32 tile geometry, all threads
    active); keep the 16-byte engine when the chunk actually fills >= 3/4 of
    the wider tile (fewer load instructions at full thread activity)."""
    plan = route_streaming(b, n, npad, k, force_main=force_main)
    if plan["kernel"] in ("main", "clus"):
        tpl = plan["tpl"]
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and int(tpl[0]) == 1024
            and int(tpl[1]) == 1
            and n >= 65536
            and k <= 2048
        ):
            # Spread underfilled low-batch split rows across twice as many SMs.
            # Rows already admitted by route_bf16's register-cluster band never
            # reach this fallback, so this recovers the remaining partial-wave
            # cells without disturbing their faster DSMEM path.
            #
            # DEGEN-GATE GUARD: a rung that lowers BLK must not shrink the
            # kernel's degen gate (CMPB/SCPB are constexprs of (BLK, KPT,
            # SPLIT); at BLK=512 CMPB halves 2048->1024 and SCPB can halve
            # 8192->4096). On bf16 tie-heavy rows the crossing-bin population
            # is 4-30x the fp32 one, and the shrunken gate flips those rows
            # into the whole-row degen path (measured 288-301us vs 20us at
            # BLK=1024 on pro_1024k BS=1). For K>=1024 no BLK-lowering rung
            # may pass unless _degen_gate(new) covers _degen_gate(old); K=512
            # retains the measured r512 latency win and has no catastrophic row.
            r512 = ((n >> 2) + 511) // 512
            if r512 > int(plan["rt"]["R"]) and b * r512 <= 148:
                kpt = 1 if k <= 512 else (2 if k <= 1024 else 4)
                cmpb_old, scpb_old = _degen_gate(int(tpl[0]), int(tpl[4]), True)
                cmpb_new, scpb_new = _degen_gate(512, kpt, True)
                if k < 1024 or (cmpb_new >= cmpb_old and scpb_new >= scpb_old):
                    tpl = (512, 1, 1, tpl[3], kpt, True, tpl[6])
                    plan["tpl"] = tpl
                    plan["grid"] = (r512, b)
                    plan["block"] = 512
                    plan["rt"]["R"] = r512
                    plan["rt"]["Q"] = ((n >> 2) + r512 - 1) // r512
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and b <= 32
            and k <= 1024
            and int(tpl[1]) == 1
            and "rt" in plan
            and int(plan["rt"].get("R", 1)) > 1
            and b * int(plan["rt"].get("R", 1)) >= 148
            and (n + int(plan["rt"]["R"]) - 1) // int(plan["rt"]["R"]) < 6144
        ):
            # Full-wave deep splits (b*R >= 148) are kept OFF the vec4 engine
            # below (measured: the 16-byte engine wins once a full wave hides
            # load latency) -- but with the fp32-derived R their sub-6144
            # chunks fill under 3/4 of the 16-byte tile. Re-derive the part
            # count in native bf16 8-element-vector units,
            # R = min(148//b, ceil(n8/1024)), so each part fills one full
            # 16-byte tile (1024 thr x 8 elems): full thread activity, full
            # load width, and roughly half the parts feeding the merge scan.
            # Where vec4 IS available (sub-wave grids) this same re-derivation
            # measured neutral-to-slower (12.99 vs 12.48 us on v32_128k_bs1),
            # so it stays scoped to the vec4-blocked band.
            r1_8 = max(148 // b, 1)
            r2_8 = max(((n >> 3) + 1023) // 1024, 1)
            r_n8 = max(min(r1_8, r2_8), 1)
            if r_n8 > 1 and r_n8 != int(plan["rt"]["R"]):
                q8 = ((n >> 3) + r_n8 - 1) // r_n8
                per8 = q8 >> 10
                u8 = 8 if per8 >= 8 else (4 if per8 >= 4 else (2 if per8 >= 2 else 1))
                if u8 > 4:
                    u8 = 4  # 32-bit classify mask: at most 4 bf16x8 vecs/thread
                kpt8 = 1 if k <= 1024 else (2 if k <= 2048 else (4 if k <= 4096 else 8))
                tshg8 = b > 15 and k <= 1024 and (n >> 2) <= 32768
                plan["tpl"] = (1024, u8, 1, tpl[3], kpt8, True, tshg8)
                plan["rt"]["R"] = r_n8
                plan["rt"]["Q"] = ((n >> 2) + r_n8 - 1) // r_n8
                plan["grid"] = (r_n8, b)
                return plan
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and int(tpl[1]) == 1
            and n >= 16384
            and "rt" in plan
            and int(plan["rt"].get("R", 1)) > 1
            and b * int(plan["rt"].get("R", 1)) < 148
        ):
            # n gate: short split rows (n ~16K, R <= 5) are dominated by the
            # sample/threshold phase, not the chunk load; measured v8-better
            r_const = int(plan["rt"]["R"])
            chunk = (n + r_const - 1) // r_const
            if chunk < (int(tpl[0]) * 8 * 3) // 4:
                plan["vec4"] = True  # 8-byte engine, tuple kept at fp32 geometry
                return plan
        if plan["kernel"] == "main" and not tpl[5] and int(tpl[1]) == 4:
            # No-split roll-count guard: halving U mirrors the fp32 tile span,
            # but when n only just exceeds that span the halved tile leaves a
            # runt roll (e.g. 64 of 16448 elements) — a full extra
            # classify/merge pass for almost no data. Keeping the fp32 U at
            # 16-byte vectors doubles the span, covers the row in strictly
            # fewer rolls, and reuses the register shape the (BLK, 4) 16-byte
            # variants already run elsewhere. Applied only at fp32 U == 4:
            # U == 8 un-halved would put 32 packed value registers in flight
            # (the round-1 spill wall).
            blk = int(tpl[0])
            rolls_full = -(-n // (blk * 4 * 8))
            rolls_half = -(-n // (blk * 2 * 8))
            if rolls_full < rolls_half:
                return plan  # keep fp32 U: fewer 16-byte tile rolls
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and int(tpl[1]) == 4
            and "rt" in plan
        ):
            # Split analog of the no-split roll-count guard above: halving U
            # mirrors fp32 tile spans, but a big-chunk split (chunk >= ~33K)
            # then pays 3-4 classify/merge rolls where the un-halved U=4
            # 16-byte tile covers the chunk in strictly fewer. Same register
            # shape as the (BLK, 4) 16-byte variants running elsewhere; the
            # 32-bit classify mask caps at 4 bf16x8 vecs/thread so U=4 is the
            # documented ceiling.
            r_c = int(plan["rt"].get("R", 1))
            chunk = (n + r_c - 1) // r_c
            rolls_full_s = -(-chunk // (int(tpl[0]) * 4 * 8))
            rolls_half_s = -(-chunk // (int(tpl[0]) * 2 * 8))
            if rolls_full_s < rolls_half_s:
                return plan  # keep fp32 U=4: fewer 16-byte rolls per chunk
        plan = _bf16_halve_u(plan)
        tpl = plan["tpl"]
        if (
            plan["kernel"] == "main"
            and tpl[5]
            and int(tpl[0]) == 1024
            and int(tpl[1]) < 4
        ):
            # Avoid a bf16 tile-roll cliff by rounding a nearly full-wave
            # split down to the preceding power of two only when the smaller
            # grid still has at least 128 CTAs and doubling U removes a roll.
            r_const = int(plan["rt"]["R"])
            r_pow2 = 1 << (r_const.bit_length() - 1)
            u_const = int(tpl[1])
            u_pow2 = min(2 * u_const, 4)
            chunk_now = (n + r_const - 1) // r_const
            chunk_pow2 = (n + r_pow2 - 1) // r_pow2
            rolls_now = (chunk_now + 1024 * u_const * 8 - 1) // (
                1024 * u_const * 8
            )
            rolls_pow2 = (chunk_pow2 + 1024 * u_pow2 * 8 - 1) // (
                1024 * u_pow2 * 8
            )
            if (
                r_pow2 < r_const
                and b * r_pow2 >= 128
                and rolls_pow2 < rolls_now
            ):
                plan["tpl"] = (tpl[0], u_pow2) + tuple(tpl[2:])
                plan["rt"]["R"] = r_pow2
                plan["rt"]["Q"] = ((n >> 2) + r_pow2 - 1) // r_pow2
                plan["grid"] = (r_pow2, b)
    return plan


_VARLEN_CACHE_BF16 = {}


def _varlen_launcher_bf16(
    num_rows: int,
    npad: int,
    k: int,
    n_env: int,
    next_n: int,
    cr: int,
) -> tuple:
    """bf16 twin of _varlen_launcher: same capture-stable family tiers, the
    bf16 route table, and the BFloat16 compiled variants."""
    key = (num_rows, npad, k, n_env, next_n, cr)
    hit = _VARLEN_CACHE_BF16.get(key)
    if hit is not None:
        return hit
    n_kernel = min(n_env, npad)
    n_route = max(n_kernel, k + 1)
    cr_shift = 0 if cr == 1 else 2
    dev = _device()
    plan_free = route_bf16(num_rows, n_route, npad, k)
    if plan_free["kernel"] == "reg_clus":
        fn = dev.get_compiled__regclus(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            dtype="bf16",
            nbh=512
            if (k > 1024 or (k in (512, 1024) and n_env >= 65536))
            else 1024,
            quadc=(
                # QC192 covers every observed CS2/CS4 VPT1 crossing
                # population (max 172), halving the one-class range scan
                # versus QC384. It also retains the existing CS10 choice;
                # CS8/VPT2 keeps its measured QC96 cutoff.
                192
                if (
                    k > 1024
                    and (
                        plan_free["tpl"][2] >= 10
                        or (
                            num_rows == 1
                            and plan_free["tpl"][1] == 1
                            and plan_free["tpl"][2] < 8
                        )
                    )
                )
                else 96
                if (
                    (k > 1024 or (k == 1024 and num_rows == 1))
                    and plan_free["tpl"][2] >= 8
                    and (num_rows > 1 or plan_free["tpl"][1] == 2)
                )
                else 384
            ),
            # CS16/VPT2 uses the same rank-0-local QC window as CS8: every
            # admitted candidate fits inside rank 0's CMPC=4096 slab.
            # Enable the one-class bypass for the v32_256k BS=1 route too.
            oneq_enabled=(k > 1024),
        )
        lc = ("reg_clus", fn, n_kernel)
        _VARLEN_CACHE_BF16[key] = lc
        return lc
    if plan_free["kernel"] in ("reg", "regimg"):
        fn = dev.get_compiled__reg(
            tuple(plan_free["tpl"]),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            dtype="bf16",
            pk16=bool(plan_free.get("pk16", False)),
        )
        rt_f = plan_free["rt"]
        lc = (
            "reg",
            fn,
            (n_kernel, rt_f["CMP"], rt_f["QC"], dev.STATIC_BYTES + plan_free["smem"]),
        )
        _VARLEN_CACHE_BF16[key] = lc
        return lc
    if plan_free["kernel"] == "clus":
        rt_f = plan_free["rt"]
        fn = dev.get_compiled__clus(
            tuple(plan_free["tpl"]),
            scap=rt_f["SCAP"],
            cmp_=rt_f["CMP"],
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            hint_free=True,
            dtype="bf16",
        )
        lc = (
            "clus",
            fn,
            (n_kernel, npad, k, rt_f["SCAP"], rt_f["CMP"], 0, 0, 0, 0, 0),
        )
        _VARLEN_CACHE_BF16[key] = lc
        return lc
    plan = route_streaming_bf16(num_rows, n_route, npad, k, force_main=True)
    tpl = tuple(plan["tpl"])  # (BLK, U, MINB, SNB, KPT, SPLIT, TSHG)
    rt = plan["rt"]
    r_const = rt["R"]
    if plan.get("vec4"):
        # deep-split chunk fits the fp32 tile: 8-byte bf16x4 engine, fp32
        # geometry (tuple NOT U-halved), identical launch ABI and workspace.
        # This route is restricted to sub-wave grids above; once 148 CTAs are
        # available, the 16-byte engine has enough latency-hiding parallelism.
        fn = _device_v4().get_compiled(
            tpl[:6] + (False,) + (next_n, cr_shift, r_const), hint_free=True, dtype="bf16"
        )
    else:
        # On nosplit BLK<512 envelopes with 16-bit indices, stage each
        # survivor as value_hi16|idx16 so later walks avoid scattered rereads.
        v16 = (not tpl[5]) and int(tpl[0]) < 512 and npad <= 65536
        fn = dev.get_compiled(
            tpl[:6] + (False,) + (next_n, cr_shift, r_const),
            hint_free=True,
            dtype="bf16",
            v16=v16,
        )
    big = num_rows * r_const <= 148
    # bf16 split-arm retune (runtime launch constants only; the fp32 launcher
    # above is untouched). K=2048 deep-split rows (r_const > 2) ran an
    # effective 2k survivor target (the amin=3.5k floor is clipped by the
    # SCPB/2=4096 cap); every staged survivor costs a global-slab publish
    # word plus a load + red.shared + cursor-emit step in the LAST-arriving
    # CTA's serial consume tail, so the tail — not the bytes — is why these
    # cells measured ~1.00x vs fp32. 1.75k trims a quarter of that tail with
    # the sample unchanged (sfac=48: ~3 sigma of the order-statistic
    # estimate, same margin class as the fp32 floors; bf16 ties at TF only
    # add survivors; a miss still lands in the exact degen fallback).  The
    # wave-rich B>8 route can trim this again to 1.625K: its many concurrent
    # rows expose the per-row survivor tail, while the B=1 route measured flat
    # and retains the safer 1.75K target.
    # Measured: v32_256k bs1/bs4/bs32 -5.9%/-5.4%/-6.5% (1.75K, r8).
    # K<=1024 deep splits KEEP the fp32 floors: their TSH double-staging
    # safety net is runtime-gated off above 131072 columns (n4 > 32768) and
    # their sfac=16 sample cannot carry a lower floor — probed 2k floors
    # degenerated w_pro_1024k_bs16 to 225us (single-CTA whole-row narrowing).
    if r_const > 2 and k > 1024:
        split_aim = (13 * k) // 8 if num_rows > 8 else (7 * k) // 4
        split_sfac = 48
    else:
        split_aim = 2 * k if k > 1024 or r_const == 2 else (7 * k) // 2
        split_sfac = 32 if r_const == 2 else (48 if k > 1024 else 16)
    # Long-envelope K=2048 nosplit rows use a 1.5K survivor target: their
    # larger sample keeps the first-attempt margin while trimming another
    # quarter of staged traffic. K=1024 one-wave nosplit rows take 1.5K at any
    # envelope (a005-r7 win evidence: retained arms improved with no retry
    # regression). Shorter K=2048 rows retain 2K after the 1.5K probe exposed
    # retries around the 64K envelope.
    aim_base = (
        (
            ((3 * k) // 2 if (k >= 1024 and (k == 1024 or n_env >= 131072)) else 2 * k)
            if r_const == 1
            else split_aim
        )
        if big
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)
    )
    sfac = split_sfac if r_const > 1 else (64 if k >= 1024 else 32)
    amin = 3 * k if r_const == 2 else split_aim
    sd_en = 1 if (k > 1024 and not big) else 0
    tsh_en = 1 if (tpl[5] and k <= 1024) else 0
    pre = (0, npad, k, rt["SCAP_"], rt["CMP_"], r_const, 0, 0, 0, 0, 0)
    tail = (aim_base, sfac, amin, sd_en, tsh_en)
    lc = ("main", fn, pre, tail)
    _VARLEN_CACHE_BF16[key] = lc
    return lc


def run_varlen_bf16(
    logits: torch.Tensor,
    kv_lens: torch.Tensor,
    indices: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
    max_seq_len: int | None = None,
    workspace: torch.Tensor | None = None,
) -> None:
    """Native-bf16 twin of ``run_varlen`` (same per-row varlen contract;
    bfloat16 logits read directly by the device code — 8-byte vectors widened
    in registers; no values epilogue). Validation battery and launch shape
    mirror the fp32 entry statement-for-statement."""
    if logits.dtype is not torch.bfloat16:
        raise RuntimeError(f"logits must be bfloat16 (got {logits.dtype}); fp32 -> run_varlen")
    if not (isinstance(kv_lens, _TENSOR) and kv_lens.is_cuda):
        raise RuntimeError("kv_lens must be a CUDA tensor")
    if kv_lens.dtype is not _I32:
        raise RuntimeError("kv_lens must be int32")
    if kv_lens.dim() != 1:
        raise RuntimeError("kv_lens must be 1-D")
    nn = _index(next_n)
    cr = _index(compress_ratio)
    if nn < 1:
        raise RuntimeError(f"next_n must be >= 1, got {nn}")
    if cr not in (1, 4):
        raise RuntimeError(f"compress_ratio must be 1 (DSv3.2) or 4 (DSv4), got {cr}")
    if len(logits.shape) != 2:
        raise RuntimeError("logits must be 2-D")
    num_rows = logits.shape[0]
    if num_rows == 0:
        return
    if num_rows % nn:
        raise RuntimeError(f"num_rows {num_rows} not divisible by next_n {nn}")
    batch = num_rows // nn
    if kv_lens.shape[0] != batch:
        raise RuntimeError(f"kv_lens length {kv_lens.shape[0]} != num_rows/next_n = {batch}")
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    if workspace is not None:
        validate_run_ws(workspace, logits)
        ws = kernel_view(workspace)
    else:
        ws = _ws_hot.get(d)
        if ws is None:
            ws = default_workspace(logits)
    if not (logits.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if indices.dtype is not _I32:
        raise RuntimeError("indices must be int32")
    if len(indices.shape) != 2 or indices.shape[0] != num_rows:
        raise RuntimeError(
            f"indices must be [num_rows={num_rows}, >=k], got {tuple(indices.shape)}"
        )
    k = indices.shape[1]
    if not (indices.is_contiguous() and kv_lens.is_contiguous()):
        raise RuntimeError("indices/kv_lens must be contiguous")
    if logits.stride(1) != 1:
        raise RuntimeError("logits inner stride must be 1")
    npad = logits.stride(0) if num_rows > 1 else logits.shape[1]
    lg = logits
    if not logits.is_contiguous():
        need = logits.storage_offset() + num_rows * npad
        if logits.untyped_storage().size() // 2 < need:
            raise RuntimeError("logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    if npad & 7:
        # bf16 streaming arms read 16B vectors (8 elems): row stride % 8 == 0
        raise RuntimeError(f"npad (logits row stride) must be a multiple of 8, got {npad}")
    if lg.data_ptr() & 15:
        raise RuntimeError("logits base must be 16-byte aligned")
    cshift = 0 if cr == 1 else 2
    if max_seq_len is not None:
        n_env = int(max_seq_len) >> cshift
    else:
        if _is_capturing():
            raise RuntimeError(
                "run_varlen_bf16 without max_seq_len reads kv_lens.max() on "
                "host — pass max_seq_len (a capture-stable engine constant)"
            )
        n_env = int(kv_lens.max().item()) >> cshift
        n_env = 1 << max(n_env - 1, 1).bit_length()
    n_env = min(max(n_env, 1), npad)
    key = (num_rows, npad, k, n_env, nn, cr)
    lc = _VARLEN_CACHE_BF16.get(key)
    if lc is None:
        if _is_capturing():
            raise RuntimeError(
                "varlen launcher not compiled for this shape — warm up before CUDA graph capture"
            )
        lc = _varlen_launcher_bf16(num_rows, npad, k, n_env, nn, cr)
    idx = indices
    if idx.shape[1] != k:
        idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
    # Hint-free engines do not read the compiled kernel's pre_idx ABI slot.
    pre_arg = idx
    if lc[0] == "reg_clus":
        # compiled ABI: (logits, pre_idx, kv_lens, out, n_envelope)
        lc[1](lg, pre_arg, kv_lens, idx, lc[2])
    elif lc[0] == "reg":
        # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, CMP, QC, smem)
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    elif lc[0] == "clus":
        # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, npad, k,
        #                SCAP, CMP, dead DYN x5)
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    else:
        _, fn, pre, tail = lc
        fn(lg, pre_arg, idx, ws, *pre, kv_lens, *tail)
    return


def route_bands(
    b: int, npad: int, k: int, n_lo: int | None = None, n_hi: int | None = None
) -> list[tuple[int, int, dict[str, object]]]:
    """Enumerate maximal n-intervals on which route_static is constant.
    Dense O(n_hi - n_lo) scan of the pure host dispatch — an offline /
    engine-init tool (seconds for the 262144-token envelope), NOT a hot
    path. Returns [(n_lo, n_hi, static_plan), ...]."""
    lo = k + 1 if n_lo is None else max(n_lo, k + 1)
    hi = npad if n_hi is None else min(n_hi, npad)
    bands = []
    cur_key, cur_lo, cur_plan = None, lo, None
    for n in range(lo, hi + 1):
        st = route_static(b, n, npad, k)
        key = repr(st)
        if key != cur_key:
            if cur_key is not None:
                bands.append((cur_lo, n - 1, cur_plan))
            cur_key, cur_lo, cur_plan = key, n, st
    if cur_key is not None:
        bands.append((cur_lo, hi, cur_plan))
    return bands


# ===========================================================================
# ==== workspace ============================================================
# ===========================================================================
"""Per-device workspace slab for the multi-CTA SPLIT path.

Semantics:
  * ONE zero-initialised slab workspace per device, lazily allocated through
    the torch caching allocator;
  * keep-alive store: module dict `_ws_keep` (tensor refcount = keep-alive);
  * double-checked locking: lock-free hot-path load (a GIL-atomic dict get
    plays an acquire load), slow path re-checks under a mutex;
  * device index bounds `0 <= d < GVR_MAX_DEV` -- checked BEFORE the
    CUDA-ness of the tensor (run() resolves the default workspace before the
    input checks, so a CPU logits tensor dies here with "device index out of
    range: -1").

Concurrent STREAMS on one device that may both take the multi-CTA SPLIT path
must pass their own workspace via run_ws().

Size: workspace_bytes() = GVR_WS_BUF_OFF + MAXC*GCAP*sizeof(int2)
    = 2048 + 160*16384*8 = 20,973,568 B.

Kernel-facing view: the compiled main-family signature takes the workspace
as a 1-D contiguous int32 tensor (fake tensor dtype Int32, assumed_align=16
-- torch caching-allocator bases are 256B-aligned so the default slab always
satisfies it).  `kernel_view()` reproduces raw `workspace.data_ptr()`
semantics for arbitrary user tensors by aliasing the underlying storage at
the tensor's byte offset.
"""


# workspace geometry constants -- must match the device kernels
GVR_MAX_DEV = 64
_MAXC = 160
_GCAP = 16384
_GVR_WS_BUF_OFF = 2048
WS_BYTES = _GVR_WS_BUF_OFF + _MAXC * _GCAP * 8  # 20,973,568
assert WS_BYTES == 20_973_568

_mu = threading.Lock()  # slow-path mutex
_ws_keep = {}  # device index -> keep-alive int32 view


def workspace_bytes() -> int:
    """Workspace bytes required by the multi-CTA SPLIT path."""
    return WS_BYTES


def default_workspace(ref: torch.Tensor) -> torch.Tensor:
    """Per-device cached workspace slab.

    Returns the kernel-facing 1-D int32 view (zero-initialised on first use;
    the kernel restores the zeros it consumes, so one zeroing suffices for
    the lifetime of the cache entry)."""
    d = ref.get_device()
    if not (0 <= d < GVR_MAX_DEV):
        raise RuntimeError(f"device index out of range: {d}")
    ws = _ws_keep.get(d)  # hot path: one (GIL-atomic) load
    if ws is not None:
        return ws
    with _mu:  # slow path: double-checked
        ws = _ws_keep.get(d)
        if ws is not None:
            return ws
        # lazy zeros via the torch caching allocator, viewed int32 for the
        # DSL launch signature.
        buf = torch.zeros(WS_BYTES, dtype=torch.uint8, device=ref.device)
        ws = buf.view(torch.int32)
        _ws_keep[d] = ws  # keep-alive (ws_keep[d] = tensor)
        return ws


def validate_run_ws(workspace: torch.Tensor, logits: torch.Tensor) -> None:
    """run_ws() workspace hardening, in a fixed predicate order:
    CUDA + same device as logits; numel*element_size >= workspace_bytes();
    base 16-byte aligned (the DSL workspace fake declares assumed_align=16)."""
    if not (workspace.is_cuda and workspace.get_device() == logits.get_device()):
        raise RuntimeError("workspace must be a CUDA tensor on the same device")
    if workspace.numel() * workspace.element_size() < WS_BYTES:
        raise RuntimeError(f"workspace too small: need {WS_BYTES} bytes")
    if workspace.data_ptr() & 15:
        raise RuntimeError("workspace must be 16-byte aligned")


def kernel_view(workspace: torch.Tensor) -> torch.Tensor:
    """Raw-pointer view of a user workspace tensor: alias the first WS_BYTES
    bytes at the tensor's data_ptr() as int32[WS_BYTES/4], ignoring
    dtype/shape.

    NOTE: the DSL-side fake tensor declares assumed_align=16, matching the
    validate_run_ws base-alignment check, so misaligned workspaces fail on
    the host with a clear message instead of at DSL conversion."""
    if (
        workspace.dtype is torch.int32
        and workspace.dim() == 1
        and workspace.is_contiguous()
        and workspace.storage_offset() == 0
        and workspace.numel() == WS_BYTES // 4
    ):
        return workspace  # already the canonical view
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


# ===========================================================================
# ==== operator entry =======================================================
# ===========================================================================
"""Operator entry: input hardening, dispatch, and bind-once launch cache.

Hardening checks run in a fixed order with fixed predicates:
  1. all three tensors CUDA
  2. dtypes: logits f32, pre_idx i32, indices i32
  3. all 2-D
  4. all contiguous
  5. n_valid unwrap: python-int fast path (strict integral cast); Tensor
     path checks torch.cuda.is_current_stream_capturing() FIRST and fails
     loudly, else .item() (the D2H sync)
  6. b/npad from logits, k = pre_idx.size(1)
  7. b == 0 -> early no-op
  8. npad % 4 == 0 (float4 row loads)
  9. logits base 16-byte aligned
 10. pre_idx/indices batch dims match
 11. indices width >= k
 12. n_valid >= 0
 13. n = min(nv, npad) clamped in unbounded ints BEFORE any narrowing

Dispatch: route(b, n, npad, k) -> compile cache keyed on (kernel family,
constexpr tuple) in the device module -> bind-once launch cache keyed on the
shape key (b, n, npad, k): caches the compiled callable + the prebuilt
runtime-scalar arg pack as plain Python ints (never pre-wrapped
cutlass.Int32 -- the FFI per-argument cost is paid every call regardless;
pre-binding removes only route()/marshal-prep work).

Error contract: launch failures surface as exceptions WITH
(b, n, npad, k) context.

The device module is imported LAZILY (first shape that routes to it), so a
missing/broken module only fails when actually reached, with (b, n, npad, k)
context.  The per-family compiled ABIs are documented at each launcher
builder in _build_launcher; only the main family takes the workspace.
"""


# shape key (b, n, npad, k) -> (fn, args tuple of python ints, needs_ws)
_LAUNCH_CACHE = {}
_DUMMY_KV = {}


def _dummy_kv(dev_index, device):
    """Cached 1-element int32 tensor per device — the dead kv_lens slot of
    the extended gvr_main ABI in legacy (batch-uniform) mode."""
    t = _DUMMY_KV.get(dev_index)
    if t is None:
        t = torch.zeros(1, dtype=_I32, device=device)
        _DUMMY_KV[dev_index] = t
    return t


# hot-path local bindings: each torch.<attr> lookup costs ~0.1 us and the
# validation battery runs on EVERY call
_F32 = torch.float32
_I32 = torch.int32
_TENSOR = torch.Tensor
_is_capturing = torch.cuda.is_current_stream_capturing
_index = operator.index
_ws_hot = _ws_keep  # shared dict object (hot-path load)
_GVR_MAX_DEV = GVR_MAX_DEV


# ---------------------------------------------------------------------------
# per-family launcher builders (cold path: once per distinct shape key)
# ---------------------------------------------------------------------------
def _build_launcher(b, n, npad, k):
    rd = route(b, n, npad, k)
    fam = rd["kernel"]
    tpl = tuple(rd["tpl"])
    rt = rd["rt"]
    if fam in ("reg", "regimg"):
        dev = _device()
        raw = dev.get_compiled__reg(tpl)

        # compiled ABI: (logits, pre_idx, kv_lens, out, n, CMP, QC,
        # smem_total) -- kv_lens is the dead varlen slot in batch-uniform
        # mode (cached dummy tensor)
        def fn(lg, pi, o, *a, _raw=raw):
            _raw(lg, pi, _dummy_kv(lg.get_device(), lg.device), o, *a)

        args = (rt["n"], rt["CMP"], rt["QC"], dev.STATIC_BYTES + rd["smem"])
        return (fn, args, False)
    if fam == "main":
        dev = _device()
        raw = dev.get_compiled(tpl)

        # compiled ABI: (logits, pre_idx, out, ws, n, npad, k, SCAP_, CMP_,
        #                R, SMP, TGT, Q, SS2, TGT2,
        #                kv_lens, aim_base, sfac, amin, sd_en, tsh_en)
        # [SCAP_/CMP_ dead, ABI parity; the trailing varlen block is dead in
        #  legacy mode — a cached dummy kv_lens tensor + five zeros]
        def fn(lg, pi, o, w, *a, _raw=raw):
            _raw(lg, pi, o, w, *a, _dummy_kv(lg.get_device(), lg.device), 0, 0, 0, 0, 0)

        args = (
            rt["n"],
            rt["npad"],
            rt["k"],
            rt["SCAP_"],
            rt["CMP_"],
            rt["R"],
            rt["SMP"],
            rt["TGT"],
            rt["Q"],
            rt["SS2"],
            rt["TGT2"],
        )
        return (fn, args, True)
    if fam == "clus":
        dev = _device()
        # compile key carries the smem-extent scalars (scap/cmp_); compiled
        # ABI: (logits, pre_idx, kv_lens, out, n, npad, k, SCAP, CMP, SMP,
        #       TGT, Q, SS2, TGT2) -- NO workspace; kv_lens is the dead
        # varlen slot in batch-uniform mode (cached dummy tensor)
        fn = dev.get_compiled__clus(tpl, scap=rt["SCAP"], cmp_=rt["CMP"])
        args = (
            rt["n"],
            rt["npad"],
            rt["k"],
            rt["SCAP"],
            rt["CMP"],
            rt["SMP"],
            rt["TGT"],
            rt["Q"],
            rt["SS2"],
            rt["TGT2"],
        )

        def _call(lg, pi, idx, _fn=fn, _args=args):
            _fn(lg, pi, _dummy_kv(lg.get_device(), lg.device), idx, *_args)

        return (_call, (), False)
    if fam == "reg_clus":
        dev = _device()
        # compiled ABI: (logits, pre_idx, kv_lens, out, n) -- kv_lens is the
        # dead varlen slot in batch-uniform mode (cached dummy tensor);
        # smem/k derived in-module
        fn = dev.get_compiled__regclus(tpl)
        n_arg = rt["n"]

        def _call(lg, pi, idx, _fn=fn, _n=n_arg):
            _fn(lg, pi, _dummy_kv(lg.get_device(), lg.device), idx, _n)

        return (_call, (), False)
    # unreachable: route() only emits the five families above
    raise RuntimeError(f"unknown dispatch family {fam!r}")


# ---------------------------------------------------------------------------
# shared implementation of the batch-uniform entries
# ---------------------------------------------------------------------------
def _run_impl(logits, pre_idx, n_valid, indices, ws, values=None):
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
    if not (logits.is_contiguous() and pre_idx.is_contiguous() and indices.is_contiguous()):
        raise RuntimeError("tensors must be contiguous")

    # n_valid unwrap: tensor path = D2H sync, illegal under CUDA graph
    # capture -- fail loudly instead of crashing the capture.
    if isinstance(n_valid, _TENSOR):
        if _is_capturing():
            raise RuntimeError(
                "tensor n_valid requires a D2H sync, illegal under CUDA "
                "graph capture — pass n_valid as a python int"
            )
        nv = int(n_valid.item())
    else:
        # strict integral cast (rejects floats/strings)
        nv = _index(n_valid)

    b, npad = lsh
    k = psh[1]
    if b == 0:  # empty batch: no-op
        return
    if npad & 3:
        raise RuntimeError(f"npad (logits stride) must be a multiple of 4, got {npad}")
    if logits.data_ptr() & 15:
        raise RuntimeError(
            "logits base must be 16-byte aligned (storage-offset views break the float4 row loads)"
        )
    if psh[0] != b or ish[0] != b:
        raise RuntimeError(f"batch dims must match: logits {b} pre_idx {psh[0]} indices {ish[0]}")
    if ish[1] < k:
        raise RuntimeError(f"indices width {ish[1]} < k={k} (k is pre_idx.size(1))")
    if nv < 0:
        raise RuntimeError(f"n_valid must be non-negative, got {nv}")
    # clamp BEFORE any narrowing (python ints are unbounded, so min() is the
    # exact 64-bit clamp)
    n = nv if nv < npad else npad

    # CUDA out-indexing mirror: every kernel derives O = out + row*k --
    # flat PACKED rows, ignoring the actual indices width.  The DSL kernels
    # index out[row, :] with the tensor's own row stride, so a wider
    # `indices` must be re-viewed packed (pure view, no copy; contiguity
    # already checked).
    if ish[1] != k:
        indices = indices.reshape(-1)[: b * k].view(b, k)

    # ---- optional values output (production parity, default OFF) ------------
    # dsa.py allocates the values scratch only for the non-CuTeDSL path, so
    # values stay opt-in. The indices are exact, so a gather epilogue
    # reproduces the in-kernel writeback bit-for-bit; the constexpr in-kernel
    # form rides the CUDA-graph per-row rewrite.
    if values is not None:
        if not values.is_cuda:
            raise RuntimeError("values must be CUDA")
        if values.dtype is not _F32:
            raise RuntimeError("values must be float32")
        vsh = values.shape
        if len(vsh) != 2 or not values.is_contiguous():
            raise RuntimeError("values must be 2-D contiguous")
        if vsh[0] != b:
            raise RuntimeError(f"batch dims must match: logits {b} values {vsh[0]}")
        if vsh[1] < k:
            raise RuntimeError(f"values width {vsh[1]} < k={k}")
        if vsh[1] != k:
            values = values.reshape(-1)[: b * k].view(b, k)

    # ---- n <= k short path (heuristicTopKDecode.cu parity) ------------------
    # Every valid position is in the top-K: emit identity indices and pad the
    # tail with -1 (the production pad convention; downstream treats -1 as
    # invalid). Order is contract-irrelevant — exactness is tie-interchangeable
    # SET semantics. Torch-op path for now; the CUDA-graph-safe per-row rewrite
    # moves this branch in-kernel (it cannot fall back per row inside a graph).
    if n <= k:
        if n > 0:
            indices[:, :n] = torch.arange(n, dtype=_I32, device=indices.device)
            if values is not None:
                values[:, :n] = logits[:, :n]
        if n < k:
            indices[:, n:] = -1
            if values is not None:
                values[:, n:] = torch.finfo(_F32).min  # -FLT_MAX pad
        return

    key = (b, n, npad, k)
    lc = _LAUNCH_CACHE.get(key)
    if lc is None:
        lc = _build_launcher(b, n, npad, k)
        _LAUNCH_CACHE[key] = lc
    fn, args, needs_ws = lc
    try:
        if needs_ws:
            fn(logits, pre_idx, indices, ws, *args)
        else:
            fn(logits, pre_idx, indices, *args)
    except Exception as e:
        raise RuntimeError(f"gvr_topk launch failed (b={b} n={n} npad={npad} k={k}): {e}") from e
    if values is not None:
        # same epilogue as run_varlen: a (never-expected) negative index
        # degrades to -FLT_MAX instead of a context-poisoning device assert
        idx64 = indices.to(torch.int64)
        values.copy_(logits.gather(1, idx64.clamp_min(0)))
        values.masked_fill_(indices < 0, torch.finfo(_F32).min)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------
def run(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    n_valid: int,
    indices: torch.Tensor,
    values: torch.Tensor | None = None,
) -> None:
    """TESTING/BENCH ONLY — production callers must use ``run_varlen`` (per-request
    device kv_lens; this entry assumes one batch-uniform host ``n_valid``,
    which real serving batches do not satisfy).

    Fast 4-arg form.  ``values`` (optional DPS output, default None = OFF)
    mirrors the production values writeback; see _run_impl.
    The default per-device slab workspace is resolved FIRST (a CPU logits
    tensor therefore dies with 'device index out of range').
    Hot path inlines the device check + atomic load + cache hit; the slow
    path allocates under the workspace lock."""
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:  # checked on EVERY call
        raise RuntimeError(f"device index out of range: {d}")
    ws = _ws_hot.get(d)
    if ws is None:
        ws = default_workspace(logits)
    _run_impl(logits, pre_idx, n_valid, indices, ws, values)


def run_ws(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    n_valid: int,
    indices: torch.Tensor,
    workspace: torch.Tensor,
    values: torch.Tensor | None = None,
) -> None:
    """TESTING/BENCH ONLY — production callers must use ``run_varlen(workspace=...)``.

    Explicit-workspace form for multi-stream callers."""
    validate_run_ws(workspace, logits)
    _run_impl(logits, pre_idx, n_valid, indices, kernel_view(workspace), values)


def run_varlen(
    logits: torch.Tensor,
    kv_lens: torch.Tensor,
    indices: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
    values: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    workspace: torch.Tensor | None = None,
) -> None:
    """Run hint-free self-sampling Top-K with per-request device KV lengths.

    Row semantics (mirror of ``heuristicTopKDecode.cu`` and the in-tree
    ``cute_dsl_gvr_topk_decode`` runner):

      ``num_rows = logits.shape[0]``, ``batch = num_rows // next_n``;
      ``kv_lens`` int32 ``[batch]`` — per-request TOTAL cache length in
      UNCOMPRESSED token space (dsa.py ``metadata.kv_lens_cuda_runtime``,
      not new-token seq_lens); row ``r`` uses
      ``n_r = (kv_lens[r // next_n] - next_n + (r % next_n) + 1) //
      compress_ratio`` valid entries (cr 1 = DSv3.2, 4 = DSv4 Flash/Pro);
      the bracket is derived from the current row itself (register families:
      min/max fold of the first k row values; streaming families do not
      consume a temporal hint on the accept path); ``k`` comes from
      ``indices.shape[1]``;
      per-row ``n_r <= k`` takes the short path (identity + ``-1`` tail).

    The per-row in-kernel engine launches once for the whole batch. Each CTA
    reads its row's kv_len on device and re-derives the sampling ladder (route_dynamic
    formula mirror), so with ``max_seq_len`` given (a capture-stable engine
    constant, e.g. dsa.py's ``indexer_max_seq_len``) the call performs NO
    host reads.  Without ``max_seq_len`` the envelope comes from ONE
    ``kv_lens.max()`` host read (documented sync, refused under capture).

    KNOWN LIMITATION: on rows containing NaN logits the selected index SET
    can differ from ``heuristicTopKDecode.cu`` (both kernels order NaNs
    implementation-specifically). Finite inputs — including +/-inf and
    denormals — are tie-aware exact.

    CONTRACT: correct and dispatched for any ``num_rows``
    (BS 1..1024+ x next_n) and any envelope up to 1M kv tokens.  Family
    selection (streaming main / clustered register-resident) is a pure
    function of the capture-stable launcher key.
    """
    if logits.dtype is not torch.float32:
        raise RuntimeError(
            f"logits must be float32 (got {logits.dtype}); bf16/fp16 paths "
            "are a follow-up — see the PR roadmap"
        )
    if not (isinstance(kv_lens, _TENSOR) and kv_lens.is_cuda):
        raise RuntimeError("kv_lens must be a CUDA tensor")
    if kv_lens.dtype is not _I32:
        raise RuntimeError("kv_lens must be int32")
    if kv_lens.dim() != 1:
        raise RuntimeError("kv_lens must be 1-D")
    nn = _index(next_n)
    cr = _index(compress_ratio)
    if nn < 1:
        raise RuntimeError(f"next_n must be >= 1, got {nn}")
    if cr not in (1, 4):
        raise RuntimeError(f"compress_ratio must be 1 (DSv3.2) or 4 (DSv4), got {cr}")
    if len(logits.shape) != 2:
        raise RuntimeError("logits must be 2-D")
    num_rows = logits.shape[0]
    if num_rows == 0:
        return
    if num_rows % nn:
        raise RuntimeError(f"num_rows {num_rows} not divisible by next_n {nn}")
    batch = num_rows // nn
    if kv_lens.shape[0] != batch:
        raise RuntimeError(f"kv_lens length {kv_lens.shape[0]} != num_rows/next_n = {batch}")
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    if workspace is not None:
        # multi-stream escape hatch (run_ws parity): concurrent varlen
        # launches on one device must not share the SPLIT publish slab
        validate_run_ws(workspace, logits)
        ws = kernel_view(workspace)
    else:
        ws = _ws_hot.get(d)
        if ws is None:
            ws = default_workspace(logits)

    # ---- per-row in-kernel engine (gvr_main varlen port) ----------------
    # Full validation battery (the engine bypasses _run_impl — every
    # check the batch-uniform path enforces is replayed here; the
    # batch-dim check is CRITICAL: the kernel grid comes from
    # logits.shape[0], so a short indices/values tensor would be written
    # out of bounds).
    if not (logits.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if logits.dtype is not _F32 or indices.dtype is not _I32:
        raise RuntimeError("logits must be float32; indices must be int32")
    if len(indices.shape) != 2 or indices.shape[0] != num_rows:
        raise RuntimeError(
            f"indices must be [num_rows={num_rows}, >=k], got {tuple(indices.shape)}"
        )
    k = indices.shape[1]
    if not (indices.is_contiguous() and kv_lens.is_contiguous()):
        raise RuntimeError("indices/kv_lens must be contiguous")
    # logits: accept row-major views with a wider row stride (the DSL
    # paged-MQA logits arena is 256-aligned and column-sliced — a legal
    # NON-contiguous view). The kernel only needs (base, row stride):
    # widen back to a compact [rows, stride] view over the same storage;
    # the tail columns are never classified (per-row n gates all reads).
    if logits.stride(1) != 1:
        raise RuntimeError("logits inner stride must be 1")
    npad = logits.stride(0) if num_rows > 1 else logits.shape[1]
    lg = logits
    if not logits.is_contiguous():
        need = logits.storage_offset() + num_rows * npad
        if logits.untyped_storage().size() // 4 < need:
            raise RuntimeError("logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    if npad & 3:
        raise RuntimeError(f"npad (logits row stride) must be a multiple of 4, got {npad}")
    if lg.data_ptr() & 15:
        raise RuntimeError("logits base must be 16-byte aligned")
    if values is not None:
        if not values.is_cuda or values.dtype is not _F32:
            raise RuntimeError("values must be CUDA float32")
        if (
            len(values.shape) != 2
            or values.shape[0] != num_rows
            or values.shape[1] < k
            or not values.is_contiguous()
        ):
            raise RuntimeError(
                f"values must be contiguous [num_rows={num_rows}, >=k], got {tuple(values.shape)}"
            )
    cshift = 0 if cr == 1 else 2
    if max_seq_len is not None:
        n_env = int(max_seq_len) >> cshift
    else:
        if _is_capturing():
            raise RuntimeError(
                "run_varlen without max_seq_len reads kv_lens.max() on "
                "host — pass max_seq_len (a capture-stable engine "
                "constant) under CUDA graph capture"
            )
        n_env = int(kv_lens.max().item()) >> cshift
        # eager mode: quantize the data-dependent envelope up to the next
        # power of two so a growing decode does not recompile at every
        # R increment (bounded plans, bounded _VARLEN_CACHE)
        n_env = 1 << max(n_env - 1, 1).bit_length()
    n_env = min(max(n_env, 1), npad)
    key = (num_rows, npad, k, n_env, nn, cr)
    lc = _VARLEN_CACHE.get(key)
    if lc is None:
        if _is_capturing():
            raise RuntimeError(
                "varlen launcher not compiled for this shape — warm up before CUDA graph capture"
            )
        lc = _varlen_launcher(num_rows, npad, k, n_env, nn, cr)
    idx = indices
    if idx.shape[1] != k:
        idx = idx.reshape(-1)[: num_rows * k].view(num_rows, k)
    vals = values
    if vals is not None and vals.shape[1] != k:
        vals = vals.reshape(-1)[: num_rows * k].view(num_rows, k)
    # Hint-free engines do not read the compiled kernel's pre_idx ABI slot.
    pre_arg = idx
    if lc[0] == "reg_clus":
        # compiled ABI: (logits, pre_idx, kv_lens, out, n_envelope)
        lc[1](lg, pre_arg, kv_lens, idx, lc[2])
    elif lc[0] == "reg":
        # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, CMP, QC, smem)
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    elif lc[0] == "clus":
        # compiled ABI: (logits, pre_idx, kv_lens, out, n_env, npad, k,
        #                SCAP, CMP, dead DYN x5)
        lc[1](lg, pre_arg, kv_lens, idx, *lc[2])
    else:
        _, fn, pre, tail = lc
        fn(lg, pre_arg, idx, ws, *pre, kv_lens, *tail)
    if vals is not None:
        idx64 = idx.to(torch.int64)
        vals.copy_(lg.gather(1, idx64.clamp_min(0)))
        vals.masked_fill_(idx < 0, torch.finfo(_F32).min)
    return


def run_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    max_row_len: int | None = None,
    workspace: torch.Tensor | None = None,
) -> None:
    """Hint-free self-sampling Top-K for prefill: row ``r`` selects the Top-K of
    ``logits[r, ks:ke]`` (compressed columns) into the local frame (column - ks)
    with a -1 pad; ``nv <= k`` rows get the identity, as ``indexer_topk_prefill``.
    No device reads, never compiles under capture; trusts 0 <= ks <= ke <= shape[1]."""
    if logits.dtype is not _F32:
        raise RuntimeError(
            f"logits must be float32 (got {logits.dtype}); bf16/fp16 paths "
            "are a follow-up — see the PR roadmap"
        )
    for _nm, _t in (("row_starts", row_starts), ("row_ends", row_ends)):
        if not (isinstance(_t, _TENSOR) and _t.is_cuda):
            raise RuntimeError(f"{_nm} must be a CUDA tensor")
        if _t.dtype is not _I32:
            raise RuntimeError(f"{_nm} must be int32")
        if _t.dim() != 1:
            raise RuntimeError(f"{_nm} must be 1-D")
        if not _t.is_contiguous():
            raise RuntimeError(f"{_nm} must be contiguous")
    if len(logits.shape) != 2:
        raise RuntimeError("logits must be 2-D")
    num_rows = logits.shape[0]
    if num_rows == 0:
        return
    if row_starts.shape[0] != num_rows or row_ends.shape[0] != num_rows:
        raise RuntimeError(
            f"row_starts/row_ends length must equal logits.shape[0]={num_rows}, "
            f"got {row_starts.shape[0]}/{row_ends.shape[0]}"
        )
    if not (logits.is_cuda and indices.is_cuda):
        raise RuntimeError("all tensors must be CUDA")
    if indices.dtype is not _I32:
        raise RuntimeError("indices must be int32")
    if len(indices.shape) != 2 or indices.shape[0] != num_rows:
        raise RuntimeError(f"indices must be [num_rows={num_rows}, k], got {tuple(indices.shape)}")
    if not indices.is_contiguous():
        raise RuntimeError("indices must be contiguous")
    k = indices.shape[1]
    if k < 4 or (k & 3):
        raise RuntimeError(f"index_topk must be a multiple of 4 and >= 4, got {k}")
    if indices.data_ptr() & 15:
        raise RuntimeError("indices base must be 16-byte aligned")
    if logits.stride(1) != 1:
        raise RuntimeError("logits inner stride must be 1")
    # key on stride(0) for every row count: DeepGEMM prefill rows are 1024B-aligned
    # with slack, and the varlen 1-row shape[1] rule would reject odd-width tiles.
    npad = logits.stride(0)
    if npad & 3:
        raise RuntimeError(f"npad (logits row stride) must be a multiple of 4, got {npad}")
    if logits.data_ptr() & 15:
        raise RuntimeError("logits base must be 16-byte aligned")
    d = logits.get_device()
    if not 0 <= d < _GVR_MAX_DEV:
        raise RuntimeError(f"device index out of range: {d}")
    lg = logits
    if logits.shape[1] != npad:
        need = logits.storage_offset() + num_rows * npad
        if logits.untyped_storage().size() // 4 < need:
            raise RuntimeError("logits view storage too small to widen to its row stride")
        lg = logits.as_strided((num_rows, npad), (npad, 1), logits.storage_offset())
    if workspace is not None:
        validate_run_ws(workspace, logits)
        ws = kernel_view(workspace)
    else:
        ws = _ws_hot.get(d)
        if ws is None:
            ws = default_workspace(logits)
    n_env = _index(max_row_len) if max_row_len is not None else logits.shape[1]
    n_env = min(max(n_env, 1), npad)
    n_bucket = _prefill_bucket(n_env)
    for r0 in range(0, num_rows, _PREFILL_ROW_SLAB):
        r1 = min(r0 + _PREFILL_ROW_SLAB, num_rows)
        tier = _prefill_tier(r1 - r0)
        lc = _PREFILL_CACHE.get(_prefill_cache_key(tier, k, n_bucket))
        if lc is None:
            if _is_capturing():
                raise RuntimeError(
                    "prefill launcher not compiled for this shape — warm up "
                    "before CUDA graph capture"
                )
            lc = _prefill_launcher(tier, k, n_bucket)
        _, fn, (scap, cmp_), tail = lc
        # varlen main ABI: pre_idx slot = row_ends, kv_lens slot = row_starts;
        # only npad / k / SCAP_ / CMP_ matter (R=1), the other scalars are dead.
        pre = (0, npad, k, scap, cmp_, 1, 0, 0, 0, 0, 0)
        fn(lg[r0:r1], row_ends[r0:r1], indices[r0:r1], ws, *pre, row_starts[r0:r1], *tail)
    return


def prefill_ready(logits: torch.Tensor, indices: torch.Tensor) -> bool:
    """True iff ``run_prefill(logits, ..., indices)`` would launch without
    compiling — the same (tier, k, envelope bucket) keys it looks up, so a
    caller can route around the engine under CUDA graph capture. Host-only."""
    num_rows = logits.shape[0]
    if num_rows == 0:
        return True
    k = indices.shape[1]
    npad = logits.stride(0)
    n_bucket = _prefill_bucket(min(max(logits.shape[1], 1), max(npad, 1)))
    for r0 in range(0, num_rows, _PREFILL_ROW_SLAB):
        tier = _prefill_tier(min(r0 + _PREFILL_ROW_SLAB, num_rows) - r0)
        if _prefill_cache_key(tier, k, n_bucket) not in _PREFILL_CACHE:
            return False
    return True


__all__ = [
    "route",
    "route_static",
    "route_dynamic",
    "route_split",
    "route_bands",
    "run",
    "run_ws",
    "run_varlen",
    "run_prefill",
    "prefill_ready",
    "warmup_varlen",
    "warmup_prefill",
    "workspace_bytes",
    "WS_BYTES",
    "default_workspace",
    "validate_run_ws",
    "kernel_view",
]


# --------------------------------------------------------------------------
# warmup: pre-compile the varlen engine for an engine envelope so no live
# request pays the first-touch DSL JIT (mirrors warmup_heuristic_topk_decode
# and warmup_cute_dsl_radix_topk). Idempotent per (device, geometry) key.
# CUDA-graph capture warmup naturally compiles the captured batch sizes;
# this covers the eager/first-touch path (num_rows defaults to (1,)).
_VARLEN_WARMUP_DONE: set = set()
_VARLEN_WARMUP_LOCK = threading.Lock()


def warmup_varlen(
    top_k: int,
    max_seq_len: int,
    compress_ratio: int = 1,
    next_n: int = 1,
    num_rows_list: Sequence[int] = (1,),
    row_stride: int | None = None,
) -> None:
    """TESTING/INIT ONLY — compile the varlen engine's envelope tuples.

    One tiny real launch per requested ``num_rows`` (compile keys do not
    depend on tensor contents). Uses the current CUDA device. The done-key
    is recorded only after every launch succeeds, so a failed or interrupted
    warmup is retried on the next call instead of short-circuiting to an
    uncompiled engine.

    ``row_stride`` must be the logits row stride the serving producer will
    emit: the launcher key includes it, so a warmup at a different stride
    compiles a variant dispatch never looks up. Callers that know the
    producer layout (e.g. the DSL paged-MQA arena's 256-element rounding)
    must pass it; the 64-element default only matches producers that round
    the same way.

    """
    dev = torch.cuda.current_device()
    nn = max(1, int(next_n))
    # round each request down to a next_n multiple (min next_n) and dedup
    req_rows = sorted({max(int(r) - int(r) % nn, nn) for r in num_rows_list})
    if not req_rows:
        return
    # BAND-AWARE enumeration: the engine compile key depends on the plan's
    # constexpr tuple (+ r_const family axis), NOT on the exact row count, so
    # warming ONE representative row per distinct engine key covers every row
    # count up to the largest request. Representatives are the first row of
    # each band, which keeps the warmup allocation bounded (~a few hundred
    # rows) even when CUDA-graph batch lists reach thousands of rows.
    n_env_c = max(1, int(max_seq_len) // int(compress_ratio))
    npad_c = (n_env_c + 63) // 64 * 64 if row_stride is None else int(row_stride)
    seen_keys = set()
    rows_list = []
    r = nn
    r_max = req_rows[-1]
    while r <= r_max:
        plan_free = route(r, max(min(n_env_c, npad_c), int(top_k) + 1), npad_c, int(top_k))
        if plan_free["kernel"] == "reg_clus":
            ekey = ("reg_clus", tuple(plan_free["tpl"]))
        elif plan_free["kernel"] in ("reg", "regimg"):
            ekey = ("reg", tuple(plan_free["tpl"]))
        else:
            p = route_streaming(
                r,
                max(min(n_env_c, npad_c), int(top_k) + 1),
                npad_c,
                int(top_k),
                force_main=True,
            )
            ekey = ("main", tuple(p["tpl"][:6]), p["rt"]["R"])
        if ekey not in seen_keys:
            seen_keys.add(ekey)
            rows_list.append(r)
        r += nn
    if not rows_list:
        return
    n_env = max(1, int(max_seq_len) // int(compress_ratio))
    if row_stride is None:
        npad = (n_env + 63) // 64 * 64
    else:
        npad = int(row_stride)
        if npad < n_env or npad % 4:
            raise RuntimeError(
                f"row_stride must be a float4-multiple >= n_env={n_env}, got {row_stride}"
            )
    key = (
        dev,
        int(top_k),
        int(max_seq_len),
        int(compress_ratio),
        nn,
        tuple(rows_list),
        npad,
    )
    # The done key covers the GPU band launches only (one per engine compile
    # key). The exact-row launcher population below is keyed by the requested
    # row counts, which the band key does not see, so it always runs: a later
    # call with a new row count inside an already-warmed band must still
    # create that row count's entry, or capture at it raises not-compiled.
    with _VARLEN_WARMUP_LOCK:
        bands_done = key in _VARLEN_WARMUP_DONE
    if not bands_done:
        rows_max = rows_list[-1]
        # one allocation at the largest geometry; smaller row counts run on
        # contiguous prefix views (compile keys depend on shapes only)
        logits = torch.zeros((rows_max, npad), dtype=torch.float32, device=dev)
        kv_lens = torch.full((rows_max // nn,), int(max_seq_len), dtype=torch.int32, device=dev)
        out = torch.empty((rows_max, int(top_k)), dtype=torch.int32, device=dev)
        for rows in rows_list:
            batch = rows // nn
            run_varlen(
                logits[:rows],
                kv_lens[:batch],
                out[:rows],
                next_n=nn,
                compress_ratio=int(compress_ratio),
                max_seq_len=int(max_seq_len),
            )
        del logits, kv_lens, out
        torch.cuda.synchronize()
    # band launches compiled every ENGINE; now populate the per-row-count
    # LAUNCHER cache entries for the exact requested row counts (pure host
    # work, zero allocation/launch — engines hit the compile cache), so a
    # CUDA-graph capture at any requested geometry finds its key immediately.
    n_env_l = min(max(int(max_seq_len) >> (0 if int(compress_ratio) == 1 else 2), 1), npad)
    for r in req_rows:
        _varlen_launcher(r, npad, int(top_k), n_env_l, nn, int(compress_ratio))
    if not bands_done:
        with _VARLEN_WARMUP_LOCK:
            _VARLEN_WARMUP_DONE.add(key)


_PREFILL_WARMUP_DONE: set = set()
_PREFILL_WARMUP_LOCK = threading.Lock()


def warmup_prefill(
    top_k: int,
    max_cols: int,
    num_rows_list: Sequence[int] = (1, 149, 297),
    row_stride: int | None = None,
) -> None:
    """Compile the prefill engine set before serving (<=6 per k): the tier-0 arm
    walks the pow2 envelope buckets up to 32768, tiers 1/2 need one launch each.
    ``max_cols`` is the compressed max column count; idempotent per done-key."""
    dev = torch.cuda.current_device()
    k = int(top_k)
    max_cols = int(max_cols)
    lo = _prefill_bucket(k + 1)
    hi = _prefill_bucket(max_cols)
    buckets = []
    b = lo
    while b <= hi:
        buckets.append(b)
        b <<= 1
    if not buckets:
        buckets = [hi]
    keys = {}  # cache_key -> (tier, bucket) representative for the launch
    for rows in num_rows_list:
        tier = _prefill_tier(int(rows))
        bset = buckets if tier == 0 else buckets[:1]
        for bk in bset:
            keys.setdefault(_prefill_cache_key(tier, k, bk), (tier, bk))
    done_key = (dev, k, max_cols, tuple(sorted(int(r) for r in num_rows_list)), row_stride)
    with _PREFILL_WARMUP_LOCK:
        if done_key in _PREFILL_WARMUP_DONE:
            return
    for tier, bk in keys.values():
        rows = _PREFILL_TIER_ROWS[tier]
        stride = row_stride if row_stride is not None else ((bk + 256 + 255) // 256 * 256)
        if stride < bk or stride % 4:
            stride = (max(stride, bk) + 256 + 255) // 256 * 256
        logits = torch.zeros((rows, stride), dtype=torch.float32, device=dev)
        ks = torch.zeros((rows,), dtype=torch.int32, device=dev)
        ke = torch.full((rows,), bk, dtype=torch.int32, device=dev)
        out = torch.empty((rows, k), dtype=torch.int32, device=dev)
        run_prefill(logits[:, :bk], ks, ke, out, max_row_len=bk)
        del logits, ks, ke, out
    torch.cuda.synchronize()
    with _PREFILL_WARMUP_LOCK:
        _PREFILL_WARMUP_DONE.add(done_key)
