#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Assemble tensorrt_llm gvr_topk_decode_ms.py from the op21 bench sources.

Deterministic extraction (exact line ranges from the frozen bench files at
op21 iter11 HEAD) + content-asserted textual transforms. Emits the PR-1
kernel-variant file. Re-runnable; every slice and every edit asserts the
expected source content so silent drift in the bench files fails loudly.

Sources (validated composition, nsys + exactness gates at bf04cebc40):
  base = ops/cute_vendored/blackwell/top_k/gvr_topk_decode.py  (pre-unified
         single-CTA production kernel #14602 lineage; byte-identical to the
         kernel op21 subclassed)
  mt   = op18_gvr_1cta_multithresh/src/gvr_mt_op.py            (M-ary ladder)
  ms   = op21_gvr_prod/src/gvr_ms_op.py                        (sandwich)
  msc  = op21_gvr_prod/src/gvr_msc_op.py                       (cluster)

Dropped (falsified / bench-only, all const_expr-dead in the production
config so the compiled kernels are unchanged): place_mode 0/1/3/4 offline
tables, smem-resident row (op20 iter6 no-op), dist_p1 / dist_p4 (op21
iter3/4 falsified), phase2_secant_search + kFTarget (replaced by the
ladder), all host dispatch tables and OP21_* env knobs (constructor flags
now; host policy moves to the runner).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUCKET = HERE.parent
BENCH = BUCKET.parent

SRC = {
    "base": BENCH / "ops/cute_vendored/blackwell/top_k/gvr_topk_decode.py",
    "mt": BUCKET.parent / "op18_gvr_1cta_multithresh/src/gvr_mt_op.py",
    "ms": BUCKET / "src/gvr_ms_op.py",
    "msc": BUCKET / "src/gvr_msc_op.py",
}
LINES = {k: open(p).read().splitlines(keepends=True) for k, p in SRC.items()}


def cut(tag, lo, hi, must_contain=None):
    """1-based inclusive slice with an optional content anchor assert."""
    seg = "".join(LINES[tag][lo - 1:hi])
    if must_contain is not None:
        assert must_contain in seg, (
            f"{tag}:{lo}-{hi} missing anchor {must_contain!r}")
    return seg


def edit(seg, old, new, count=1, tag=""):
    n = seg.count(old)
    assert n == count, f"edit[{tag}]: found {n} of expected {count}: {old[:80]!r}"
    return seg.replace(old, new)


# ===========================================================================
# Module header
# ===========================================================================
HEADER = '''\
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

"""GVR-MS (multi-threshold sandwich) Top-K decode kernel — cuTe DSL, sm_100.

Sibling variant of ``gvr_topk_decode.py`` (opt-in via the runner; see
``CuteDSLGvrTopKDecodeMsRunner`` in ``cute_dsl_custom_ops.py``). Same
operator contract as the GVR kernel: per-row exact top-K indices of
``logits[row, :N_eff(row)]`` with ``pre_idx`` (previous-step top-K) as a
seeding hint, request-level ``seq_lens``, ``next_n`` MTP rows and
``compress_ratio`` 1 (DSv3.2) / 4 (DSv4) handled identically.

Algorithm (replaces the secant P2 / collect P3 / snap P4 pipeline):

  P1   gather the K prev-step values once; min/max/mean stats AND an smem
       stash of the gathered values (``phase1_stats_stash``).
  P1b  rank-quantile seeding: 256-bin in-smem histogram of the stashed
       values + parallel suffix scan places M=4 ladder thresholds at order
       statistics of the valid count (``qfracs`` of K_valid). Column 0 =
       min gathered value (a guaranteed count>=K_valid anchor).
       Distribution-free: no offline tables, N drops out of placement.
  P2   ONE fused M-threshold ladder scan (``block_count_collect_multi``):
       M predicated counts + speculative per-thread slot-collect of every
       v >= thr[1] during the same pass (``fuse_collect``; slot overflow
       falls back to a classic collect). 16-bit inputs use native
       ``set.ge.{bf16x2,f16x2}`` packed compares with thresholds quantized
       to the dtype grid at P1b emit (bit-equivalent to the fp32 ladder).
  P3   sandwich: the tightest column pair (thr1, thr0) with
       count(thr1) >= K > count(thr0) splits the row into direct winners
       (v >= thr0, streamed straight to the output row) and a band
       [thr1, thr0) of <= kC candidates in smem.
  P4   exact band refine (``phase4_band_rank_scatter``): coarse histogram
       -> straddling bin b* + rank_above -> fast paths (A) whole-bin
       equality emit, (B) <= 32 members: warp0 exact register ranking,
       (C) fallback = exact value-edge band snap. A fixed-depth
       sub-histogram is NOT a valid path C — a fine bin is a value
       interval, not a tie group (upstream revert ec04147502); only paths
       that terminate on a data value are exact.

``GvrMsClusterKernel`` scales the same pipeline out to ``cluster_size``
CTAs per row (row-chunked slices, replicated P1/P1b, DSMEM count merge,
distributed P3 direct-write at rank prefixes, band remote-store push into
the leader's smem via ``st.shared::cluster``, leader-only P4).

Exactness authority is unchanged from the GVR kernel: thresholds are only
seeds; counts + the exact band refine decide membership. All-invalid
``pre_idx`` rows emit identity indices (inherited GVR contract). Fail-soft
under distribution shift: no sandwich pair -> classic collect path; band
overflow -> bounded-bisection retry-shrink landing count in [K, kC].

Provenance: assembled from the op21 kernel campaign (TensorRT-LLM perf
bench, iter11), which measured nsys pure-kernel cold-L2 geomean 1.14x
(fp32) / 1.29x (bf16) / 1.27x (fp16) vs this file's sibling GVR kernel on
the B200 P0 grid (K in {512,1024,2048}, N 65K-262K, BS 1-16), HW-invariant
on B300. Kernel code is byte-identical to the validated bench composition
up to the removal of const_expr-dead bench-only branches.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator

from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait
from .block_scan import warp_scan
from .gvr_topk_decode import (
    GvrParams,
    _fmin_f32_inline,
    float_as_uint32,
    ld_shared_cluster_f32,
    ld_shared_cluster_i32,
    mapa_shared_cluster,
)

_INT_MAX = 0x7FFFFFFF

'''

# ===========================================================================
# GvrMsKernel class header + merged __init__ (hand-written)
# ===========================================================================
MS_CLASS_HEADER = '''\

class GvrMsKernel:
    """Single-CTA-per-row GVR-MS kernel (one CTA processes one row).

    Constructor knobs (production defaults; each was an A/B lever in the
    op21 campaign — flip only for benchmarking):

    * ``M_thr`` / ``R_rounds``: ladder width / rounds. Production = (4, 1):
      one fused M=4 round; a second round only refines when the first finds
      no acceptable sandwich pair.
    * ``band_accept``: stop refining once band <= this (only relevant for
      ``R_rounds > 1``).
    * ``qfracs``: P1b rank fractions for ladder columns 1..M-1 (descending
      rank => ascending value). Column 1 doubles as the fused-collect
      predicate column.
    * ``qbins``: P1b quantile-histogram bins (<= num_threads).
    * ``fuse_collect``: collect candidates during the ladder scan. Needs
      one-CTA-wave residency headroom for the slot smem; host rule:
      ``batch_rows <= num_sms and 4 * top_k <= kC``.
    * ``p4_rank_scatter``: exact rank-scatter band refine (False = legacy
      runtime-k histogram snap, the A/B reference).
    * ``p4_smallbin``: P4 small-bin fast paths A/B (False = always snap).
    * ``p2_native``: native 16-bit ladder compares (no-op for fp32).
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        top_k: int,
        next_n: int = 1,
        num_threads: int = 512,
        compress_ratio: int = 1,
        use_256bit_load: bool = False,
        min_blocks_per_mp: int = 1,
        enable_unroll_4: bool = True,
        enable_phase3_unroll: bool = True,
        use_constant_hint: bool = False,
        enable_warp_parallel_reduce: Optional[bool] = None,
        return_output_values: bool = False,
        M_thr: int = 4,
        R_rounds: int = 1,
        band_accept: int = 64,
        mt_unroll: int = 4,
        qfracs: tuple = (0.75, 0.5, 0.25),
        qbins: int = 256,
        fuse_collect: bool = False,
        p4_rank_scatter: bool = True,
        p4_smallbin: bool = True,
        p2_native: bool = True,
    ):
        self.dtype = dtype
        self.top_k = top_k
        self.next_n = next_n
        # KV compression ratio of the indexer feeding this kernel:
        #   1 -> DSv3.2; preIdxOffset = (row % next_n) + 1 ("newest token
        #       appended" + MTP windowing).
        #   4 -> DSv4 (overlap compressor); compressed-token-index space,
        #       preIdxOffset = 0.
        assert compress_ratio in (1, 4), (
            f"compress_ratio must be 1 (V3.2) or 4 (V4); got {compress_ratio}")
        self.compress_ratio = compress_ratio

        self.WARP_SIZE = 32
        self.num_threads = num_threads
        self.num_warps = num_threads // self.WARP_SIZE
        self.min_blocks_per_mp = min_blocks_per_mp
        # Vector load width for the scan loops: 128-bit default, 256-bit
        # (LDG.E.256) needs 32-byte-aligned rows — the runner checks.
        self.use_256bit_load = use_256bit_load
        self.vec_bits = 256 if use_256bit_load else 128
        self.vec_align_bytes = self.vec_bits // 8
        self.enable_unroll_4 = enable_unroll_4
        self.enable_phase3_unroll = enable_phase3_unroll
        self.use_constant_hint = use_constant_hint
        # Warp-parallel reduce replaces tid==0 serial loops; pays only at
        # 32 warps (same policy as the GVR kernel).
        if enable_warp_parallel_reduce is None:
            enable_warp_parallel_reduce = num_threads == 1024
        self.enable_warp_parallel_reduce = enable_warp_parallel_reduce
        # The sandwich P3 defer-writes indices only (values are never
        # staged); the DSA indexer consumes indices only.
        assert not return_output_values, "GvrMsKernel is indices-only"
        self.return_output_values = return_output_values

        if dtype == cutlass.Float32:
            self._dtype_name = "float32"
        elif dtype == cutlass.BFloat16:
            self._dtype_name = "bfloat16"
        elif dtype == cutlass.Float16:
            self._dtype_name = "float16"
        else:
            raise ValueError(f"Unsupported dtype for GvrMsKernel: {dtype}")

        params = GvrParams.get(self._dtype_name, top_k, self.compress_ratio)
        self.kC = params.kC
        self.kNumBins = params.kNumBins

        self.FLT_MAX = 3.4028235e38
        self.NEG_FLT_MAX = -self.FLT_MAX

        # ---- ladder (op18 lineage) ----
        self.M_thr = int(M_thr)
        self.R_rounds = int(R_rounds)
        self.mt_unroll = int(mt_unroll)  # LSU-ILP unroll of the M-ary scan
        # ---- sandwich / rank-quantile (op21 lineage) ----
        self.band_accept = int(band_accept)
        # Native 16-bit ladder compares: thresholds are quantized to the
        # dtype grid at P1b emit (thr_q = f32(dtype(thr))), which makes
        # 16-bit-domain compares bit-equivalent to the fp32 compares every
        # other phase performs on the exactly-embedded values. The M-column
        # counts accumulate in packed 16x2 lanes (set.ge + add.rn), flushed
        # to int32 every 16 vec iters (per-half growth <= 8/iter => <= 128
        # << the 256 bf16 integer grid). The collect column uses a packed
        # mask (set.ge.u32) so the slot cursor stays exact per element.
        # fp32 binaries are untouched (const_expr).
        self.p2_native = bool(p2_native)
        self.qbins = int(qbins)
        assert self.qbins in (64, 128, 256) and self.num_threads >= self.qbins
        self.QBINS = self.qbins
        # Exact rank-scatter band refine vs legacy runtime-k snap (A/B).
        self.p4_rank_scatter = bool(p4_rank_scatter)
        # P4 small-bin fast paths (host probe on synth+real rows: cnt(b*)
        # p50=2 max=4, so path B covers ~100%); path C fallback is the
        # EXACT value-edge band snap.
        self.p4_smallbin = bool(p4_smallbin)
        # Rank fractions for the P1b quantile placement, mapped to ladder
        # columns 1..M-1 (descending rank => ascending value). Column 1
        # (qfracs[0]) doubles as the fused-collect column (pred_col).
        self.qfracs = tuple(float(f) for f in qfracs)
        assert len(self.qfracs) == self.M_thr - 1, "qfracs must be M-1 long"
        # P1 stashes the K gathered prev-step values into the (still
        # unused) P3 candidate buffer so P1b histograms from smem instead
        # of re-gathering K L2 loads per row.
        assert self.kC >= self.top_k, "P1 stash needs kC >= top_k"
        # Fused P2+P3 slot-collect during the ladder scan.
        self.fuse_collect = bool(fuse_collect)
        self.pred_col = 1 if self.M_thr >= 3 else 0
        # slot_cap needs headroom over the per-thread mean cand/threads
        # (overflow falls back to the classic collect); floor at 8.
        self.slot_cap = max(8, self.kC // self.num_threads)
'''

MSC_CLASS_HEADER = '''\

class GvrMsClusterKernel(GvrMsKernel):
    """Row-chunked ``cluster_size``-CTA cluster around the MS pipeline.

    C CTAs cooperate on ONE row (aggregate L2 bandwidth for the
    small-batch / large-N regime):

    - P1 stash + P1b rank-quantile seeding run REPLICATED on every CTA
      (same inputs -> bit-identical thresholds; zero cross-CTA traffic).
    - ONE fused ladder pass per CTA over its 64-elt-aligned slice (slot
      indices stored GLOBAL via the slice base offset).
    - DSMEM count merge (M ints per CTA, one cluster barrier) -> every CTA
      picks the same sandwich pair.
    - P3 distributed: per-CTA direct-write of >= thr0 winners straight to
      the output row at a rank-prefix offset; band entries are pushed
      straight into the LEADER's smem at the pre-known global band prefix
      via ``st.shared::cluster`` (``p3_push``; False restores the leader
      DSMEM gather, the A/B reference).
    - Leader (rank 0) runs the unchanged exact P4 for the last K-m0 slots.
    - Fallback (no pair / band > kC / slot overflow): leader re-runs the
      exact classic collect over the FULL row; peers idle. Rare and exact.

    Requires ``R_rounds == 1`` and ``fuse_collect=True`` (all thresholds
    known before the scan).
    """

    def __init__(self, *a, cluster_size: int = 4, p3_push: bool = True, **kw):
        kw.setdefault("fuse_collect", True)
        super().__init__(*a, **kw)
        self.C_cta = int(cluster_size)
        # P3 band remote-store push: during the slot walk each CTA writes
        # its band entries into the LEADER's smem at its global band prefix
        # (known pre-walk from the ladder counts) via st.shared::cluster —
        # deletes the leader DSMEM gather pass and one cluster barrier
        # pair.
        self.p3_push = bool(p3_push)
        assert self.R_rounds == 1 and self.fuse_collect
        assert self.C_cta >= 2, "use GvrMsKernel for cluster_size == 1"
'''


def main():
    out = []
    out.append(HEADER)

    # ---- 16-bit PTX helpers (op21 iter9) ----
    out.append(cut("ms", 68, 172, "iter9 native 16-bit compare primitives"))
    # ---- st.shared::cluster helpers (op21 iter7) ----
    out.append("\n")
    out.append(cut("msc", 60, 104, "iter7 DSMEM remote-store primitives"))

    # ================= GvrMsKernel =================
    out.append(MS_CLASS_HEADER)

    # base: _make_load_copy_atom .. warp reduces (incl. banner comments)
    out.append("\n")
    out.append(cut("base", 242, 300, "def _make_load_copy_atom"))
    # base: block_count_ge (banner + method)
    out.append("\n")
    seg = cut("base", 479, 615, "def block_count_ge")
    out.append(seg)
    # base: phase3_collect_candidates -> renamed stream-write worker
    out.append("\n")
    seg = cut("base", 757, 980, "def phase3_collect_candidates")
    seg = edit(
        seg,
        "def phase3_collect_candidates(",
        "def phase3_collect_stream_write(",
        tag="rename base phase3")
    seg = edit(
        seg,
        "    # Phase 3: Ballot-free candidate collect\n",
        "    # Phase 3 worker: ballot-free candidate collect (stream-write).\n"
        "    # Invoked by phase3_collect_candidates (the bounded-bisection\n"
        "    # override below) with the threshold already landed (done=1).\n",
        tag="base phase3 banner")
    out.append(seg)
    # base: block_fused_snap_iter + phase4_histogram_snap
    out.append("\n")
    out.append(cut("base", 981, 1100, "def block_fused_snap_iter"))
    out.append("\n")
    out.append(cut("base", 1101, 1409, "def phase4_histogram_snap"))

    # op18: block_count_ge_multi (banner + method)
    out.append("\n")
    out.append(cut("mt", 66, 161, "def block_count_ge_multi"))

    # op21 single-CTA methods
    out.append("\n")
    out.append(cut("ms", 256, 272, "def _p2n"))
    out.append("\n")
    out.append(cut("ms", 274, 429, "def block_count_collect_multi"))
    # phase3_from_slots .. phase1b_rank_quantile (skip the smem-row ladder
    # variant at 431-480; production smem_row is dropped)
    out.append("\n")
    seg = cut("ms", 482, 1500, "def phase3_from_slots")
    seg = edit(
        seg,
        "        GvrMultiThreshKernel.phase3_collect_candidates(\n"
        "            self, input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_wcnt,\n"
        "            s_thr, s_iscalars, tidx, warp_id, lane)",
        "        self.phase3_collect_stream_write(\n"
        "            input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_wcnt,\n"
        "            s_thr, s_iscalars, tidx, warp_id, lane)",
        tag="delegate to stream_write")
    # comment references the old vendored-super delegation; fix wording
    seg = edit(
        seg,
        "marks\n    # done=1, then delegates to the vendored prefix-sum + stream-write\n",
        "marks\n    # done=1, then delegates to the stream-write worker\n",
        tag="phase3 override comment")
    # phase1b banner references the dropped place_mode knob; fix wording
    seg = edit(
        seg,
        "    # op21 place_mode=5: rank-quantile round-0 placement. Histogram the K\n",
        "    # op21 P1b: rank-quantile round-0 placement. Histogram the K\n",
        tag="phase1b banner place_mode")
    # phase1 banner references the dropped base stats-only variant
    seg = edit(
        seg,
        "    # the min/max/mean stats exactly like the base phase1_preidx_stats AND\n",
        "    # the min/max/mean stats of the valid gathered values AND\n",
        tag="phase1 banner preidx_stats")
    out.append(seg)

    # op21 single-CTA kernel body, slimmed to the production (mode-5) config
    out.append("\n")
    seg = cut("ms", 1502, 1821, "def gvr_topk_kernel")
    # (a) drop the smem-row buffer allocation
    seg = edit(
        seg,
        "        # op20 iter6: smem-resident row buffer (small N)\n"
        "        smem_rowbuf = None\n"
        "        if cutlass.const_expr(self.smem_row_elems > 0):\n"
        "            smem_rowbuf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((self.smem_row_elems,), order=(0,)), byte_alignment=128)\n",
        "",
        tag="drop smem_rowbuf alloc")
    # (b) drop the smem-row bulk load + non-mode5 phase1
    seg = edit(
        seg,
        "            row_in_smem = cutlass.Int32(0)\n"
        "            if cutlass.const_expr(self.smem_row_elems > 0):\n"
        "                if N <= cutlass.Int32(self.smem_row_elems):\n"
        "                    row_in_smem = cutlass.Int32(1)\n"
        "                    ib = cutlass.Int32(tidx)\n"
        "                    while ib < N:\n"
        "                        smem_rowbuf[ib] = self._load_fp32(input_row, ib)\n"
        "                        ib = ib + cutlass.Int32(num_threads)\n"
        "            # phase1's internal barrier also publishes the bulk-load stores\n"
        "            if cutlass.const_expr(self.place_mode == 5):\n"
        "                # gathers once: stats + smem stash (smem_keys free until P3)\n"
        "                self.phase1_stats_stash(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,\n"
        "                                        smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)\n"
        "            else:\n"
        "                self.phase1_preidx_stats(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,\n"
        "                                         smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)\n",
        "            # P1 gathers once: stats + smem stash (smem_keys is free\n"
        "            # until P3)\n"
        "            self.phase1_stats_stash(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,\n"
        "                                    smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)\n",
        tag="drop smem-row load + non-mode5 phase1")
    # (c) P1b call is unconditional; pmean only fed the dropped mode-2 path
    seg = edit(
        seg,
        "                # ---- P2: adaptive M-ary ladder + sandwich pair tracking ----\n"
        "                pmean = s_thr[0]\n"
        "                if cutlass.const_expr(self.place_mode == 5):\n"
        "                    self.phase1b_rank_quantile(smem_keys, pre_idx_count,\n"
        "                                               smem_hist, s_thr, s_mt_thr,\n"
        "                                               s_mt_cnt, tidx)\n",
        "                # ---- P2: adaptive M-ary ladder + sandwich pair tracking ----\n"
        "                self.phase1b_rank_quantile(smem_keys, pre_idx_count,\n"
        "                                           smem_hist, s_thr, s_mt_thr,\n"
        "                                           s_mt_cnt, tidx)\n",
        tag="unconditional phase1b")
    # (d) round-0 placement: rank-quantile columns are already placed by
    # P1b; drop the offline-table place modes, keep the refine rounds.
    seg = edit(
        seg,
        "                        if rr == cutlass.Int32(0):\n"
        "                            if cutlass.const_expr(self.place_mode == 5):\n"
        "                                pass  # op21: s_mt_thr already placed by phase1b\n"
        "                            elif cutlass.const_expr(self.place_mode == 3):\n"
        "                                for m in cutlass.range_constexpr(M):\n"
        "                                    s_mt_thr[m] = lo + d * cutlass.Float32(self.fracs[m])\n"
        "                            elif cutlass.const_expr(self.place_mode == 0):\n"
        "                                for m in cutlass.range_constexpr(M):\n"
        "                                    s_mt_thr[m] = lo + d * (cutlass.Float32(m) / cutlass.Float32(M))\n"
        "                            elif cutlass.const_expr(self.place_mode == 1):\n"
        "                                s_mt_thr[0] = lo\n"
        "                                for m in cutlass.range_constexpr(M - 1):\n"
        "                                    s_mt_thr[m + 1] = lo + d * cutlass.Float32(1.0 / (1 << (M - 1 - m)))\n"
        "                            else:\n"
        "                                pm = pmean\n"
        "                                if pm <= lo or pm >= hi:\n"
        "                                    pm = (lo + hi) * cutlass.Float32(0.5)\n"
        "                                half = cutlass.const_expr(M // 2)\n"
        "                                for m in cutlass.range_constexpr(half):\n"
        "                                    s_mt_thr[m] = lo + (pm - lo) * (cutlass.Float32(m) / cutlass.Float32(half))\n"
        "                                for m in cutlass.range_constexpr(M - half):\n"
        "                                    s_mt_thr[half + m] = pm + (hi - pm) * (cutlass.Float32(m) / cutlass.Float32(M - half))\n"
        "                        else:\n",
        "                        if rr == cutlass.Int32(0):\n"
        "                            pass  # round-0 columns placed by phase1b\n"
        "                        else:\n",
        tag="drop non-mode5 placement")
    # (e) fused ladder: drop the smem-row branch
    seg = edit(
        seg,
        "                    if cutlass.const_expr(self.fuse_collect):\n"
        "                        # R==1 gated: all thresholds known up-front, collect at\n"
        "                        # the l1 column during the same pass (op20 iter4)\n"
        "                        if cutlass.const_expr(self.smem_row_elems > 0):\n"
        "                            if row_in_smem == cutlass.Int32(1):\n"
        "                                self.block_count_collect_multi_smem(\n"
        "                                    smem_rowbuf, N, s_mt_thr, smem_ptcnt_multi,\n"
        "                                    smem_wcnt_multi, s_mt_cnt, smem_slotk,\n"
        "                                    smem_slotv, tidx, warp_id, lane)\n"
        "                            else:\n"
        "                                self.block_count_collect_multi(\n"
        "                                    input_row, N, s_mt_thr, smem_ptcnt_multi,\n"
        "                                    smem_wcnt_multi, s_mt_cnt, smem_slotk,\n"
        "                                    smem_slotv, tidx, warp_id, lane)\n"
        "                        else:\n"
        "                            self.block_count_collect_multi(\n"
        "                                input_row, N, s_mt_thr, smem_ptcnt_multi,\n"
        "                                smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv,\n"
        "                                tidx, warp_id, lane)\n",
        "                    if cutlass.const_expr(self.fuse_collect):\n"
        "                        # R==1 gated: all thresholds known up-front, collect at\n"
        "                        # the pred_col column during the same pass\n"
        "                        self.block_count_collect_multi(\n"
        "                            input_row, N, s_mt_thr, smem_ptcnt_multi,\n"
        "                            smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv,\n"
        "                            tidx, warp_id, lane)\n",
        tag="drop smem-row ladder branch")
    out.append(seg)

    # op18 __call__ (single-CTA launch)
    out.append("\n")
    out.append(cut("mt", 354, 360, "def __call__"))

    # ================= GvrMsClusterKernel =================
    out.append(MSC_CLASS_HEADER)
    out.append("\n")
    out.append(cut("msc", 345, 503, "def block_count_collect_multi_base"))
    out.append("\n")
    out.append(cut("msc", 505, 597, "def phase3_from_slots_mc"))
    out.append("\n")
    out.append(cut("msc", 798, 825, "def _p3_leader_band_gather"))

    # cluster kernel body, slimmed: drop dist_p1 / dist_p4
    out.append("\n")
    seg = cut("msc", 827, 1142, "def gvr_topk_kernel")
    # drop the s_p1f allocation (dist_p1-only scratch)
    seg = edit(
        seg,
        "        # iter3 dist-P1 local stats publish (min/max/sum) for the DSMEM merge\n"
        "        s_p1f = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)\n",
        "",
        tag="drop s_p1f alloc")
    # drop dist_p1 phase1 branch
    seg = edit(
        seg,
        "            # ---- P1: distributed (K/C gather + DSMEM stats merge) or the\n"
        "            # iter2 replicated reference ----\n"
        "            if cutlass.const_expr(self.dist_p1):\n"
        "                self.phase1_dist_stats(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,\n"
        "                                       rank, smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1,\n"
        "                                       s_p1f, s_cluster, s_thr, s_iscalars, tidx, warp_id, lane)\n"
        "            else:\n"
        "                self.phase1_stats_stash(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,\n"
        "                                        smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)\n",
        "            # ---- P1: replicated stash + stats (every CTA gathers the\n"
        "            # same K addresses; after the first CTA misses, the rest\n"
        "            # hit L2 — replication beats a distributed gather + DSMEM\n"
        "            # merges) ----\n"
        "            self.phase1_stats_stash(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,\n"
        "                                    smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)\n",
        tag="drop dist_p1 phase1")
    # drop dist_p1 phase1b branch
    seg = edit(
        seg,
        "                if cutlass.const_expr(self.dist_p1):\n"
        "                    self.phase1b_dist(smem_keys, smem_hist, s_thr, s_mt_thr,\n"
        "                                      tidx)\n"
        "                else:\n"
        "                    self.phase1b_rank_quantile(smem_keys, pre_idx_count,\n"
        "                                               smem_hist, s_thr, s_mt_thr,\n"
        "                                               s_mt_cnt, tidx)\n",
        "                self.phase1b_rank_quantile(smem_keys, pre_idx_count,\n"
        "                                           smem_hist, s_thr, s_mt_thr,\n"
        "                                           s_mt_cnt, tidx)\n",
        tag="drop dist_p1 phase1b")
    # drop the dist_p4 branch; keep the leader-P4 else-body (dedent by 4)
    dist_p4_block = (
        "                    if cutlass.const_expr(self.dist_p4):\n"
        "                        # ---- iter4: distributed P4 (bulk emitted by every\n"
        "                        # CTA; only the boundary bin goes to the leader) ----\n"
        "                        k_rem4 = cutlass.Int32(top_k) - m0g\n"
        "                        self.phase4_dist(rank, m0g, k_rem4, smem_keys,\n"
        "                                         smem_vals, smem_hist, smem_ptcnt_up,\n"
        "                                         smem_slotk, smem_slotv, smem_wcnt,\n"
        "                                         s_cluster, s_thr, s_swf, s_iscalars,\n"
        "                                         output_indices_row, tidx, warp_id,\n"
        "                                         lane)\n"
        "                    else:\n")
    assert seg.count(dist_p4_block) == 1, "dist_p4 block anchor not found"
    pre, post = seg.split(dist_p4_block)
    # dedent the else-body (everything up to the matching `else:` of the
    # `if ok == 1:` — i.e. lines up to and NOT including the line
    # "                else:\n" that starts the fallback branch)
    body_end_anchor = "                else:\n                    # ---- fallback: leader-only exact classic path, full row.\n"
    assert post.count(body_end_anchor) == 1
    body, rest = post.split(body_end_anchor)
    dedented = []
    for ln in body.splitlines(keepends=True):
        if ln.strip() == "":
            dedented.append(ln)
        else:
            assert ln.startswith("    " * 6), f"unexpected indent: {ln!r}"
            dedented.append(ln[4:])
    seg = pre + "".join(dedented) + body_end_anchor + rest
    out.append(seg)

    # cluster __call__
    out.append("\n")
    out.append(cut("msc", 1144, 1151, "def __call__"))

    text = "".join(out)
    # final sanity: no references to dropped attributes / names remain
    for banned in ("smem_row_elems", "place_mode", "dist_p1", "dist_p4",
                   "self.fracs", "c_accept", "kFTarget", "phase2_secant",
                   "s_p1f", "GvrMultiThreshKernel", "GvrSandwichKernel",
                   "phase1_preidx_stats", "block_count_collect_multi_smem",
                   "phase1_dist_stats", "phase1b_dist", "phase4_dist",
                   "import os", "os.environ", "smem_rowbuf", "row_in_smem",
                   "pmean = s_thr[0]"):
        assert banned not in text, f"banned symbol survived: {banned}"
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "gvr_topk_decode_ms.py"
    dst.write_text(text)
    import ast as _ast
    _ast.parse(text)
    print(f"wrote {dst}: {len(text.splitlines())} lines, AST-clean")


if __name__ == "__main__":
    main()
