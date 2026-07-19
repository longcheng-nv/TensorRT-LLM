# [p4f1] v4 splice: start from the v2 backup (level-0 original text +
# insertions 1/2, hot scatter under `if need_more==0`, deep under else) and
# produce v4:
#   <original scatter+pad, unconditional, ZERO modifications>
#   if need_more==1:
#       if cnt_strad <= CAP: separate collect pass -> scratch + thread0 select
#       else: integer-bisection threshold select (deep recursion DELETED)
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "gvrpkgf1", "top_k", "gvr_topk_decode.py")
BAK = os.path.join(HERE, "gvr_topk_decode.py.v2fixed.bak")
shutil.copy(BAK, F)

src = open(F).read()
lines = src.split("\n")

IF_HOT = "                if need_more == cutlass.Int32(0):"
ELSE_L = "                else:"
ELIF_MARK = ("            # ---- EXACT: one fine-histogram recursion on the "
             "straddling bin b* ----")

i_if = lines.index(IF_HOT)
i_else = next(i for i in range(i_if + 1, len(lines)) if lines[i] == ELSE_L)
i_elif = next(i for i in range(i_else + 1, len(lines)) if lines[i] == ELIF_MARK)

hot = lines[i_if + 1 : i_else]           # original scatter + pad (indent 20)
hot_d = [l[4:] if l.strip() else l for l in hot]  # de-indent to 16

V4 = '''\
                # [p4f1] v4: straddle-tie resolution OUTSIDE the original
                # scatter (which ran verbatim above). Fires on ~0/25 real
                # bench rows; firing rows pay one bounded collect pass +
                # a thread-0 exact select. cnt_straddle > CAP (pathological)
                # falls back to an exact integer-bisection threshold select
                # (the v2/v3 deep recursion is DELETED for icache).
                if need_more == cutlass.Int32(1):
                    need_b = cutlass.Int32(kK) - rank_above_fine
                    if cnt_strad <= cutlass.Int32(CAP):
                        # ---- collect pass: straddle elements -> scratch ring
                        # smem_hist[4..259] as (value_bits, cand_idx) pairs.
                        # cb/sb recomputed with the IDENTICAL expressions.
                        if tidx == cutlass.Int32(0):
                            s_iscalars[0] = cutlass.Int32(0)  # collect counter
                        cute.arch.barrier()
                        ic4 = tidx
                        while ic4 < cand_count:
                            v = smem_keys[ic4]
                            bin_i = cutlass.Int32((v - bmin_r) * inv1)
                            if bin_i < cutlass.Int32(0):
                                bin_i = cutlass.Int32(0)
                            if bin_i > cutlass.Int32(kBins - 1):
                                bin_i = cutlass.Int32(kBins - 1)
                            if bin_i == b_star:
                                sb = cutlass.Int32((v - f_lo) * finv)
                                if sb < cutlass.Int32(0):
                                    sb = cutlass.Int32(0)
                                if sb > cutlass.Int32(fbins - 1):
                                    sb = cutlass.Int32(fbins - 1)
                                if sb == sb_star0:
                                    o = atomicAdd(s_iscalars.iterator + cutlass.Int32(0), cutlass.Int32(1))
                                    if o < cutlass.Int32(CAP):
                                        smem_hist[o + o + cutlass.Int32(4)] = float_as_uint32(v)
                                        smem_hist[o + o + cutlass.Int32(5)] = smem_vals[ic4]
                            ic4 = ic4 + cutlass.Int32(num_threads)
                        cute.arch.barrier()
                        # ---- thread-0 exact top-`need` select over <=128
                        # scratch entries; rewrites output positions
                        # rank_above_fine .. rank_above_fine+need-1.
                        if tidx == cutlass.Int32(0):
                            j2 = cutlass.Int32(0)
                            while j2 < need_b:
                                bestv = cutlass.Float32(self.NEG_FLT_MAX)
                                besti = cutlass.Int32(0)
                                i2 = cutlass.Int32(0)
                                while i2 < cnt_strad:
                                    vbits = smem_hist[i2 + i2 + cutlass.Int32(4)]
                                    vv = cutlass.Float32(
                                        llvm.bitcast(
                                            cutlass.Float32.mlir_type,
                                            vbits.ir_value(),
                                        )
                                    )
                                    if vv > bestv:
                                        bestv = vv
                                        besti = i2
                                    i2 = i2 + cutlass.Int32(1)
                                pos = rank_above_fine + j2
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(bestv)
                                output_indices_row[pos] = smem_hist[besti + besti + cutlass.Int32(5)]
                                # mark consumed (NEG_FLT_MAX never wins again)
                                smem_hist[besti + besti + cutlass.Int32(4)] = float_as_uint32(
                                    cutlass.Float32(self.NEG_FLT_MAX)
                                )
                                j2 = j2 + cutlass.Int32(1)
                    else:
                        # ---- pathological (cnt_straddle > CAP): exact
                        # threshold select by INTEGER BISECTION on the order-
                        # preserving key skey(b) = b ^ ((b>>31) & 0x7FFFFFFF)
                        # (signed-int compare == fp32 value order; equal keys
                        # are bit-identical values, interchangeable for value-
                        # set exactness). NOTE: an arbitrary-CAP-subset
                        # chunked collect is NOT exact when cnt > CAP + need
                        # (no safe picks per round), hence bisection.
                        # smem: [260]=lo key, [261]=hi key, [262]=done,
                        # [263]=g(hi); invariant g(lo) > need >= g(hi).
                        if tidx == cutlass.Int32(0):
                            bmin_i = cutlass.Int32(
                                llvm.bitcast(cutlass.Int32.mlir_type, bmin_r.ir_value())
                            )
                            bmax_i = cutlass.Int32(
                                llvm.bitcast(cutlass.Int32.mlir_type, bmax_r.ir_value())
                            )
                            skl = bmin_i ^ ((bmin_i >> cutlass.Int32(31)) & cutlass.Int32(0x7FFFFFFF))
                            skh = bmax_i ^ ((bmax_i >> cutlass.Int32(31)) & cutlass.Int32(0x7FFFFFFF))
                            smem_hist[260] = skl - cutlass.Int32(1)
                            smem_hist[261] = skh + cutlass.Int32(1)
                            smem_hist[262] = cutlass.Int32(0)
                            smem_hist[263] = cutlass.Int32(0)
                        cute.arch.barrier()
                        itb = cutlass.Int32(0)
                        while itb < cutlass.Int32(40) and smem_hist[262] == cutlass.Int32(0):
                            lo_k = smem_hist[260]
                            hi_k = smem_hist[261]
                            # overflow-safe floor average of signed ints
                            mid_k = (lo_k & hi_k) + ((lo_k ^ hi_k) >> cutlass.Int32(1))
                            if tidx == cutlass.Int32(0):
                                s_iscalars[0] = cutlass.Int32(0)
                            cute.arch.barrier()
                            if mid_k == lo_k:  # interval collapsed (uniform)
                                if tidx == cutlass.Int32(0):
                                    smem_hist[262] = cutlass.Int32(1)
                            else:
                                icb = tidx
                                while icb < cand_count:
                                    v = smem_keys[icb]
                                    bin_i = cutlass.Int32((v - bmin_r) * inv1)
                                    if bin_i < cutlass.Int32(0):
                                        bin_i = cutlass.Int32(0)
                                    if bin_i > cutlass.Int32(kBins - 1):
                                        bin_i = cutlass.Int32(kBins - 1)
                                    if bin_i == b_star:
                                        sb = cutlass.Int32((v - f_lo) * finv)
                                        if sb < cutlass.Int32(0):
                                            sb = cutlass.Int32(0)
                                        if sb > cutlass.Int32(fbins - 1):
                                            sb = cutlass.Int32(fbins - 1)
                                        if sb == sb_star0:
                                            vi = cutlass.Int32(
                                                llvm.bitcast(cutlass.Int32.mlir_type, v.ir_value())
                                            )
                                            sk = vi ^ ((vi >> cutlass.Int32(31)) & cutlass.Int32(0x7FFFFFFF))
                                            if sk > mid_k:
                                                atomicAdd(s_iscalars.iterator + cutlass.Int32(0), cutlass.Int32(1))
                                    icb = icb + cutlass.Int32(num_threads)
                                cute.arch.barrier()
                                if tidx == cutlass.Int32(0):
                                    gmid = s_iscalars[0]
                                    if gmid > need_b:
                                        smem_hist[260] = mid_k
                                    else:
                                        smem_hist[261] = mid_k
                                        smem_hist[263] = gmid
                            cute.arch.barrier()
                            itb = itb + cutlass.Int32(1)
                        # final write pass: skey > T -> top block; skey == T
                        # ties fill the remaining need - g(T) positions.
                        t_k = smem_hist[261]
                        gt_k = smem_hist[263]
                        if tidx == cutlass.Int32(0):
                            s_iscalars[0] = cutlass.Int32(0)  # > T counter
                            s_iscalars[1] = cutlass.Int32(0)  # tie counter
                        cute.arch.barrier()
                        icf = tidx
                        while icf < cand_count:
                            v = smem_keys[icf]
                            bin_i = cutlass.Int32((v - bmin_r) * inv1)
                            if bin_i < cutlass.Int32(0):
                                bin_i = cutlass.Int32(0)
                            if bin_i > cutlass.Int32(kBins - 1):
                                bin_i = cutlass.Int32(kBins - 1)
                            if bin_i == b_star:
                                sb = cutlass.Int32((v - f_lo) * finv)
                                if sb < cutlass.Int32(0):
                                    sb = cutlass.Int32(0)
                                if sb > cutlass.Int32(fbins - 1):
                                    sb = cutlass.Int32(fbins - 1)
                                if sb == sb_star0:
                                    vi = cutlass.Int32(
                                        llvm.bitcast(cutlass.Int32.mlir_type, v.ir_value())
                                    )
                                    sk = vi ^ ((vi >> cutlass.Int32(31)) & cutlass.Int32(0x7FFFFFFF))
                                    if sk > t_k:
                                        o = atomicAdd(s_iscalars.iterator + cutlass.Int32(0), cutlass.Int32(1))
                                        pos = rank_above_fine + o
                                        if cutlass.const_expr(self.return_output_values):
                                            output_values_row[pos] = self.dtype(v)
                                        output_indices_row[pos] = smem_vals[icf]
                                    elif sk == t_k:
                                        o = atomicAdd(s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1))
                                        if o < need_b - gt_k:
                                            pos = rank_above_fine + gt_k + o
                                            if cutlass.const_expr(self.return_output_values):
                                                output_values_row[pos] = self.dtype(v)
                                            output_indices_row[pos] = smem_vals[icf]
                            icf = icf + cutlass.Int32(num_threads)'''

new_lines = lines[:i_if] + hot_d + V4.split("\n") + lines[i_elif:]
open(F, "w").write("\n".join(new_lines))
print(f"v4 spliced: hot={len(hot)} deep-deleted={i_elif - i_else - 1} lines, "
      f"v4-branch={len(V4.splitlines())} lines")
