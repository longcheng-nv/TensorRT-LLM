# [p4f1] splice the v2 ON-branch body into gvr_topk_decode.py
import io
import os

F = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "gvrpkgf1", "top_k", "gvr_topk_decode.py")

START = "            ):  # [p4f1]\n"
END = ("            # ---- EXACT: one fine-histogram recursion on the "
       "straddling bin b* ----\n")

BODY = '''\
                # [p4f1] v2: level 0 executes the ORIGINAL one-shot text
                # (two marked insertions only); ONE block-uniform runtime
                # need_more test picks between the ORIGINAL 3-class scatter
                # (hot path) and a deep iterative section (fine levels 1..3
                # + chain scatter) for genuine same-fine-bin straddles.
                fbins = cutlass.const_expr(256)
                fbpw = cutlass.const_expr(256 // self.num_warps)
                MAXL = 4  # [p4f1] fine levels 0..3 total
                HB = cutlass.const_expr(256)
                # [p4f1] deep-section SMEM metadata (above the 256 fine bins;
                # kNumBins >= 512 asserted in ctor):
                #   smem_hist[256+4k+0/1] = f_lo_k / finv_k bits
                #   smem_hist[256+4k+2/3] = sb_star_k / ra_k
                #   smem_hist[272+k]      = per-level mid scatter counters
                #   smem_hist[276]        = need_more (written ONCE, level 0)
                #   smem_hist[277]        = L (levels used)
                #   smem_hist[278]        = deep-loop done flag (NOT 276: a
                #     rewrite of 276 in the deep path could race late readers
                #     of the uniform need_more test and diverge barriers)
                # ---- level 0: ORIGINAL one-shot fine recursion ----
                f_lo = bmin_r + cutlass.Float32(b_star) / inv1
                finv = (cutlass.Float32(fbins - 1) + cutlass.Float32(0.99)) * inv1
                iz = tidx
                while iz < cutlass.Int32(fbins):
                    smem_hist[iz] = cutlass.Int32(0)
                    iz = iz + cutlass.Int32(num_threads)
                cute.arch.barrier()
                ifb = tidx
                while ifb < cand_count:
                    vf = smem_keys[ifb]
                    cb = cutlass.Int32((vf - bmin_r) * inv1)
                    if cb < cutlass.Int32(0):
                        cb = cutlass.Int32(0)
                    if cb > cutlass.Int32(kBins - 1):
                        cb = cutlass.Int32(kBins - 1)
                    if cb == b_star:
                        sb = cutlass.Int32((vf - f_lo) * finv)
                        if sb < cutlass.Int32(0):
                            sb = cutlass.Int32(0)
                        if sb > cutlass.Int32(fbins - 1):
                            sb = cutlass.Int32(fbins - 1)
                        atomicAdd(smem_hist.iterator + sb, cutlass.Int32(1))
                    ifb = ifb + cutlass.Int32(num_threads)
                cute.arch.barrier()
                fws = cutlass.Int32(0)
                for jbf in cutlass.range_constexpr(fbpw):
                    bif = (
                        cutlass.Int32(fbins - 1)
                        - warp_id * cutlass.Int32(fbpw)
                        - cutlass.Int32(jbf)
                    )
                    fws = fws + smem_hist[bif]
                if lane == cutlass.Int32(0):
                    smem_wcnt[warp_id] = fws
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    cumf = rank_above
                    twf = cutlass.Int32(num_warps - 1)
                    fnd = cutlass.Int32(0)
                    for w2 in cutlass.range_constexpr(self.num_warps):
                        cumf = cumf + smem_wcnt[w2]
                        if cumf >= cutlass.Int32(kK) and fnd == cutlass.Int32(0):
                            twf = cutlass.Int32(w2)
                            fnd = cutlass.Int32(1)
                    pre = rank_above
                    for w3 in cutlass.range_constexpr(self.num_warps):
                        if cutlass.Int32(w3) < twf:
                            pre = pre + smem_wcnt[w3]
                    s_iscalars[4] = pre  # prefix into target fine warp
                    s_iscalars[1] = twf  # target fine warp
                cute.arch.barrier()
                pre_f = s_iscalars[4]
                twf2 = s_iscalars[1]
                if warp_id == twf2 and lane == cutlass.Int32(0):
                    base_f = pre_f
                    sb_star = cutlass.Int32(fbins - 1)
                    ra_fine = base_f
                    sd = cutlass.Int32(0)
                    for jb3 in cutlass.range_constexpr(fbpw):
                        sbi = (
                            cutlass.Int32(fbins - 1)
                            - twf2 * cutlass.Int32(fbpw)
                            - cutlass.Int32(jb3)
                        )
                        ra_b = base_f
                        base_f = base_f + smem_hist[sbi]
                        if base_f >= cutlass.Int32(kK) and sd == cutlass.Int32(0):
                            sb_star = sbi
                            ra_fine = ra_b
                            sd = cutlass.Int32(1)
                    # [p4f1] insertion 1: stage the straddle-bin count BEFORE
                    # the hist[2]/[3] scratch writes below clobber bins 2/3
                    smem_hist[HB + 2] = smem_hist[sb_star]
                    smem_hist[2] = sb_star
                    smem_hist[3] = ra_fine
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                    s_iscalars[0] = cutlass.Int32(0)  # cnt_mid (b*, sub>sb*)
                    s_iscalars[1] = cutlass.Int32(0)  # cnt_strad (b*, sub==sb*)
                    # [p4f1] insertion 2: the ONE runtime deep-recursion test
                    # (result read block-uniformly after the barrier below)
                    ra_f0 = smem_hist[3]
                    cnt_str0 = smem_hist[HB + 2]
                    nm = cutlass.Int32(0)
                    if ra_f0 + cnt_str0 > cutlass.Int32(kK):
                        width0 = cutlass.Float32(1.0) / finv
                        af0 = cute.arch.fmax(f_lo, cutlass.Float32(0.0) - f_lo)
                        ulp0 = cute.arch.fmax(
                            af0, cutlass.Float32(1e-30)
                        ) * cutlass.Float32(1.1920928955078125e-07)
                        if width0 > ulp0:
                            nm = cutlass.Int32(1)
                    smem_hist[276] = nm
                cute.arch.barrier()
                sb_star = smem_hist[2]
                rank_above_fine = smem_hist[3]
                need_more = smem_hist[276]  # [p4f1] block-uniform
                if need_more == cutlass.Int32(0):
                    # ---- hot path: ORIGINAL 3-class scatter + pad ----
                    isc = tidx
                    while isc < cand_count:
                        v = smem_keys[isc]
                        bin_i = cutlass.Int32((v - bmin_r) * inv1)
                        if bin_i < cutlass.Int32(0):
                            bin_i = cutlass.Int32(0)
                        if bin_i > cutlass.Int32(kBins - 1):
                            bin_i = cutlass.Int32(kBins - 1)
                        if bin_i > b_star:
                            pos = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cutlass.Int32(1))
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        elif bin_i == b_star:
                            sb = cutlass.Int32((v - f_lo) * finv)
                            if sb < cutlass.Int32(0):
                                sb = cutlass.Int32(0)
                            if sb > cutlass.Int32(fbins - 1):
                                sb = cutlass.Int32(fbins - 1)
                            if sb > sb_star:
                                o = atomicAdd(s_iscalars.iterator + cutlass.Int32(0), cutlass.Int32(1))
                                pos = rank_above + o
                                if pos < cutlass.Int32(kK):
                                    if cutlass.const_expr(self.return_output_values):
                                        output_values_row[pos] = self.dtype(v)
                                    output_indices_row[pos] = smem_vals[isc]
                            elif sb == sb_star:
                                o = atomicAdd(s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1))
                                pos = rank_above_fine + o
                                if pos < cutlass.Int32(kK):
                                    if cutlass.const_expr(self.return_output_values):
                                        output_values_row[pos] = self.dtype(v)
                                    output_indices_row[pos] = smem_vals[isc]
                        isc = isc + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    cnt_strad = s_iscalars[1]
                    filled = rank_above_fine + cnt_strad
                    if filled > cutlass.Int32(kK):
                        filled = cutlass.Int32(kK)
                    ipad = filled + tidx
                    while ipad < cutlass.Int32(kK):
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                        output_indices_row[ipad] = cutlass.Int32(-1)
                        ipad = ipad + cutlass.Int32(num_threads)
                else:
                    # ---- [p4f1] deep section: publish level-0 state, run
                    # fine levels 1..MAXL-1, then the chain scatter ----
                    if tidx == cutlass.Int32(0):
                        smem_hist[HB + 0] = float_as_uint32(f_lo)
                        smem_hist[HB + 1] = float_as_uint32(finv)
                        smem_hist[HB + 2] = sb_star
                        smem_hist[HB + 3] = rank_above_fine
                        smem_hist[277] = cutlass.Int32(1)  # L so far (level 0)
                        smem_hist[278] = cutlass.Int32(0)  # deep done flag
                    cute.arch.barrier()
                    for lvl in cutlass.range_constexpr(1, MAXL):  # [p4f1]
                        done_l = smem_hist[278]
                        # block-uniform skip-body guard (read after barrier)
                        if done_l == cutlass.Int32(0):
                            f_lo_l = cutlass.Float32(
                                llvm.bitcast(
                                    cutlass.Float32.mlir_type,
                                    smem_hist[HB + 4 * lvl + 0].ir_value(),
                                )
                            )
                            finv_l = cutlass.Float32(
                                llvm.bitcast(
                                    cutlass.Float32.mlir_type,
                                    smem_hist[HB + 4 * lvl + 1].ir_value(),
                                )
                            )
                            # chain metadata of levels j < lvl
                            chain_flo = []
                            chain_finv = []
                            chain_sb = []
                            for j in cutlass.range_constexpr(lvl):
                                chain_flo.append(
                                    cutlass.Float32(
                                        llvm.bitcast(
                                            cutlass.Float32.mlir_type,
                                            smem_hist[HB + 4 * j + 0].ir_value(),
                                        )
                                    )
                                )
                                chain_finv.append(
                                    cutlass.Float32(
                                        llvm.bitcast(
                                            cutlass.Float32.mlir_type,
                                            smem_hist[HB + 4 * j + 1].ir_value(),
                                        )
                                    )
                                )
                                chain_sb.append(smem_hist[HB + 4 * j + 2])
                            seed_l = smem_hist[HB + 4 * (lvl - 1) + 3]
                            # zero the 256 fine bins + build hist over the CHAIN
                            iz = tidx
                            while iz < cutlass.Int32(fbins):
                                smem_hist[iz] = cutlass.Int32(0)
                                iz = iz + cutlass.Int32(num_threads)
                            cute.arch.barrier()
                            ifb = tidx
                            while ifb < cand_count:
                                vf = smem_keys[ifb]
                                cb = cutlass.Int32((vf - bmin_r) * inv1)
                                if cb < cutlass.Int32(0):
                                    cb = cutlass.Int32(0)
                                if cb > cutlass.Int32(kBins - 1):
                                    cb = cutlass.Int32(kBins - 1)
                                if cb == b_star:
                                    in_chain = cutlass.Int32(1)
                                    for j in cutlass.range_constexpr(lvl):
                                        sbj = cutlass.Int32(
                                            (vf - chain_flo[j]) * chain_finv[j]
                                        )
                                        if sbj < cutlass.Int32(0):
                                            sbj = cutlass.Int32(0)
                                        if sbj > cutlass.Int32(fbins - 1):
                                            sbj = cutlass.Int32(fbins - 1)
                                        if sbj != chain_sb[j]:
                                            in_chain = cutlass.Int32(0)
                                    if in_chain == cutlass.Int32(1):
                                        sb = cutlass.Int32((vf - f_lo_l) * finv_l)
                                        if sb < cutlass.Int32(0):
                                            sb = cutlass.Int32(0)
                                        if sb > cutlass.Int32(fbins - 1):
                                            sb = cutlass.Int32(fbins - 1)
                                        atomicAdd(smem_hist.iterator + sb, cutlass.Int32(1))
                                ifb = ifb + cutlass.Int32(num_threads)
                            cute.arch.barrier()
                            # fine 3-step search seeded at seed_l
                            fws = cutlass.Int32(0)
                            for jbf in cutlass.range_constexpr(fbpw):
                                bif = (
                                    cutlass.Int32(fbins - 1)
                                    - warp_id * cutlass.Int32(fbpw)
                                    - cutlass.Int32(jbf)
                                )
                                fws = fws + smem_hist[bif]
                            if lane == cutlass.Int32(0):
                                smem_wcnt[warp_id] = fws
                            cute.arch.barrier()
                            if tidx == cutlass.Int32(0):
                                cumf = seed_l
                                twf = cutlass.Int32(num_warps - 1)
                                fnd = cutlass.Int32(0)
                                for w2 in cutlass.range_constexpr(self.num_warps):
                                    cumf = cumf + smem_wcnt[w2]
                                    if cumf >= cutlass.Int32(kK) and fnd == cutlass.Int32(0):
                                        twf = cutlass.Int32(w2)
                                        fnd = cutlass.Int32(1)
                                pre = seed_l
                                for w3 in cutlass.range_constexpr(self.num_warps):
                                    if cutlass.Int32(w3) < twf:
                                        pre = pre + smem_wcnt[w3]
                                s_iscalars[4] = pre
                                s_iscalars[1] = twf
                            cute.arch.barrier()
                            pre_fl = s_iscalars[4]
                            twf2l = s_iscalars[1]
                            if warp_id == twf2l and lane == cutlass.Int32(0):
                                base_f = pre_fl
                                sb_star_l = cutlass.Int32(fbins - 1)
                                ra_l = base_f
                                sd = cutlass.Int32(0)
                                for jb3 in cutlass.range_constexpr(fbpw):
                                    sbi = (
                                        cutlass.Int32(fbins - 1)
                                        - twf2l * cutlass.Int32(fbpw)
                                        - cutlass.Int32(jb3)
                                    )
                                    ra_b = base_f
                                    base_f = base_f + smem_hist[sbi]
                                    if base_f >= cutlass.Int32(kK) and sd == cutlass.Int32(0):
                                        sb_star_l = sbi
                                        ra_l = ra_b
                                        sd = cutlass.Int32(1)
                                # publish ABOVE the fine bins (hist intact)
                                smem_hist[HB + 4 * lvl + 2] = sb_star_l
                                smem_hist[HB + 4 * lvl + 3] = ra_l
                            cute.arch.barrier()
                            # thread0: continue decision + publish next level
                            if tidx == cutlass.Int32(0):
                                sb_s = smem_hist[HB + 4 * lvl + 2]
                                ra_v = smem_hist[HB + 4 * lvl + 3]
                                cnt_str = smem_hist[sb_s]  # fine hist intact
                                cont = cutlass.Int32(0)
                                if cutlass.const_expr(lvl + 1 < MAXL):
                                    if ra_v + cnt_str > cutlass.Int32(kK):
                                        width = cutlass.Float32(1.0) / finv_l
                                        af = cute.arch.fmax(
                                            f_lo_l, cutlass.Float32(0.0) - f_lo_l
                                        )
                                        ulp_floor = cute.arch.fmax(
                                            af, cutlass.Float32(1e-30)
                                        ) * cutlass.Float32(1.1920928955078125e-07)
                                        if width > ulp_floor:
                                            cont = cutlass.Int32(1)
                                smem_hist[277] = cutlass.Int32(lvl + 1)  # L
                                if cont == cutlass.Int32(0):
                                    smem_hist[278] = cutlass.Int32(1)  # done
                                else:
                                    f_lo_n = f_lo_l + cutlass.Float32(sb_s) / finv_l
                                    finv_n = (
                                        cutlass.Float32(fbins - 1) + cutlass.Float32(0.99)
                                    ) * finv_l
                                    smem_hist[HB + 4 * (lvl + 1) + 0] = float_as_uint32(f_lo_n)
                                    smem_hist[HB + 4 * (lvl + 1) + 1] = float_as_uint32(finv_n)
                            cute.arch.barrier()
                    # ---- chain scatter over ALL candidates ----
                    L_used = smem_hist[277]
                    ra_last = smem_hist[HB + 4 * (L_used - cutlass.Int32(1)) + cutlass.Int32(3)]
                    # hoist per-level scatter params (entries beyond L_used are
                    # never used: guarded by k < L_used below)
                    sc_flo = []
                    sc_finv = []
                    sc_sb = []
                    sc_base = []
                    for k in cutlass.range_constexpr(MAXL):
                        sc_flo.append(
                            cutlass.Float32(
                                llvm.bitcast(
                                    cutlass.Float32.mlir_type,
                                    smem_hist[HB + 4 * k + 0].ir_value(),
                                )
                            )
                        )
                        sc_finv.append(
                            cutlass.Float32(
                                llvm.bitcast(
                                    cutlass.Float32.mlir_type,
                                    smem_hist[HB + 4 * k + 1].ir_value(),
                                )
                            )
                        )
                        sc_sb.append(smem_hist[HB + 4 * k + 2])
                        if cutlass.const_expr(k == 0):
                            sc_base.append(rank_above)
                        else:
                            sc_base.append(smem_hist[HB + 4 * (k - 1) + 3])
                    if tidx == cutlass.Int32(0):
                        s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                        s_iscalars[1] = cutlass.Int32(0)  # cnt_strad (final level)
                        for kk in cutlass.range_constexpr(MAXL):
                            smem_hist[272 + kk] = cutlass.Int32(0)  # mid counters
                    cute.arch.barrier()
                    isc = tidx
                    while isc < cand_count:
                        v = smem_keys[isc]
                        bin_i = cutlass.Int32((v - bmin_r) * inv1)
                        if bin_i < cutlass.Int32(0):
                            bin_i = cutlass.Int32(0)
                        if bin_i > cutlass.Int32(kBins - 1):
                            bin_i = cutlass.Int32(kBins - 1)
                        if bin_i > b_star:
                            pos = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cutlass.Int32(1))
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        elif bin_i == b_star:
                            placed = cutlass.Int32(0)
                            for k in cutlass.range_constexpr(MAXL):
                                if placed == cutlass.Int32(0) and cutlass.Int32(k) < L_used:
                                    sb = cutlass.Int32((v - sc_flo[k]) * sc_finv[k])
                                    if sb < cutlass.Int32(0):
                                        sb = cutlass.Int32(0)
                                    if sb > cutlass.Int32(fbins - 1):
                                        sb = cutlass.Int32(fbins - 1)
                                    if sb > sc_sb[k]:
                                        o = atomicAdd(
                                            smem_hist.iterator + cutlass.Int32(272 + k),
                                            cutlass.Int32(1),
                                        )
                                        pos = sc_base[k] + o
                                        if pos < cutlass.Int32(kK):
                                            if cutlass.const_expr(self.return_output_values):
                                                output_values_row[pos] = self.dtype(v)
                                            output_indices_row[pos] = smem_vals[isc]
                                        placed = cutlass.Int32(1)
                                    elif sb < sc_sb[k]:
                                        placed = cutlass.Int32(1)  # below: drop
                                    # sb == sb_star_k -> descend
                            if placed == cutlass.Int32(0):
                                # survived all L levels: final straddle class
                                o = atomicAdd(s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1))
                                pos = ra_last + o
                                if pos < cutlass.Int32(kK):
                                    if cutlass.const_expr(self.return_output_values):
                                        output_values_row[pos] = self.dtype(v)
                                    output_indices_row[pos] = smem_vals[isc]
                        isc = isc + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    cnt_strad = s_iscalars[1]
                    filled = ra_last + cnt_strad
                    if filled > cutlass.Int32(kK):
                        filled = cutlass.Int32(kK)
                    ipad = filled + tidx
                    while ipad < cutlass.Int32(kK):
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                        output_indices_row[ipad] = cutlass.Int32(-1)
                        ipad = ipad + cutlass.Int32(num_threads)
'''

with io.open(F, "r") as fh:
    src = fh.read()

i0 = src.index(START) + len(START)
i1 = src.index(END)
assert i0 < i1, "marker order"
new = src[:i0] + BODY + src[i1:]
with io.open(F, "w") as fh:
    fh.write(new)
print("spliced:", len(src), "->", len(new))
