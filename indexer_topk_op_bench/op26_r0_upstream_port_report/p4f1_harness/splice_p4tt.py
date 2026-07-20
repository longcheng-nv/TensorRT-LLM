# [p4tt] splice: add the tiny-tie fast-path branch to gvrpkgprod2's
# phase4_rank_scatter exact-tail. Structure:
#   if const_expr(p4_exact_tail and p4_tail_fast):   # [p4tt] new branch
#       need0 = ...
#       if cnt_strad > need0 and need0 > 0:
#           if cnt_strad <= 128: <collect + thread0 select>
#           else: <VERBATIM copy of the radix body, +4 indent>
#   elif const_expr(p4_exact_tail):                  # original if -> elif
#       <original text untouched>
import os

F = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "gvrpkgprod2", "top_k", "gvr_topk_decode.py")
src = open(F).read()
lines = src.split("\n")

TAIL_IF = "                if cutlass.const_expr(self.p4_exact_tail):"
NEED0 = "                    need0 = cutlass.Int32(kK) - rank_above_fine"
FIRE_IF = "                    if cnt_strad > need0 and need0 > cutlass.Int32(0):"
APPROX_ELSE = "            else:"

i_if = lines.index(TAIL_IF)
assert lines[i_if + 1] == NEED0 and lines[i_if + 2] == FIRE_IF
# end of the exact-tail body: the next line at indent <= 16 after the body
j = i_if + 3
while j < len(lines) and (lines[j].startswith(" " * 17) or not lines[j].strip()):
    j += 1
i_end = j  # first line after the exact-tail block (the approx `else:`)
assert lines[i_end] == APPROX_ELSE, lines[i_end]

radix_body = lines[i_if + 3 : i_end]  # fire-branch body at indent 24
radix_r = ["    " + l if l.strip() else l for l in radix_body]  # -> indent 28

FAST = '''\
                # [p4tt] tiny-tie fast path: when the exact-tail gate fires
                # with a small (b*, sb*) tie class (cnt_strad <= 128 — the
                # real firing cells hold 2), ONE candidate pass collects the
                # class and thread0 selects the top-need exactly, replacing
                # the 4 unconditional radix passes. Larger classes take the
                # UNMODIFIED radix select below (verbatim copy).
                if cutlass.const_expr(self.p4_exact_tail and self.p4_tail_fast):  # [p4tt]
                    need0 = cutlass.Int32(kK) - rank_above_fine
                    if cnt_strad > need0 and need0 > cutlass.Int32(0):
                        if cnt_strad <= cutlass.Int32(128):
                            # [p4tt] SMEM: (value_bits, cand_idx) pairs at
                            # smem_hist[2*o]/[2*o+1], o < 128 (slots 0..255).
                            # The 256 digit bins are dead here (the fast path
                            # replaces the radix levels that used them); the
                            # sb_star/ra staging in slots 2/3 was read by
                            # every thread before the pre-scatter barrier.
                            # Persistent radix scalars [256..258] untouched.
                            # Collect counter = s_iscalars[0] (dead after the
                            # scatter; same reuse as the radix rewrite pass).
                            if tidx == cutlass.Int32(0):
                                s_iscalars[0] = cutlass.Int32(0)
                            cute.arch.barrier()
                            itc = tidx
                            while itc < cand_count:
                                tv = smem_keys[itc]
                                tb = cutlass.Int32((tv - bmin_r) * inv1)
                                if tb < cutlass.Int32(0):
                                    tb = cutlass.Int32(0)
                                if tb > cutlass.Int32(kBins - 1):
                                    tb = cutlass.Int32(kBins - 1)
                                if tb == b_star:
                                    ts = cutlass.Int32((tv - f_lo) * finv)
                                    if ts < cutlass.Int32(0):
                                        ts = cutlass.Int32(0)
                                    if ts > cutlass.Int32(fbins - 1):
                                        ts = cutlass.Int32(fbins - 1)
                                    if ts == sb_star:
                                        to = atomicAdd(s_iscalars.iterator + cutlass.Int32(0), cutlass.Int32(1))
                                        if to < cutlass.Int32(128):
                                            smem_hist[to + to] = float_as_int32(tv)
                                            smem_hist[to + to + cutlass.Int32(1)] = smem_vals[itc]
                                itc = itc + cutlass.Int32(num_threads)
                            cute.arch.barrier()
                            # [p4tt] thread0 exact top-need0 select rewriting
                            # positions [rank_above_fine, kK). Consumed flag =
                            # the cand_idx slot set to -1 (indices are always
                            # >= 0), so a genuine -FLT_MAX value in the class
                            # remains selectable (no value sentinel). Ties
                            # (bit-equal values) pick arbitrarily: value-set
                            # exact.
                            if tidx == cutlass.Int32(0):
                                tj = cutlass.Int32(0)
                                while tj < need0:
                                    tbv = cutlass.Float32(self.NEG_FLT_MAX)
                                    tbi = cutlass.Int32(-1)
                                    ti = cutlass.Int32(0)
                                    while ti < cnt_strad:
                                        tvi = smem_hist[ti + ti + cutlass.Int32(1)]
                                        if tvi >= cutlass.Int32(0):
                                            tvb = smem_hist[ti + ti]
                                            tvv = cutlass.Float32(
                                                llvm.bitcast(
                                                    cutlass.Float32.mlir_type,
                                                    tvb.ir_value(),
                                                )
                                            )
                                            take = cutlass.Int32(0)
                                            if tbi < cutlass.Int32(0):
                                                take = cutlass.Int32(1)
                                            elif tvv > tbv:
                                                take = cutlass.Int32(1)
                                            if take == cutlass.Int32(1):
                                                tbv = tvv
                                                tbi = ti
                                        ti = ti + cutlass.Int32(1)
                                    pos = rank_above_fine + tj
                                    if cutlass.const_expr(self.return_output_values):
                                        output_values_row[pos] = self.dtype(tbv)
                                    output_indices_row[pos] = smem_hist[tbi + tbi + cutlass.Int32(1)]
                                    smem_hist[tbi + tbi + cutlass.Int32(1)] = cutlass.Int32(-1)
                                    tj = tj + cutlass.Int32(1)
                            cute.arch.barrier()
                        else:'''

NEW_ELIF = ("                elif cutlass.const_expr(self.p4_exact_tail):"
            "  # [p4tt] if->elif only")

new_lines = (
    lines[:i_if]
    + FAST.split("\n")
    + radix_r
    + [NEW_ELIF]
    + lines[i_if + 1 : i_end]
    + lines[i_end:]
)
open(F, "w").write("\n".join(new_lines))
print(f"p4tt spliced: radix body {len(radix_body)} lines duplicated")
