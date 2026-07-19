# [p4f1] v3 splice: restructure the flag-ON branch of gvr_topk_decode.py
#   v2:  if need_more==0: <original scatter+pad>  else: <deep section>
#   v3:  <original scatter+pad, unconditional, + scratch-store insertion 3>
#        if need_more==1:
#            if cnt_strad <= CAP: <lane0 serial tail-select over scratch>
#            else: <deep section (re-indented +4)>
# Scratch ring: smem_hist[0..255] = (value_bits, cand_idx) pairs, o < CAP=128.
# Fine bins 0..255 are dead after the level-0 search (sb_star0/ra staged into
# hist[2]/[3] are read into locals before the scatter; the need staging lives
# at 258; the deep fallback re-zeroes 0..255 anyway).
import os

F = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "gvrpkgf1", "top_k", "gvr_topk_decode.py")
src = open(F).read()
lines = src.split("\n")

IF_HOT = "                if need_more == cutlass.Int32(0):"
ELSE_L = "                else:"
ELIF_MARK = ("            # ---- EXACT: one fine-histogram recursion on the "
             "straddling bin b* ----")

i_if = lines.index(IF_HOT)
# the matching `else:` at the same indent AFTER i_if
i_else = next(i for i in range(i_if + 1, len(lines)) if lines[i] == ELSE_L)
i_elif = next(i for i in range(i_else + 1, len(lines))
              if lines[i] == ELIF_MARK)

hot = lines[i_if + 1 : i_else]
deep = lines[i_else + 1 : i_elif]
assert any("deep section" in l for l in deep[:3]), deep[:3]

# 1. de-indent hot block by 4
hot_d = [l[4:] if l.strip() else l for l in hot]

# 2. insertion 3: scratch store in the straddle branch of the scatter
anchor = [
    "                        elif sb == sb_star0:",
    "                            o = atomicAdd(s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1))",
]
ai = next(i for i in range(len(hot_d) - 1)
          if hot_d[i] == anchor[0] and hot_d[i + 1] == anchor[1])
# end of the `if pos < kK:` write block = next line at indent <= 28 after ai+3
j = ai + 3
while j < len(hot_d) and (hot_d[j].startswith(" " * 29) or not hot_d[j].strip()):
    j += 1
scratch = [
    "                            # [p4f1] insertion 3 (v3): stash straddle",
    "                            # entry (value_bits, cand_idx) in the scratch",
    "                            # ring smem_hist[0..255] for the tail-select",
    "                            if o < cutlass.Int32(CAP):",
    "                                smem_hist[o + o] = float_as_uint32(v)",
    "                                smem_hist[o + o + cutlass.Int32(1)] = smem_vals[isc]",
]
hot_d[j:j] = scratch

# 3. new conditional + select + re-indented deep
select = [
    "                # [p4f1] v3: exact tail-select (or deep fallback) only on",
    "                # need_more rows; the hot path above ran the ORIGINAL",
    "                # scatter and falls through.",
    "                if need_more == cutlass.Int32(1):",
    "                    if cnt_strad <= cutlass.Int32(CAP):",
    "                        # [p4f1] one-thread exact top-`need` selection over",
    "                        # the <=128 scratch entries; rewrites output",
    "                        # positions rank_above_fine..rank_above_fine+need-1.",
    "                        # Serial simplest-correct (cnt<=128; fires rarely).",
    "                        if tidx == cutlass.Int32(0):",
    "                            need_t = cutlass.Int32(kK) - rank_above_fine",
    "                            j2 = cutlass.Int32(0)",
    "                            while j2 < need_t:",
    "                                bestv = cutlass.Float32(self.NEG_FLT_MAX)",
    "                                besti = cutlass.Int32(0)",
    "                                i2 = cutlass.Int32(0)",
    "                                while i2 < cnt_strad:",
    "                                    vbits = smem_hist[i2 + i2]",
    "                                    vv = cutlass.Float32(",
    "                                        llvm.bitcast(",
    "                                            cutlass.Float32.mlir_type,",
    "                                            vbits.ir_value(),",
    "                                        )",
    "                                    )",
    "                                    if vv > bestv:",
    "                                        bestv = vv",
    "                                        besti = i2",
    "                                    i2 = i2 + cutlass.Int32(1)",
    "                                pos = rank_above_fine + j2",
    "                                if cutlass.const_expr(self.return_output_values):",
    "                                    output_values_row[pos] = self.dtype(bestv)",
    "                                output_indices_row[pos] = smem_hist[besti + besti + cutlass.Int32(1)]",
    "                                # mark consumed (NEG_FLT_MAX never wins again)",
    "                                smem_hist[besti + besti] = float_as_uint32(",
    "                                    cutlass.Float32(self.NEG_FLT_MAX)",
    "                                )",
    "                                j2 = j2 + cutlass.Int32(1)",
    "                    else:",
    "                        # [p4f1] pathological (cnt_straddle > CAP): bounded-",
    "                        # correctness backstop = the v2 deep recursion; it",
    "                        # rewrites ALL output positions so the scatter's",
    "                        # writes above are simply superseded.",
]
deep_r = ["    " + l if l.strip() else l for l in deep]

new_lines = (lines[:i_if] + hot_d + select + deep_r + lines[i_elif:])
open(F, "w").write("\n".join(new_lines))
print(f"v3 spliced: hot={len(hot)} deep={len(deep)} lines")
