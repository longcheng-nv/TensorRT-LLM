# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Segment an ncu source-page CSV export of the [ptime]+[p4sub] timed twin
by its executed clock64 (CS2R) stamp landmarks and bucket per-instruction
counters per kernel phase / P4 sub-stage.

Input: `ncu --import <rep> --page source --csv` output (one row per SASS
instruction, in address order). The timed twin executes exactly these
stamps in program order on the profiled path:

  t0 t1 t2 t3 t4 t5 s8 s9 s10 s11 s12 s13 s14 t6 t7   (15; non-degenerate)

Degenerate cells execute fewer/other stamps (P2 bracket collapse writes
t2..t5 in one block; P4 copy-out collapses s10..s14) — the parser labels
segments by executed-stamp ordinal and the caller supplies the expected
sequence when it differs.

Output JSON: per-segment {label, n_sass, inst_executed, opcode-class
tallies, top mnemonics, PC-sampling stall sums (every numeric sampling
column found)}.

  python3 parse_ncu_p4.py --csv <src.csv> --out <seg.json> [--labels ...]
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict

DEFAULT_LABELS = [
    "p1_gather_stats", "smem_stage", "p1b_rungs",
    "p2_count_admission", "p3_collect", "p4_peer_wait", "p4_dsmem_gather",
    "p4_minmax", "p4_coarse_hist", "p4_coarse_search", "p4_fine",
    "p4_scatter", "p4_tail", "epilogue",
]
# segment i = code between executed stamp i and stamp i+1; label = what that
# code region computes = the phase ENDING at stamp i+1 -> shift by one:
# seg after t0 is p1_gather_stats ... seg after t7 (exit code) = "exit".

OPCLASS = [
    (r"^(LDS|LDSM)", "smem_load"),
    (r"^STS", "smem_store"),
    (r"^(ATOMS|RED\.S)", "smem_atomic"),
    (r"^(LDG|LD\.E|LDGSTS)", "gmem_load"),
    (r"^(STG|ST\.E)", "gmem_store"),
    (r"^(ATOM|RED)(?!S)", "gmem_atomic"),
    (r"^MAPA", "cluster_mapa"),
    (r"^(BAR|ARRIVES|CLUSTERBAR)", "barrier"),
    (r"^(FADD|FMUL|FFMA|FMNMX|FSETP|FSEL|MUFU|F2I|I2F|F2F)", "fp"),
    (r"^(IMAD|IADD3|ISETP|LEA|SHF|LOP3|SGXT|BFE|BFI|FLO|POPC|IABS|IMNMX)", "int"),
    (r"^(MOV|SEL|SHFL|PRMT|CS2R|S2R|R2UR|UMOV|ULDC|USHF|ULOP3|UIADD3|UIMAD|ULEA|UISETP|USEL|VOTEU?|P2R|R2P|PLOP3|UPLOP3)", "move_misc"),
    (r"^(BRA|BSSY|BSYNC|WARPSYNC|EXIT|RET|JMP|CALL|KILL|NANOSLEEP|YIELD)", "control"),
]


def opclass(mn):
    for pat, cls in OPCLASS:
        if re.match(pat, mn):
            return cls
    return "other"


def fnum(s):
    if s is None:
        return 0.0
    s = s.strip().replace(",", "")
    if not s or s == "-":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None,
                    help="comma list overriding DEFAULT_LABELS")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else DEFAULT_LABELS

    # ncu --page source --csv prefixes a `"Kernel Name",<name>` row before
    # the per-instruction header; skip to the real header line.
    lines = open(args.csv).read().splitlines()
    h = next((i for i, l in enumerate(lines)
              if l.startswith('"Address","Source"')), None)
    if h is None:
        sys.exit("no source-page header found")
    rows = list(csv.DictReader(lines[h:]))
    if not rows:
        sys.exit("empty csv")
    cols = rows[0].keys()

    def pick(*cands):
        for c in cands:
            for k in cols:
                if c.lower() == k.lower():
                    return k
        for c in cands:
            for k in cols:
                if c.lower() in k.lower():
                    return k
        return None

    c_src = pick("Source")
    c_addr = pick("Address")
    c_exec = pick("Instructions Executed")
    assert c_src and c_exec, f"missing columns; have {list(cols)}"
    # every other numeric-looking column gets summed per segment
    numeric_cols = [k for k in cols if k not in (c_src, c_addr)]

    # mnemonic per row (strip predicate + trailing args)
    def mnemonic(txt):
        txt = txt.strip()
        txt = re.sub(r"^@!?U?P\d+\s+", "", txt)
        m = re.match(r"([A-Z0-9._]+)", txt)
        return m.group(1) if m else "?"

    # find executed stamp landmarks: CS2R clock reads only (CS2R R, SRZ is
    # a zero idiom, not a stamp), with executed > 0 (skip untaken degenerate
    # collapse paths)
    marks = [i for i, r in enumerate(rows)
             if mnemonic(r[c_src]).startswith("CS2R")
             and "SR_CLOCK" in r[c_src] and fnum(r[c_exec]) > 0]
    nseg = len(marks) - 1
    print(f"[parse] {len(rows)} sass rows, {len(marks)} executed CS2R stamps "
          f"-> {nseg} segments (labels {len(labels)})")

    segs = []
    for si in range(nseg):
        lo, hi = marks[si], marks[si + 1]
        body = rows[lo + 1:hi + 1]  # instrs after stamp si up to+incl stamp si+1
        label = labels[si] if si < len(labels) else f"seg{si}"
        sums = defaultdict(float)
        opc = Counter()
        opcls = Counter()
        for r in body:
            ex = fnum(r[c_exec])
            if ex <= 0:
                continue
            mn = mnemonic(r[c_src])
            base = mn.split(".")[0]
            opc[mn] += ex
            opcls[opclass(mn)] += ex
            for k in numeric_cols:
                sums[k] += fnum(r[k])
        segs.append(dict(
            label=label, sass_rows=len(body),
            addr_lo=rows[lo].get(c_addr), addr_hi=rows[hi].get(c_addr),
            inst_executed=sums.get(c_exec, 0.0),
            opcode_class=dict(opcls.most_common()),
            top_opcodes=dict(opc.most_common(15)),
            column_sums={k: v for k, v in sums.items() if v},
        ))

    json.dump(dict(n_stamps=len(marks), segments=segs),
              open(args.out, "w"), indent=1)
    print(f"[parse] -> {args.out}")
    for s in segs:
        top = list(s["opcode_class"].items())[:3]
        print(f"  {s['label']:20s} inst={s['inst_executed']:>12.0f}  {top}")


if __name__ == "__main__":
    main()
