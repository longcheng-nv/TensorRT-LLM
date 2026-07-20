#!/usr/bin/env python3
"""Combine run1.log (clock64 cycles + fractions) with nsys_anchor.csv
(trusted absolute kernel-duration medians) into the final
phase_breakdown.csv + PHASE_BREAKDOWN.md.

us_est = clock64 phase fraction x nsys PROD kernel median. The driver's
CUDA-event walls are kept as a secondary column (this node's event timer
quantizes to 2.048us ticks, so nsys is the anchor per house discipline).
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

PHASES = ["p1_gather_stats", "smem_stage", "p1b_rungs", "p2_count_admission",
          "p3_collect", "p4_select", "epilogue"]
LOGLBL = {
    "P1 gather/stats": "p1_gather_stats",
    "smem-stage": "smem_stage",
    "P1b rungs": "p1b_rungs",
    "P2 count+admission(+refine)": "p2_count_admission",
    "P3 collect": "p3_collect",
    "P4 select(+tail)": "p4_select",
    "epilogue": "epilogue",
}
COL = {
    "p1_gather_stats": "P1 gather/stats",
    "smem_stage": "smem-stage",
    "p1b_rungs": "P1b rungs",
    "p2_count_admission": "P2 count+adm(+refine)",
    "p3_collect": "P3 collect",
    "p4_select": "P4 select(+tail)",
    "epilogue": "epilogue",
}

# ---- parse run1.log ----
cells = []
cur = None
for line in (HERE / "run1.log").read_text().splitlines():
    m = re.match(r"\[(\w+)/(\w+) L(\d+)\] K=(\d+) N=(\d+) cr=(\d+) hit=([\d.]+) cfg=(\{.*\})", line)
    if m:
        cur = dict(model=m[1], isl=m[2], layer=int(m[3]), K=int(m[4]), N=int(m[5]),
                   cr=int(m[6]), hit=float(m[7]), cfg=eval(m[8]), cyc={}, frac={})
        cur["cell"] = f"{m[1]}/{m[2]}/L{m[3]}"
        cells.append(cur)
        continue
    m = re.match(r"\s+wall prod=([\d.]+)us timed=([\d.]+)us \(\+?(-?[\d.]+)%\) "
                 r"exact p/t=(\w+)/(\w+) mono=(\w+)", line)
    if m and cur:
        cur.update(ev_prod=float(m[1]), ev_timed=float(m[2]),
                   exact=(m[4] == "True" and m[5] == "True"), mono=m[6] == "True")
        continue
    m = re.match(r"\s+(.+?)\s{2,}(\d+) cyc\s+([\d.]+)%", line)
    if m and cur:
        key = LOGLBL[m[1].strip()]
        cur["cyc"][key] = int(m[2])
        cur["frac"][key] = float(m[3]) / 100.0
assert len(cells) == 6 and all(len(c["cyc"]) == 7 for c in cells)

# ---- nsys anchor ----
anchor = {}
with open(HERE / "nsys_anchor.csv") as f:
    for row in csv.DictReader(f):
        anchor[(row["cell"], row["arm"])] = float(row["med_us"])

FINDINGS = {
    "flash/32k/L22": (
        "Single-CTA cell near the BS=1 latency floor (~9us). P4 select is already the "
        "largest phase (40%, 3.6us) ahead of P2 admission (21%) and the P1 gather (17%). "
        "R0 admits on the rung ladder (high hit 0.69) so P2 stays cheap; the smem stage "
        "and epilogue are negligible."),
    "flash/128k/L22": (
        "cs=8 parallelizes the P2/P3 row scans (P3 down to 8%) but P4 runs leader-only, "
        "so it balloons to 58% (7.6us) — the t5-t6 window also absorbs the cluster "
        "handoff wait for the slowest peer's collect. P2 count+admission is the only "
        "other material phase (18%)."),
    "pro/128k/L30": (
        "Same shape as flash/128k but K=1024 and low hit (0.33): P4 select stays 58% "
        "(8.4us) and P2 17%. The rung ladder still admits (no visible refine tax; t3-t4 "
        "matches flash/128k within 3%), so low hit-rate cost shows up mostly as a "
        "slightly longer P1b/P1, not extra count passes."),
    "pro/512k/L30": (
        "The p4_exact_tail + p4tt firing cell: P4 (incl. tail fast path) is 54% "
        "(9.7us) and P2 grows to 19% (4.3us) at N=131k/CTA-slice 16k. This is also the "
        "worst instrumentation overhead cell (+6.7% nsys) — the extra stamps sit on the "
        "leader's critical path around the tail select."),
    "v32/128k/L34": (
        "K=2048 cr=1 with the kNumBins=512 diet: P4 still 53% (9.6us) — the 4x-smaller "
        "histogram does not change the leader-only structural picture. P1/P1b are the "
        "largest among all cells in cycles (K=2048 gather + rung build), but remain "
        "<17% combined."),
    "flash/1024k/L22": (
        "Largest N (262k): P2 (21%, 3.8us) and P3 (16%, 3.0us) grow with the per-CTA "
        "slice (33k elems), yet P4 remains dominant at 51% (9.4us). Epilogue/final "
        "cluster barrier stays <1% everywhere — cluster teardown is not a cost."),
}

# ---- CSV ----
with open(HERE / "phase_breakdown.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cell", "phase", "cycles_med", "frac", "us_est"])
    for c in cells:
        us_p = anchor[(c["cell"], "prod")]
        for p in PHASES:
            w.writerow([c["cell"], p, c["cyc"][p], f"{c['frac'][p]:.4f}",
                        f"{c['frac'][p] * us_p:.3f}"])
        w.writerow([c["cell"], "total", sum(c["cyc"].values()), "1.0000", f"{us_p:.3f}"])

# ---- MD ----
L = []
L.append("# GVR prod top-K kernel (PR#16457 head) — in-kernel per-phase breakdown")
L.append("")
L.append("**Method.** `gvrpkgtimed/` = spliced copy of `gvrpkgprod2/` (PR head, md5 "
         "`3396037c`, untouched) with `# [ptime]`-marked `cute.arch.clock64()` stamps in "
         "`_run_phases`, written by the leader CTA (cta_in_cluster==0) thread 0 to a "
         "`phase_ts` int64[num_rows, 8] GMEM tensor threaded through "
         "`__call__ -> gvr_topk_kernel -> run_one_row -> _run_phases`. No barriers added. "
         "BS=1 fp32 real captured cells, 10 warmup + 30 cold-L2 launches (512MB evict "
         "between launches), per-phase MEDIAN cycles.")
L.append("")
L.append("**Absolute us.** `us_est = phase fraction x nsys PROD kernel-duration median` "
         "(30 cold-L2 launches, NVTX-segmented single nsys pass). CUDA events on this "
         "node (umbriel-b200-081) quantize to 2.048us ticks and include ~4-5us launch "
         "overhead at BS=1, so nsys is the wall-time anchor; event walls are shown for "
         "reference only. Implied SM clock = window_cycles / nsys timed-wall "
         "(consistency check, ~1.6-1.7 GHz).")
L.append("")
L.append("**Timestamp map.** t0 entry | t1 P1 preidx gather/stats | t2 smem stage (==t1, "
         "cache disabled in prod config) | t3 P1b h-space rungs | t4 threshold final "
         "(R0 M-ary count + admission + fb_fix refine; secant on the no-R0 path) | t5 P3 "
         "collect (leader's own; cs>1 handoff #1 included in the P3 bucket) | t6 P4 "
         "select incl. cluster DSMEM gather, rank-scatter, p4_exact_tail/p4tt | t7 end "
         "(final cluster barrier). At cs=8 the t5->t6 bucket also absorbs the leader's "
         "wait on the slowest peer's collect (handoff #2).")
L.append("")
hdr = (["cell", "K", "N", "cs", "T", "hit"] + [COL[p] for p in PHASES]
       + ["total us (nsys prod)", "timed vs prod (nsys)", "exact", "mono"])
L.append("| " + " | ".join(hdr) + " |")
L.append("|" + "---|" * len(hdr))
for c in cells:
    us_p = anchor[(c["cell"], "prod")]
    us_t = anchor[(c["cell"], "timed")]
    row = [c["cell"], str(c["K"]), str(c["N"]), str(c["cfg"]["cluster_size"]),
           str(c["cfg"]["num_threads"]), f"{c['hit']:.2f}"]
    row += [f"{c['frac'][p] * us_p:.2f}us ({100 * c['frac'][p]:.0f}%)" for p in PHASES]
    row += [f"{us_p:.2f}", f"{100 * (us_t / us_p - 1):+.1f}%",
            "Y" if c["exact"] else "N", "Y" if c["mono"] else "N"]
    L.append("| " + " | ".join(row) + " |")
L.append("")
L.append("## Validation")
L.append("")
L.append("- (a) Output exactness: timed AND untimed index value-sets exact vs torch.topk "
         "on all 6 cells (unique count == K, gathered-value sets bitwise equal).")
L.append("- (b) Instrumentation overhead (nsys kernel medians, timed vs prod): "
         + ", ".join(f"{c['cell']} {100 * (anchor[(c['cell'], 'timed')] / anchor[(c['cell'], 'prod')] - 1):+.1f}%"
                     for c in cells)
         + " — all within the ~7% gate (worst pro/512k +6.7%).")
L.append("- (c) Monotonic t0<=t1<=...<=t7 on every one of the 30 launches per cell.")
L.append("- CUDA-event walls (quantized, launch-inclusive; reference only): "
         + ", ".join(f"{c['cell']} {c['ev_prod']:.1f}/{c['ev_timed']:.1f}us" for c in cells) + ".")
L.append("")
L.append("## Per-cell findings")
L.append("")
for c in cells:
    us_p = anchor[(c["cell"], "prod")]
    us_t = anchor[(c["cell"], "timed")]
    ghz = sum(c["cyc"].values()) / us_t / 1e3
    L.append(f"### {c['cell']} (K={c['K']}, N={c['N']}, cr={c['cr']}, hit={c['hit']:.3f})")
    L.append(f"- cfg `{c['cfg']}`; nsys prod {us_p:.2f}us / timed {us_t:.2f}us "
             f"({100 * (us_t / us_p - 1):+.1f}%); window {sum(c['cyc'].values())} cyc; "
             f"implied SM clock ~{ghz:.2f} GHz.")
    L.append(f"- {FINDINGS[c['cell']]}")
    L.append("")
L.append("## Cross-cell summary")
L.append("")
L.append("P4 select(+tail) is the dominant phase everywhere: 40% at the single-CTA cell "
         "and 51-58% at every cs=8 cell (7.6-9.7us absolute), because Phase 4 runs "
         "leader-only while cs=8 parallelizes only P2/P3 — and the bucket additionally "
         "absorbs peer-collect wait. P2 count+admission is the #2 phase (17-21%) and "
         "scales mildly with per-CTA slice; P1 gather/stats is 9-17%; P1b rungs 3-6%; "
         "smem-stage and epilogue are noise (<1%). This silicon-confirms the op35 "
         "finding that the P4 block is the battleground for further BS=1 optimization.")
(HERE / "PHASE_BREAKDOWN.md").write_text("\n".join(L) + "\n")
print("wrote", HERE / "phase_breakdown.csv", "and", HERE / "PHASE_BREAKDOWN.md")
