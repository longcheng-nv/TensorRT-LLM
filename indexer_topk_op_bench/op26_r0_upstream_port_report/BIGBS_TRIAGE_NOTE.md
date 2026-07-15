# Big-BS PR-vs-op26 gap triage — harness artifact confirmed (2026-07-15, umbriel-b200-094 GPU0)

**Question**: report §7 shows PR (#16457) 1.6–6× slower than op26_r0auto at BS≥64.
The PR branch's production dispatch (`CuteDSLGvrTopKDecodeRunner.forward` +
`_pick_tuning`, pre-existing upstream, untouched by the PR) IS BS-aware — so is
the gap real or a measurement artifact?

**Method**: `ab_bigbs_runnercfg.py` — same-process 3-arm CUDA-event cold-L2
(256 MB flush/rep, median of 30) on op22-§env synth cells, BS∈{64,256,1024}:
- `pr_frozen`  = the §7 harness instantiation (cs = N≥65536?4:1, T1024, mbpm1,
  256-bit, enable_r0=True) — frozen BS=1 config, reproduces the report arm.
- `pr_runner`  = SAME kernel, config picked by a faithful replica of the
  production runner policy (cs 8/4/2/1 by (BS,N), T/mbpm/vec by `_pick_tuning`).
- `op26`       = op26_r0auto via build_call (report anchor).

**Result (20 cells, all 3 arms exact=True on all)**:

| geomean t/t(op26) | value |
|---|---|
| pr_frozen | **2.016** (max 5.48 @ bf16 K512 65K BS1024) — reproduces §7 gap |
| pr_runner | **0.918** — PR is ~8% FASTER than op26_r0auto |

Detail: pr_runner beats op26 on fp32 across the board (0.76–0.98; op26's
fp32 small-N plain-reroute and mb2 choices lose to R0-ladder@T512/mb2 at big
BS); parity on bf16 except K512 131072 BS≥256 where op26 leads 1.03–1.15
(residual mbpm/kC tuning band). Runner cs2/mc band at BS=64 N≥65536: parity.

**Conclusion**: essentially the ENTIRE §7 large-BS "PR collapse" is the
harness's frozen kernel config (bypassing the runner because the custom op
doesn't plumb `enable_r0`), NOT an algorithmic deficit of the PR. Driven
through its real production dispatch the PR matches or beats op26_r0auto at
large BS. §7's op26/pr curves at BS>32 must be read as "value of host
dispatch", and the §7 narrative sentence ("auto-dispatch switches to the
multi-CTA config" at high BS) is also backwards — the auto arm switches AWAY
from mc to 1cta at BS≥128 (`dispatch_r0_arm_op26`).

**nsys CONFIRMED (same day, same box, canonical protocol)**: `bigbs_nsys.py`
(one nsys process, NVTX cold-L2 ranges via `measure_cell`, 30 cold reps,
`parse_nsys_full.parse_rep` kernel-sum, evict-filtered) → `parse_bigbs.py` →
`bigbs_triage.csv`:

| geomean t/t(op26), nsys | value | CUDA-event (first pass) |
|---|---|---|
| pr_frozen | **2.265** (max 6.00 @ bf16 K512 65K BS1024) | 2.016 |
| pr_runner | **0.952** (range 0.747–1.181) | 0.918 |

20/20 cells 3-arm exact. Same picture, slightly larger frozen gap (nsys strips
the launch tax that padded the frozen arm's denominator). Residuals unchanged:
pr_runner beats op26 on fp32 (0.75–0.97) and bf16 16K (0.92–0.93); op26 leads
only bf16 K512 65K–131K by 3–18% (mbpm/kC tuning band). REPORT.html §7
narrative + KPI updated from this CSV (gen_report.py reads it).

**Caveats**: synth `best` scenario mostly K512 (+1 K1024 cell, +1 worst cell);
single GPU (b200-094 GPU0), single day.

**Follow-up**: (a) fix §7 blurb + add this note's KPI to the report; (b) the
op22 §10 dispatch-analysis claim "op26_r0auto BS≤64 optimal" stays true, but
the PR+runner path is an equally good production default — PR#2's dispatch
guard should target hit-rate/worst-axis routing, NOT BS routing (already
handled by the runner).
