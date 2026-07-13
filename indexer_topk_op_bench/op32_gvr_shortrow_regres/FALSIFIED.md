# op32 falsification ledger  (scoped triples: conclusion / domain / evidence / root-class)
F1 — register-resident to save scan re-read traffic — DEAD.
  domain: fp32 BS=1 short-N (4-16K), single-CTA. evidence: NCU (dram__throughput 0.06%, sm 0.08%).
  root-class: structural-wall (L2-trap; the row is 32KB, all re-reads are L2/register hits). Same
  mechanism as op15 smem-resident (warm-L2 parity). Revival: only if a variant is memory-bound —
  it is not at BS=1. Register-residency may still SUPPORT a latency opt but cannot win on traffic.

F2 — raise threads/CTA 512->768/1024 at BS=1 short-N — WASH-to-loss.
  domain: fp32 BS=1 N4-16K K512. evidence: L1 cold-L2 A/B (t512 best 7/9, high variance);
  t768 exactness-broke 2 cells. root-class: structural-wall — critical path is the serial
  barrier-dependency chain (data-dependent secant), not warp occupancy; extra warps have no
  independent work to overlap. Revival: only alongside a restructuring that creates independent
  overlappable work across the barrier chain (none proposed yet).

F3 — reduction final-aggregate (tid0 16-sum vs warp0 shuffle) at 512 threads — WASH, not the cost.
  domain: fp32 BS=1 short-N. evidence: L1 A/B (noise 0.66-1.08). root-class: measurement/structural —
  final-sum is ~16 int-adds drowned in barrier latency. The bottleneck is the barrier COUNT, not reduce.

CORRECTION (2026-07-13, double-check vs count_ge_multi_bench/REPORT.html) —
  "M=4 multi-threshold = per-element compare tax" is a MIS-ATTRIBUTION. Do NOT repeat it.
  Evidence (nsys cold-L2, B200, BS=1, block_count_ge micro-bench, ×vs M=1):
    N     M=2    M=4    M=6    M=8
    4K    1.00   1.01   1.03   1.12
    8K    1.01   1.20   1.23   1.56
    32K   1.01   1.23   1.40   1.61
    256K  1.05   1.46   1.89   2.40
  count_ge is memory-bound: the row is read once, +M cheap predicated compares. M=4 costs
  ~1.15-1.46× a single scan (NOT 4×), amortized 0.31-0.35/threshold; M=2 is ~free everywhere.
  The report RECOMMENDS M=4 for iterative Phase-2 refine (fewer total scans).
  => The M-COUNT is cheap; it is NOT why op26 ships M=2 or why R0 lost at short-N.
  TRUE reasons op26 retreated to M=2 (three, none is count cost):
   (i)  single-round admission economics: R0's one M-ary pass already admits 96.8% at M=2; the
        extra M=4 columns cost 1.2-1.46× on ALL rows to first-pass-admit ~3% more — not repaid
        (the cheap R1 log-falsi shot covers the 3%). M=4-for-refine (report) ≠ M=4-single-round.
   (ii) secant already at the iteration floor: base secant = 1.46 iters (Q5e); M-ary saves ~0.46
        iter but the per-pass barriers do NOT shrink → Opt-F multi-threshold P2 measured WASH.
   (iii) the short-N R0 loss ("小N R0门", plain wins 1.10-1.14×) is the 256-bin HISTOGRAM BUILD
        fixed cost (zero 256 bins + K atomicAdds + warp-0 rung extraction), NOT the M-count.
  META: this is a "microbench isolates the primitive → full-kernel silicon slaps the projection"
  case (the report recommends M=4; production ships M=2). count_ge cost ≠ full-kernel latency.
  The op32 wall (W1) stands: short-N is barrier/pass-count latency-bound; the ~9.7µs floor is
  ~2.5 latency-bound count passes + P1/P4/barriers, and every op26-family pass-reduction lever
  (R0 hist / Opt-F) is either fixed-cost-dominated or at the iteration floor. NO-SHIP unchanged.

F4 — path-A barrier-cheapened secant (all-thread-redundant control flow) — SLOWER on silicon.
  domain: fp32 BS=1 short-N (K512 N8192 tested). evidence: nsys (base 9721 vs op32 11277 ns, +16%);
  exactness ALL PASS (27 cells) so it's CORRECT but slower. root-class: structural-wall — removing
  ~2.5 barriers by making 512 threads redundantly interpolate + re-sum smem_wcnt (8192 smem reads/
  iter, bank contention + issue pressure) costs MORE than the barriers saved. The block barrier at
  512 threads is CHEAPER than the redundant work to avoid it. rank-scatter's "cut barriers → +19%"
  does NOT transfer (it also reduced work; this added work). Revival: only a barrier cut that does
  NOT add per-thread redundant work (none identified within the secant skeleton).

F5 — exponential/log binning of the 256-hist (vs current LINEAR) — WORSE, not better.
  Hypothesis (user): logits CCDF is ~exponential near top-K, so log-spaced bins would place the
  rank-quantile rungs more precisely than linear bins → better M=2/4 initial thresholds + tighter
  fast-write sandwich. HOST ablation (scripts/abl_expbin2.py, 24 rungs, fp32 BS=1 K512/1024
  N8192/16384, count(full-N>=rung) vs EXACT kthvalue rung as ground truth):
    binning count-error  q0.85=55.9  q0.35=11.5  q0.15=11.7  q0.05=4.8  (LINEAR)
                         q0.85=102   q0.35=42.7  q0.15=27.2  q0.05=21.6 (LOG)
    OVERALL: LINEAR 21.0 vs LOG 48.5 — LINEAR wins at EVERY qfrac incl the deep tail.
  root-class: premise-error. (1) LINEAR 256-bin is ALREADY near-optimal: ~1-3% count error at the
  rungs (range ~6.5 units / 256 bins = 0.025 value res). (2) The hist bins the PREV-topK VALUES
  (bounded [pmin,pmax] sample), NOT the full-N CCDF; the rank-quantile rungs (q0.85/0.35 = 436/180th
  of K) sit in the prev-topK BODY, not its exponential tail — log-dense-near-vlo misallocates
  resolution AWAY from the rungs. Even the deep thr0 tail (q0.05) favors LINEAR. (3) admission is
  already 96.8% (M2D) so placement-precision headroom is tiny regardless.
  fast-write corollary: the "if more precise, combine with fast-write" premise fails at step 1
  (log is not more precise), AND fast-write = op27 sandwich (excluded op18 lineage, loses to base
  at fp32 short-N). Killed cheaply by host ablation — no kernel written. Revival: only a
  boundary-ADAPTIVE binning (dense at the unknown v_K), but linear's 1-3% error is not the bottleneck.
