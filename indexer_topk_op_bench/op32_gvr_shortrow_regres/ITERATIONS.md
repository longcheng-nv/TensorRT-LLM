# op32 iterations — GVR short-row register-resident (INSIGHTS P0)
Env: B200 sm_100 (cap 10,0, 148 SM), GPU1, cold-L2 flush (256MB) + CUDA-graph median (L1).
Incumbent = gvr_cutedsl_op26 (op26_r0auto fp32 N<65536 route). Scope (user): fp32 BS=1 only.

## iter0 — 2026-07-13 — CRUX: NO-GO(traffic) / GO(latency reframe)
Hypothesis (ledger: op15 smem-resident dead / L2-trap): register-resident cuts scan re-load passes.
Probe rung-0 = NCU attribution, N=8192 K512 fp32 BS=1 real:
  grid (1,1,1)x(512,1,1) · dram__thr **0.06%** · sm__thr 0.08% · warps_active 24.98%
  · issue_active **15.30%** · dur 29µs(locked).
Result:
  - NO-GO for the stated motivation: DRAM is idle (0.06%) → NO traffic to save. The register/
    smem re-read is already an L2 (actually register) hit. This is the op15 warm-L2 falsification
    reconfirmed by NCU (L2-trap veto, Phase-1.4 #1). "Fewer-passes/less-traffic" is a-priori idle.
  - GO for a REFRAMED lever: the kernel is LATENCY/ISSUE-bound (issue 15%, warps 25% structural
    single-CTA), NOT throughput-bound. The win must raise issue-rate / hide latency, i.e. the OPEN
    ledger lever "intra-CTA warp pipelining (Q3'/Q4', not run)". Headroom: issue 15%→~2x matches the
    sglang short-row gap (1.7-2.3x). Register-residency demoted to a SUPPORTING technique.
Baseline (L1 cold, us): N4096 ~19-22 / N8192 ~16-18 / N16384 ~17-19 (N4096 SLOWER than N8192 →
  fixed-overhead/launch-floor dominated, not scan-dominated — consistent with issue-bound).
Ledger write-back: FALSIFIED F1 (register-resident-for-traffic, L2-trap, dram 0.06%).
Next: attack the barrier-latency serial chain (14 block barriers; secant loop is data-dependent-serial).

## iter1 — 2026-07-13 — WASH: threads/CTA 512->768/1024
Hypothesis (reframe): more resident warps (16->32) hide barrier/dep latency, raise issue-rate.
Probe rung-2 = build GvrOp26Kernel with num_threads in {512,768,1024}, cold-L2 A/B, 3 scen x 3 N.
Result: t512 best in 7/9 cells; t768/t1024 net WASH-to-loss and HIGH variance (near launch floor).
  t768 broke exactness on 2 cells (! ) — non-512 thread counts violate a reduction/slot assumption.
Diagnosis: bottleneck is the SERIAL barrier-dependency chain (secant iter N+1 depends on iter N's
  block-reduced count), NOT warp starvation. Extra warps have no independent work to overlap across
  the 14-barrier chain → issue-rate unmoved. More warps ≠ more ILP when the critical path is serial.
Ledger write-back: FALSIFIED F2 (threads-raise at BS=1 short-N: barrier-bound, not warp-starved).
Next: the only skeleton-preserving latency lever left = CUT/CHEAPEN the barriers themselves
  (warp-level shuffle reduce replacing block-barrier+smem reduce per secant iter; the mc path's
  enable_warp_parallel_reduce is a candidate primitive). Tension: the biggest structural win (few
  barriers / single pass) IS sglang's skeleton, which the user excluded — bounds the achievable.

## iter2 — 2026-07-13 — WASH: enable_warp_parallel_reduce @512 threads
Hypothesis: replace the per-count-pass tid0-serial 16-slot final-sum with a warp0 shuffle reduce.
Probe: L1 cold A/B (off vs on), 3 scen x 3 N. Result: ratios 0.66-1.08 = pure noise, exact all Y.
Diagnosis: the final-aggregate is ~16 serial int-adds, drowned in barrier/interpolation latency —
  NOT the bottleneck. Ledger: FALSIFIED F3. Also: L1 event timing at this launch floor is unusable
  (same kernel 14-26µs) → all verdicts require nsys.

## iter3 — 2026-07-13 — BOUND: secant-refine is NOT removable (nsys)
Meta-op (UB/LB bounding): nsys pure-kernel, N=8192 K512 fp32 BS=1, MAX_REFINE_ITERS 15 vs 0.
Result: refine=15 (base) **9766/9733 ns** (best/real) = TRUE floor ~9.7µs (L1's 16-26µs was noise);
  refine=0 (no secant) **10475 ns** = SLOWER, not faster.
Diagnosis: killing the secant makes the initial threshold non-converge → P3 over-collects → P4 blows
  up. The secant is LOAD-BEARING (bounds candidate count), already ~1.46 iters (Q5e). A perfect
  barrier-fusion rewrite caps at ~1.46 iters ×2 barriers ≈ 0.6-1.2µs = 6-12% of 9.7µs, high impl
  risk, cheap probes already WASH. Ledger: WALLS W1.

## VERDICT — 2026-07-13 — NO-SHIP (structural wall, double-locked; pre-authorized negative)
Within "keep M2+secant skeleton, no sglang-copy", fp32 BS=1 short-N is at a structural wall; no
skeleton-preserving lever gives substantial improvement.
LOCK 1 (measured): NCU latency/issue-bound (issue 15% / warps 25% / dram 0.06%, single-CTA); nsys
  floor ~9.7µs; secant not removable (refine=0 slower); threads & warp-reduce & register-resident all
  WASH/dead; barrier-fusion ceiling ~6-12%, within probe noise.
LOCK 2 (relaxed control): op26's OWN R0 ladder (fewer barriers, single multi-thresh pass) already
  A/B-LOST to base at fp32 short-N (plain wins 1.10-1.14×, memory 小N R0门). Only variant beating base
  = sglang single-pass histogram (~7µs) = the excluded skeleton.
CONCLUSION: the ~30-40% gap to sglang IS the 5-phase-secant vs 2-phase-histogram structural
  difference; closing it needs the excluded skeleton (or lifting BS=1/short-N — op29 HBE owns
  N≥65536). op26_r0auto stays the best skeleton-compliant option here. NO further spend on this axis.

## iter4 — 2026-07-13 — DECOMP (aggressive re-open): cost is K-independent barrier chain
Probe (nsys, N=8192 fp32 BS=1 real): K-scaling + rs on/off.
  K512 rs 9762 / K1024 rs 9597 / K2048 rs 11176 ns; K512 snap 11584 ns.
Result:
  - K512→K1024 FLAT (9762≈9597) → cost is NOT K-dependent → P1-hint(∝K) and P4(∝cand∝K) are
    NOT the dominant cost at K≤1024. Kills "shrink-P4-via-kFTarget" and "overlap-P1-hint" as BIG
    levers (both K-dependent, but the kernel is K-flat here). Only K2048 shows +16% (P1/P4/smem).
  - rank-scatter already 19% faster than snap (9762 vs 11584) → P4's best algorithm is deployed;
    the win came from CUTTING BARRIERS (snap's iters → rank-scatter one pass). Precedent: cut
    barriers → win, ~19% for one phase.
Diagnosis: the dominant ~9.7µs is the K-independent structural chain (count/secant/P3 + their
  barriers) = W1, confirmed a 3rd way. Single-phase aggressive levers (kFTarget/P1) have limited
  headroom. The ONLY skeleton-preserving lever with precedent = apply "cut barriers" (that gave
  rank-scatter its 19%) to the count/secant/P3 barriers: (a) all-thread-redundant secant control
  (drop the tid0-only broadcast barrier/iter), (b) warp-specialization (Q3'/Q4', fill 85% idle).
  Ceiling ~10-20% STACKED, not 2×. 2× needs single-pass (excluded skeleton) or lifting BS=1/short-N
  (op29 HBE owns N≥65536). Next: user fork — (A) pursue ~10-20% barrier-fusion+warp-spec restructure
  within skeleton (risky, nsys+exact-gated), or (B) relax skeleton to port register-resident single-
  pass (only path to 2×). Recommend A only if 10-20% is worth the restructure risk; else B.

## iter5 — 2026-07-13 — FALSIFIED (silicon): path-A barrier-cheapened secant is SLOWER
Implemented GvrOp32Kernel(redundant_secant=True): per refine-iter, bracket-update + interpolation
run REDUNDANTLY on ALL 512 threads from registers (removing barrier B bracket-visibility + barrier A
nv-broadcast), keeping only block_count_ge's internal reduce barrier + 1 protect. Secant math copied
bit-for-bit.
Gate: exactness ALL PASS (27 cells, 3K×3N×3scen fp32 BS=1, tie-aware vdiff=0) — the restructure is
CORRECT.
Verdict (nsys, K512 N8192 best): base(red=0) **9721 ns** vs op32(red=1) **11277 ns** = **+16% SLOWER**.
Diagnosis: the "redundant interpolation is free (all threads parallel)" assumption is WRONG on silicon.
  512 threads each doing log2/div + a 16-slot smem re-sum (512×16 = 8192 smem reads/iter → bank
  contention + issue-slot pressure) costs MORE than the ~2.5 barriers removed. The rank-scatter
  precedent (cut barriers → +19%) does NOT transfer: rank-scatter cut barriers AND reduced work;
  this cut barriers but ADDED redundant work. At 512 threads the block barrier is CHEAPER than the
  redundant work needed to avoid it → confirms the barriers are already near-optimal.
Ledger: FALSIFIED F4. WALLS W1 reinforced (barrier chain is near-optimal, not cheapenable this way).

## VERDICT (updated) — 2026-07-13 — NO-SHIP, TRIPLE-locked
Path A (skeleton-preserving barrier-cheapening) is now FALSIFIED on silicon (iter5, +16% slower),
adding a 3rd lock to the register-resident (F1) and single-phase-lever (iter4) walls. Every skeleton-
preserving lever tried — register-resident (dead), threads (wash), warp-reduce (wash), secant-removal
(slower), single-phase kFTarget/P1 (K-flat = no headroom), barrier-cheapening (slower) — fails. The
9.7µs floor is the near-optimal barrier chain of the 5-phase secant skeleton. Substantial improvement
requires the EXCLUDED single-pass skeleton (path B = port op29 HBE to short-N) or lifting BS=1/short-N.
op26_r0auto remains the best skeleton-compliant option at fp32 BS=1 short-N. Campaign CLOSED.
