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
