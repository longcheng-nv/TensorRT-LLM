# op17 GVR threshold-portfolio — LEARNINGS

## Current best
- **File:** `src/gvr_portfolio_cluster_op.py` (`gvr_portfolio_cluster(..., G="auto")`).
- **What:** single cooperative-cluster kernel. G CTAs/row each scan FULL N at their own
  threshold thr_r=pmin+r*(pmax-pmin)/(G-1) (redundant, free at low BS via idle bandwidth),
  DSMEM-share counts, pick best_r=tightest count≥K; the WINNER (rank==best_r) already holds
  its smem_ptcnt at thr* → P3 collect (done=1, no recount) + P4 with a TIGHT candidate set.
- **Result (fp32, exact everywhere, nsys pure-kernel):** BS=1 win 1.21–1.67× vs single-CTA
  gvr_cutedsl; event ×3-median avg ~1.22× (conservative — event penalizes cluster launch).
  All dtypes win (bf16 1.13–1.39×, fp16 1.08–1.38×, event). No regression vs baseline.
- **Dispatch:** `pick_G` = {16,8,4} with bs*G ≤ NUM_SMS//2, else G=1→baseline. BS≤16 win
  1.17–1.27×; BS≥32 → baseline (no regression). vs the EXISTING PR#15198 cluster: portfolio
  wins N≤65K, loses N≥262K (multicta partitions the scan) → use multicta for very-large-N.

## Effective techniques (by impact)
- **Speculative parameter portfolio in ONE cooperative cluster** (the win): the sweep
  REPLACES the search phase (no extra pass); winner reuses its own smem scratch for the tail.
- **band=[pmin,pmax] of preIdx values**: count(pmin)≥K ⟺ v_K≥pmin (all K preIdx ≥ pmin) so it
  always brackets the K-th value. (pmean is wrong — count(pmean) can be < K.)
- **G is the resolution↔overhead knob**: G=16 tighter cand (fixed the two G=8 ~1% dips) but
  higher per-cluster DSMEM-barrier cost → cap via bs*G budget.
- **done=2 (or the winner path's smem reuse)**: P3 only re-populates smem_ptcnt/cand_count
  when done≠1; a seeded threshold must either recount (done=2) or be the winner's own pass.

## Ineffective directions (falsified this bucket)
- **Single-CTA M-way multi-threshold P2** (iter4): M compares/element not hidden in one
  latency-bound CTA → net ≤1.0×; narrow K512/N16K/M=4 1.20× only.
- **2-kernel multi-CTA** (iter5): sweep is a SEPARATE kernel → adds a full-N pass + launch,
  not a P2 replacement → 0.63–0.84×.
- **G=2 clusters**: unstable ("unspecified launch failure") → never emit G<4.

## Architecture notes (B200 sm_100)
- BS=1 single-CTA is bandwidth-starved (~0.35% peak at 262K) → G≤148 redundant scans ~free,
  but only while data ≤ L2 (rows ≤262K fp32 = ≤1MB ≪ 50MB L2) and bs*G doesn't saturate SMs.
- Cluster launch has real host overhead → event/graph timing UNDERSTATES cluster-kernel wins;
  nsys pure-kernel is the truth (repo rule) and showed LARGER speedups here.

## Honest bound
- Original "+40% avg all seqlen/BS" is NOT reached (avg ~1.22× event / ~1.3-1.67× nsys BS=1).
  It is a real, exact, no-regression, all-dtype win over the single-CTA target baseline, in
  the low-BS/decode regime; very-large-N and high-BS dispatch to existing paths.

## Open follow-ups
- Deeper fusion: partition the scan across CTAs (like multicta) AND multi-threshold portfolio
  → could reclaim the N≥262K regime from the existing cluster.
- Register the op as a report column; ×3-median nsys over the full grid; real-data GSM8K gate.
