# DESIGN — HBE-C: hint-ladder cluster single-pass (op29 next campaign / "op30")

> Status: DESIGN (2026-07-13, deep-dive per user request). Ledger-checked.
> Inputs: gvr_agent_retrospective/RETROSPECTIVE.md · op21 HLS (MATH §10,
> gvr_ms_op.py place_mode=5) · op25/op27 ladder ships · op29 iter1-12 ·
> results_b200_op29 (cluster-domain rival timings).

## 1. Where preIdx was fairly beaten — and where it was NOT

op29 falsified the hint **only in the streaming domain (BS>512)**, for two
domain-specific reasons:
- gather economics: K×BS random reads (BS=1024 ⇒ ~0.9x cell, iter3);
- estimator quality: hint-value quantiles vs the true threshold —
  q0.9 is 100% one-sided but cand med 2.7-6.5xK / p90 29xK (iter1), while
  the 4096-pt coalesced row sample gives cand 1.3-2.6xK, scenario-invariant.

Neither reason survives in the **cluster domain (BS<=512, N>cluster_floor)**:
- gather cost: K/4 subsampled reads (~4-16KB effective) vs row 0.5-4MB
  ⇒ 0.3-0.8% of a pass. The iter3 falsification domain does not apply.
- coordination: preIdx is row-global BY NATURE — 8 CTAs can use the same
  threshold guess with ZERO cross-CTA reduce. Any sample-based estimator
  needs a DSMEM histogram all-reduce first (a serial latency link, exactly
  what dominates BS=1 cells at 10.9-33.3us). This is the hint's unique
  structural asset that no sample can replicate.
- estimator noise is answered by HLS, not by giving up: the ladder.

## 2. What HLS actually taught about preIdx (the reusable core)

HLS = h-tracked Ladder-Secant (op21 MATH §10):
  theta_opt ≈ G_S^-1(h·K) — a plug-in ORDER STATISTIC of the preIdx-gather
  value sample; the ONLY unknown is the scalar h (effective retention).
- op21 ships: qfracs=(0.75,0.5,0.25) rank-quantile ladder over a 256-bin
  hint-value histogram (place_mode=5), log-count regula-falsi fallback,
  distributed cluster fallback (iter14, worst 2.0x).
- op27 ship: K2048 TAIL ladder (0.75,0.45,0.048) — the shipped answer to
  "hint quantiles land ~2000 bins low at K2048" (op29 iter1's falsification;
  ledger revival condition "a better K2048 hint statistic" = SATISFIED by
  ladder geometry, not by a better single point).
- op25 lesson: "the ladder is also currency" (M=5 tax 7-19%) ⇒ M<=3.
- HLS Step 3 (h-hat adaptive column placement) was DEFERRED — this design
  is its revival in a 1-pass architecture.
- op23 caveat honestly stated: its UB (0.851 vs radix) bounded the op21
  1-CTA threshold-parallel FAMILY vs radix; phase conclusions do not
  transfer across kernel families (retrospective §3.5-⑤). It does NOT bound
  a data-parallel 1-pass cluster design vs sglang_v2.

## 3. The prize (results_b200_op29, real scenario, rival=sglang_v2)

Cluster domain = **369/906 report cells (41%)** — vs HBE's current 12%.
- mid-BS wave-bound: N=262144 BS 64/256/512 = 94/104/191us (K512);
  hugeN BS=64: 142-276us. Full pass-elimination value available: est 1.4-1.9x.
- BS=1 latency-bound: 10.9 (131K) - 33.3us (1M). Near launch floor at small
  N (≈wash expected); at 512K/1M the phase-chain shortening (drop hist pass
  + replace 4096-bin all-reduce with M×8 scalars) is worth est 1.3-1.6x.

## 4. Algorithm — HBE-C (5th dispatch tier, BS<=512 && N>cluster_floor)

Keep sglang_v2's persistent 8-CTA cluster pool and work distribution
(inherited verbatim; Opt-B's "cluster at BS>~SMs" falsification does not
apply — that was GVR's own cluster growth, and BS<=512 is the incumbent
cluster domain anyway). Replace its per-row 2-phase body:

```
C0 (all 8 CTAs, redundant — redundant reads are L2 hits, op17 crux):
    gather preIdx values, stride-4 subsample (K/4 reads)
    256-bin hint-value histogram (smem, per-CTA identical)
    place M=3 rungs v1>v2>v3 at rank-quantile fracs of the hint sample
      fracs per K from the op18 CDF-aware tables; K2048 = op27 tail
      geometry (0.75, 0.45, 0.048-class)
C1 (each CTA, its N/8 slice, ONE scan):
    per element: <=3 cmps -> rung band
    val >= v3 (lowest rung): append {val,idx} to per-CTA smem buf
                             (+ per-CTA global spill), per-rung counts
C2 (cluster):
    DSMEM reduce of M×8 scalar counts (vs stock 4096-bin hist all-reduce)
    r* = tightest rung with cluster_count(>=v_r) >= K and no overflow
    -> candidates >= v_r* are COMPLETE across CTAs (count-validity, op29)
    distributed candidate mini-hist (DSMEM 4096-bin, ~2-4K atomics total)
    -> exact b* -> each CTA classifies its own candidates;
    >b* emit via the stock map_shared_rank output path; =b* stock tie
MISS (no valid rung / overflow):
    stock cluster Phase1+Phase2 for that row (HLS Step-2 precedent,
    op21 iter14) — fail-soft, exactness unconditional.
```

Savings vs stock cluster per row: (a) Phase1's per-element
F2F+twiddle+smem-atomic histogram build (the iter9 issue-wall cost) -> <=3
cmps; (b) 4096-bin×8 DSMEM all-reduce -> M×8 scalars; (c) the second full
collect pass -> candidate-only resolve. Kept: single scan (info floor), one
cluster sync, stock exactness machinery.

## 5. Ledger checkpoint (Phase 2.5) — all hits cited

| Red-line hit | Disposition |
|---|---|
| hint-quantile @K2048 falsified (op29 iter1) | revival condition satisfied by op27 tail-ladder geometry + count-validity fail-soft |
| hint gather = fixed tax (op29 iter3) | falsified domain was BS=1024 streaming; here BS<=512, gather 0.3-0.8% of row bytes |
| cluster DSM at high BS (Opt-B/Q5f) | domain excluded: BS<=512, inherit sglang persistent pool |
| P2 multi-threshold (Opt-F/op8/op27#3) | that was iterative-refinement inside secant; this is single-shot speculative ladder (op18/op25/op27 SHIPPED class) |
| ladder-is-currency (op25) | M<=3, compile-keyed |
| HBE fused pass in L2-trap (op29 iter3) | BS<=512 × N>=131072: BS·N·4B = 256MB+ at BS>=256... NOTE: BS<=32 × N<=262144 sits INSIDE the L2 trap (rival's 2nd pass L2-hot) — the win there must come from the ISSUE saving + all-reduce shortening, not DRAM; crux must measure this pole separately |

## 6. Probe ladder (nothing gets built before rung 0-2 pass)

- rung 0 CRUX (host, ~1 h): on all 27 op22rr bundles × real V4 captures:
  hint-ladder placement replay -> per-(scenario,K,N): tightest-rung cand
  counts, per-rung bracket rate, miss rate. GO if real-scenario expected
  passes <= ~1.2 and miss <= ~10%.
- rung 1: extend replay_hbe_cand.py with the C0 ladder estimator.
- rung 2 MICROBENCH: DSMEM M×8-scalar reduce vs 4096-bin all-reduce
  latency at BS=1 (isolates the serial-chain saving; decides whether BS=1
  small-N cells are worth engaging at all).
- rung 3: implement as tier-5 behind a flag in gvr29 (baseline immutable);
  gate 3-track (incl. adversarial hints — hint must never affect
  exactness); pilot cells: (131072..1048576) × BS {1,16,64,256,512};
  L2 nsys same-batch vs sglang_v2 + gvr_cutedsl anchor.

## 7. Follow-ons (recorded, not in scope)

- P3: sub-65536 streaming hint tier (hint subsample replaces sample+
  find_threshold fixed phases at N=32768; small stakes, ledger revival
  condition "cut fixed costs" partially available).
- P4: HLS Step-3 proper — temporal h-hat tracking across decode steps
  (per-row persistent state; production plumbing; needs user decision).
- Production integration decision for tiers 4+5 (user).
