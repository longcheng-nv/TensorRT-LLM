# op19 sandwich GVR — knowledge base

## Effective techniques (inherited, sorted by impact)
- **CDF-aware compile-time frac placement (op18 place_mode=3)**: turned
  large-N 0.70-0.99x into 1.06-1.35x. Multi-seed (5) tightest-count fracs;
  fracs[0]=0 anchors exactness. Reuse op18 results/fracs_table.json.
- **M-ary one-pass ladder w/ cached count columns (op18)**: the "cheaper P2"
  lever op16 demanded. M2 ~free, M4 ~x1.25/pass. Winning column seeds P3
  with zero recount.
- **Cooperative-cluster threshold portfolio (op17)**: BS≤16 gm 1.18x, nsys
  1.21-1.67x. G redundant scans ≈ 1 pass at BS=1. pick_G: bs*G ≤ NUM_SMS/2,
  never G=2.
- **Tight threshold ⇒ cand-linear P4 shrink** (op17 iter1b), but with a
  ~7500cyc floor and placement sensitivity (op18 iter2).

## Ineffective directions (do NOT revisit without new evidence)
- Serial secant to pin tight/two thresholds (op16: tax-bound).
- op16 Scheme X free-peel WITH phase4_partition M0-wide 2-pass (pure cost).
- M=16 single-CTA multi-threshold (op17 iter4: ALU tax ≤1.0x).
- 2-kernel multi-CTA sweep+tail (op17 iter5: extra pass + launch, 0.63-0.84x).
- Branchless Int32(cmp) rewrite / unroll 8/16 of the M-ary scan (op18 iter2:
  latency-exposure-bound, no effect).
- Counting baseline's 2nd BS=1 pass as full-cost (L2 trap — op14/15/17/18).

## Architecture notes (B200 sm_100)
- L2 ≈ 126MB; BS=1 row (≤1MB) is L2-resident after pass 1 → pass-collapse
  only pays ~5K cyc at BS=1 large-N; at BS≥256 aggregate ≫ L2 → full price.
- smem cand capacity kC: 5120 (K512/1024), 6144 (K2048); packing in
  block_fused_snap_iter breaks if kC > 65535 (16-bit packed counts).
- num_threads: 512 default, 1024 for bs≤SMs && n≥65536 (op18 _config).

## Current best
- (none yet — iter0 = offline ceiling sim)

## Open questions
- Realized M0/K fraction on the real CCDF shapes per (K,N)? (iter0)
- Does band-P4 with runtime k_rem + seeded range hit the snap floor (~1 iter)?
- High-BS (≥256) pass-collapse: how big is the win once L2 no longer hides
  the baseline's extra passes?
- Multi-CTA sandwich: does concurrent direct-write CTA + band-refine CTA
  overlap cleanly (no extra cluster barrier)?
