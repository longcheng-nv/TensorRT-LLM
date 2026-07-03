# op20 iterations

Protocol: every iter = tier_bench run (in-run 3-way: gvr_x vs gvr_cutedsl base vs
radix_cutedsl rival, cold-L2 CUDA-graph median) + entry here + git commit.
Primary metric = rival/x per cell (≥1.0 = fastest); secondary x/base.
Priority: tier1 fp32 K512/1024 > tier2 fp32 K2048 > tier3 16-bit.

## Iter 0 — 2026-07-03 — baseline (op19 verbatim)

**Strategy**: `gvr_sw_auto` copied unchanged; harness bring-up.
**Implementation**: `src/gvr_x_op.py` (= op19), `scripts/tier_bench.py`.

Smoke (GPU0, solo, 4 cells): exact 4/4; x/base gm 1.125; rvl/x 0.751 (N4K) – 1.103 (N65K).
Full tier1 (84 cells): `results/iter0_tier1.jsonl` — running.
Known holes (report loss-map, fp32): N≤32K BS≤256 (need ~1.22–1.29×),
N=262K BS≤16 (need ~1.69×; in-run spot 0.63).

**Bottleneck**: P4 snap loop = per-distinct-value stepping (mean 3.7–5.5 full-band
scans w/ fmin/fmax reduces, max 15) after a 1/256 histogram start.

**Next**: iter1 = level-2 sub-histogram refinement in P4.

## Iter 1 — 2026-07-03 — P4 level-2 sub-histogram (D2 part 1)

**Strategy**: between the level-1 256-bin locate and the exact snap loop, add one
band pass that (a) counts candidates ≥ bin-hi (fresh, so approximate level-1
binning can't break invariants), (b) re-histograms the target bin into 256
sub-bins, (c) moves thr to the sub-bin edge → snap starts at 1/65536 resolution,
0–1 iterations instead of 4–15. Guard: band ≥ 512 (small bands: snap already cheap).
Snap loop unchanged = exactness authority.

Smoke (GPU1, concurrent w/ GPU0 baseline — ratios valid, abs µs noisy; 12 cells):
exact 12/12. x/base at the small-N holes: N4096 1.11–1.12 (iter0: ~1.00),
N16384 1.12–1.24, N65536 BS64 rvl/x 1.47. Still losing N4096 (rvl/x 0.77–0.85).

Full tier1 (GPU0, solo): pending baseline completion.

**Next**: full-tier1 A/B vs iter0; then D3 (P1 fold ~2.2–2.7µs) or D1
(ladder-interpolation → tighter thr1 → smaller band) for the remaining N4K gap.

### Iter 1 FULL RESULT (GPU0 solo, 84 cells) — FALSIFIED as always-on
exact 84/84, but kernel iter1/iter0 gm **0.963** (min 0.881, max 1.063);
fastest-vs-rival 59→54. Sporadic wins only (N65536 BS1-4: +2.7-6.3%).
**Root cause of the miss**: the "snap = 4-15 full-band scans" datum came from the
BASELINE kernel's P4 (report iters_data); op19's sandwich P4 starts from a tight
[thr1,thr0) band where level-1 alone converges snap in ~1-2 iters — level-2's
extra band pass + 2 barriers is a tax with nothing to save (op16-pattern).
GPU1 smoke "improvement" was concurrent-run noise → never trust cross-GPU smoke
for accept/reject; full solo A/B only.
**Action**: revert src to iter0 (level-2 kept in git history); iter2 = measure
op19's OWN phase budget at the losing cells before touching code again
(clock64/nsys on N4096/N8192 BS1-64 and N131K/262K BS1-4).
