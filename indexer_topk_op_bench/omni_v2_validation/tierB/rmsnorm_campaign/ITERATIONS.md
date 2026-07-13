# ITERATIONS — rmsnorm_campaign

## iter 0 — 2026-07-13 — CHARACTERIZATION (no kernel; SKILL Phase 1 + Phase 3 rung 0)
Hypothesis (ledger check: grep FALSIFIED.md/WALLS.md/../*/FALSIFIED.md → NO HITS, fresh
dense-class ledger — miss noted per protocol): where does the incumbent leave headroom?

Probes:
- Smoke: flashinfer.norm.rmsnorm vs fp32-upcast eager reference, maxdiff 0.0078 < 1e-2 → semantics confirmed.
- L1 bench_cold grid (incumbent vs eager rival): incumbent 2.1-15.9x faster than eager
  everywhere (eager rival eliminated as a threat; T=16384: 79.6 vs 1265.9 µs cold).
  L1 artifact caught: T=16 cold (21.3 µs) > T=256 (19.0) — graph-launch noise; nsys shows
  monotone 2.96/4.04 µs (M1 escalation validated).
- L3 ncu_attrib incumbent @T=16384 (KERNEL_REGEX=regex:RMSNormKernel; without the regex: prefix
  ncu -k matches nothing — gotcha logged): dram_read/input ratio 1.00 → single HBM pass,
  traffic levers VOID. grid 16384x(128 thr), 52 regs, occ 52%, mem throughput ~70% pk elapsed.
- Rung-0 crux P1/P2 (src/probe_copy.py, same-traffic torch elementwise; copy_ lowers to
  DtoD memcpy — gotcha, use torch.mul out=): nsys ceiling vs incumbent:
  | T | copy ceiling µs | incumbent µs | headroom |
  |---|---|---|---|
  | 1 | 1.64 | 2.83 | 1.73x |
  | 16 | 2.25 | 2.96 | 1.31x |
  | 256 | 3.64 | 4.04 | 1.11x |
  | 4096 | 21.54 | 21.82 | 1.01x (saturated) |
  | 16384 | 75.09 | 71.79 | incumbent beats torch copy (6.54 TB/s) |

Result: GO for a Triton candidate. Win region = small T (latency/occupancy regime:
incumbent uses only 128 threads/row); T>=4096 is bandwidth-saturated margin defense
(candidate must stay >=0.98, target = match).
Diagnosis: RMSNorm@7168 is 1-read-1-write; all remaining space is launch/latency/ILP at
small T, ~0-5% BW margin at large T.
Ledger write-back: WALLS.md + FALSIFIED.md #1 (traffic levers void — NCU-locked).
Anchor set: incumbent T=4096 nsys = 21.82 µs ± 3%.
Next: iter 1 = single-pass Triton kernel, 1 CTA/row, fp32 accum, autotune num_warps by T.

## iter 1 — 2026-07-13 — PIVOT (node migration 027→035 mid-iter; re-anchored 21.17 µs)
Hypothesis (ledger check: FALSIFIED #1 not triggered — launch/occupancy lever, not
traffic; WALLS large-T saturation ⇒ T>=4096 is margin defense only): a single-pass
Triton kernel (1 CTA/row, BLOCK=8192 masked, fp32 accum, autotune num_warps
{4,8,16,32}) closes the small-T latency gap while defending >=0.98 at large T.
Probe: rung 3 direct (kernel shape identical to incumbent's structure; rung 0 was iter0's
copy-ceiling table). Prior session's unverified partial claimed T=4096 ≈ 0.933 — treated
as hypothesis; re-measured below (actual 0.898, the partial was optimistic).
Result: gate 5/5 synth + 5/5 adversarial GREEN at every cell (50/50; no real track in
this campaign per PLAN). L2 nsys ×3-median, anchor drift +0.2% (21.212 vs 21.17):
  | T | cand µs | incumbent µs | ratio |
  |---|---|---|---|
  | 1 | 2.666 | 2.791 | 1.047 |
  | 16 | 2.976 | 2.961 | 0.995 |
  | 256 | 3.940 | 4.037 | 1.025 |
  | 4096 | 23.553 | 21.152 | 0.898 |
  | 16384 | 75.621 | 71.965 | 0.952 |
  worst 0.898 (T=4096) · geomean 0.982 · best 1.047 (T=1). Ship rule FAILS
  (geomean < 1.00; two cells < 0.98).
Diagnosis: small-T wins are real but modest (occupancy lever works: 1.047/0.995/1.025);
large-T loses 5-10% effective BW (cand 4.99 TB/s vs incumbent 5.55 @T=4096; 6.21 vs
6.53 @16384) — suspected code-shape artifact (masked 8192-tail = 12.5% dead lanes,
fp32 register pressure), NOT a wall (torch's own elementwise hits 5.45 TB/s @4096).
Ledger write-back: none yet — large-T deficit needs NCU attribution before naming a
wall or falsifying; that is iter 2's rung 0.
Next: iter 2 = NCU attribution of cand@4096 vs incumbent, then large-T repair variants
(exact-fit chunking / register diet); fallback lever if unfixable = regime dispatch
(triton T<=256, flashinfer T>=4096; 1 rule, within dispatch<=3 budget).

## iter 2 — 2026-07-13 — FALSIFIED (domain: Triton 1-CTA/row single-pass, T>=4096, bf16 h=7168, B200)
Hypothesis (ledger check: no hits — config lever, not traffic; single-pass preserved):
the large-T 5-10% deficit is a fixable code-shape artifact (warm-L2-miscalibrated
autotune num_warps, missing cache/eviction hints).
Probe rung 0 (L3 NCU, cand@4096): dram_read ratio 1.00 (single pass intact), 26 regs
(no spills), achieved occ 82%, autotune had picked 1024 thr → nothing pathological;
GO to config screen.
Probe rung 2 (L1 cold screen, 7 configs: NW {4,8,16,32} × eviction {none, evict_first
load/store/both}): best-of-cell T=4096 = NW8+ev/ev cold 0.995; T=16384 = NW4 plain
0.975. L2 nsys escalation of both winners (anchor drift -0.3%, 21.10):
  T=4096 NW8+ev/ev = 0.9271 (L1's 0.995 was graph-bias fiction — M1 again)
  T=16384 NW4      = 0.9519 (identical to iter1 autotune 0.9517 — config-insensitive)
Result: NO config reaches the 0.98 floor at either large-T cell.
Diagnosis: flashinfer's CUDA kernel runs ABOVE the generic elementwise BW ceiling
(iter0: beats torch same-traffic elementwise by ~2% @4096 and ~4.5% @16384); a Triton
single-pass 1-CTA/row kernel plateaus AT that generic ceiling (6.21 vs 6.53 TB/s
@16384). The loss is config-insensitive within Triton → structural.
Ledger write-back: FALSIFIED #2 (config levers at large T, evidence nsys) + WALLS
"flashinfer large-T BW-efficiency edge" (one-line test: nsys candidate-vs-incumbent
@T=16384; Triton caps ~0.95).
Next: iter 3 = productize via regime dispatch (SKILL Phase 6 "dispatch by regime"):
triton autotuned arm for T<=512, flashinfer arm for T>512 (1 rule, budget 3) →
expected grid 1.047/0.995/1.025/1.00/1.00, geomean ~1.013 → ship-rule candidate.
