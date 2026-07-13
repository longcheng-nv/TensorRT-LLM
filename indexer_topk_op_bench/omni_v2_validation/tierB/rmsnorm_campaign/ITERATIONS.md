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

## iter 3 — 2026-07-13 — SHIP (src/candidate_dispatch.py)
Hypothesis (ledger check: cites WALLS "large-T bandwidth saturation" + "flashinfer
large-T BW-efficiency edge" as the motivation — route the walled regime to the
incumbent arm; FALSIFIED #1/#2 not re-proposed): a 1-rule regime dispatch
(T<=512 → iter1 Triton autotuned arm; T>512 → flashinfer) meets the ship rule.
Probe: none needed beyond iter1/iter2 measurements (both arms already nsys-charted);
threshold 512 sits in the (256, 4096] grid gap with 2x margin below the first losing cell.
Result: gate 5/5 synth + 5/5 adversarial GREEN at every cell (50/50). L2 nsys ×3-median
on the FINAL artifact, anchor drift -0.15% (21.139 vs 21.17); baselines reproduce
iter1 within 0.5% at every cell (T=16384: 71.965 = iter1's 71.965 — bracket holds):
  | T | cand µs | incumbent µs | ratio | arm |
  |---|---|---|---|---|
  | 1 | 2.667 | 2.766 | 1.0372 | triton |
  | 16 | 2.970 | 2.957 | 0.9957 | triton |
  | 256 | 4.018 | 4.037 | 1.0048 | triton |
  | 4096 | 21.126 | 21.153 | 1.0013 | flashinfer (self) |
  | 16384 | 72.067 | 71.965 | 0.9986 | flashinfer (self) |
  worst 0.9957 (T=16) · geomean 1.0074 · best 1.0372 (T=1)
Ship rule: geomean 1.0074 >= 1.00 ✓ · min cell 0.9957 >= 0.98 ✓ · exactness green ✓ ·
dispatch rules = 1 <= 3 ✓ · hard constraints: CUDA-graph compatible (host-side shape
dispatch, resolved at capture), out-of-place, zero incumbent edits ✓ → SHIP.
Diagnosis: the campaign's entire real win is the T=1 latency cell (+3.7-4.7%, from
wider CTAs: 1024 vs 128 threads on the single resident row); T=16/256 are parity-band;
large-T cells are the incumbent by construction. Honest framing: this is a small,
narrow win — flashinfer remains the best single kernel on 4/5 cells.
Ledger write-back: none (no falsification; walls unchanged).
Next: campaign verdict below. Un-spent levers for any future campaign: split-row
(multi-CTA/row) attack on the T=1/16 latency floor (iter0 headroom table says up to
1.73x/1.31x remains); TileIR/TMA streaming path vs WALL #3 (revival condition in
FALSIFIED #2).

---

# FINAL CAMPAIGN VERDICT — 2026-07-13 — SHIP (converged, iter 3 of 5; ~2 h wall)

**Artifact**: `src/candidate_dispatch.py` — 1-rule regime dispatch:
T<=512 → single-pass Triton (1 CTA/row, fp32 accum, autotuned num_warps);
T>512 → flashinfer.norm.rmsnorm (incumbent, unmodified).

**Per-cell nsys table (candidate/incumbent, ×3-median, anchored 21.17 µs ±3%,
umbriel-b200-035 GPU1)**:
| T | candidate µs | incumbent µs | ratio |
|---|---|---|---|
| 1 | 2.667 | 2.766 | 1.0372 |
| 16 | 2.970 | 2.957 | 0.9957 |
| 256 | 4.018 | 4.037 | 1.0048 |
| 4096 | 21.126 | 21.153 | 1.0013 |
| 16384 | 72.067 | 71.965 | 0.9986 |

**Verdict axes**: worst 0.9957 (T=16) · geomean 1.0074 · best 1.0372 (T=1).

**Ship rule (KICKOFF.md, verbatim clauses)**:
- geomean >= 1.00 vs incumbent: **1.0074 → PASS**
- no cell < 0.98: **min 0.9957 → PASS**
- exactness green (dense bf16 atol/rtol 1e-2): **50/50 (5 synth + 5 adversarial × 5 cells) → PASS**
- dispatch rules <= 3: **1 → PASS**
- hard constraints (CUDA-graph compatible, out-of-place, no incumbent source edits): **PASS**
→ **SHIP** (per AUTONOMY, actual production integration / PR is a human decision;
this campaign ends at this verdict).

**Honest negative content (pre-authorized)**: flashinfer remains the best *single*
kernel on the envelope — it is unbeaten at T ∈ {16, 4096, 16384} and its large-T BW
efficiency exceeds the generic Triton/elementwise ceiling (WALLS #3, FALSIFIED #2).
The shipped artifact's only genuine kernel win is the T=1 (and marginally T=256)
latency regime, worth +3.7-4.7% (T=1) via wider CTAs. If a 1-rule dispatch wrapper is
judged not worth carrying for a ~0.7% geomean gain, KEEP FLASHINFER AS-IS — that
call is the human's; the numbers above are the complete basis for it.

**Iterations used**: 4 of 5 (iter0 characterization, iter1 PIVOT, iter2 FALSIFIED,
iter3 SHIP). Ledger: 2 FALSIFIED entries, 3 WALLS entries — every residual loss maps
to a named wall (large-T → WALLS #2/#3; T=16 0.9957 = parity-band noise, not a wall).
