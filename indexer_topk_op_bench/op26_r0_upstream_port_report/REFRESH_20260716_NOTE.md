# PR-contract refresh — full re-measurement note (2026-07-16, umbriel-b200-094)

**What**: every pr/base cell in REPORT.html re-measured with the arms driven
through the PR's own launch-shape contract (`GvrTopKKernel.launch` →
`pick_config`, branch HEAD `018251950f`), replacing the frozen
cs4/T1024/mbpm1/v256-always instantiation. BS-scaling grids extended to FULL
ISL coverage: synth 9 N rungs (4K–1M, was 3) × 3 dtype × best/worst × 3 K;
real ALL captured ISL rungs (flash/pro 9, v32 7, was 1) × 3 dtype × 11 BS.
54 nsys batches (op28 protocol), 8316 measurements, **0 errors**.

## Quality gates
- Exactness: gvr_pr **2772/2772**; op26 anchor n/a-gated as before; gvr_base
  2736/2772 — the 36 misses are ALL the known Flash-512k base-secant
  undershoot (hit≈0.06), now exposed across every dtype×BS by the full grid,
  and **repaired by R0 on every one of them**.
- Cross-run §8 comparability: op26 anchor drift new(094)/old(044) over 1122
  overlapping cells: **median 1.002, p95 1.047** (gate ≤1.15) → rival rows
  (unchanged code, 07-15 run) remain comparable.

## Headline shifts vs the frozen-config report
| metric | old (frozen) | new (launch shapes) |
|---|---|---|
| §3 synth PR/base gm (BS=1 fp32) | 1.118 | **1.148** |
| §3 synth PR-vs-op26 (p/o time) | 1.051 (op26 +5%) | **0.943 (PR +6%)** |
| §4 real PR/base gm | 1.330 | **1.309** (flash 1.265↑, pro 1.337↓, v32 1.332) |
| §4 flash 512k cell | 1.002 | **2.08** (cs=8; base undershoots there) |
| §4 flash 1024k cell | 0.977 | 0.977 (only remaining <1 real cell; PR≈op26 1.008) |
| §7 PR/base by BS | flat 1.106→1.117 (3 N) | flat **1.114→1.146** (9 N, real 1.204→1.176) |
| §7 op26-vs-pr | op26 2.7× ahead @BS1024 (artifact) | **parity**: o/p 1.100@BS1 → 1.013@BS1024 |
| §8 fp32 BS=1 pr vs radix/FI | behind | **ahead** (t(pr)/t = 0.98 / 0.96) |
| §8 16-bit pr vs radix/FI | ~1.5× behind (v256 mis-tune) | **1.20–1.30× behind** |
| §8 sglang_v2 vs pr | ~1.4× ahead of op26 | ~1.26× ahead of pr |

Mechanism of the shifts: (a) **cs=8 rung** (BS≤4, N≥131072) — pick_config
goes where the op-bench anchor's dispatch tops out at cs=4 (~6–12% on large-N
rungs); (b) **T=512 below 64K** + mbpm tiers; (c) **fp32-only 256-bit loads**
(removes the 5–11% half-prec cvt tax the old harness carried). §6 rewritten:
the old "~5% op26-ahead cluster-barrier floor" was a matched-shape statement;
under PR shapes the synthetic residual inverts (PR +6%), real is mixed
(flash PR +3.5%, pro op26 +2%, v32 op26 +5%).

## Artifacts
`refresh_harness/{ops_refresh,sweep_refresh,batches_refresh,parse_refresh}.py`
+ `drive_refresh_shard.sh` → `aggregate_refresh.py` → synth_3arm / real_3arm /
bs_synth (1716 rows) / bs_real (825 rows) CSVs + rival_long.csv GVR-row
replacement (2618 rival rows kept + 8316 GVR rows). gvrpkg_snapshot refreshed
to `018251950f`. REPORT.html regenerated (§3/§4/§6/§7/§8 narratives updated;
§7 N radio 9 rungs; §7 real ISL radio; §8 BS-view N/ISL filters).
nsys-reps/sqlite live in /tmp only (env-token rule).

## Follow-ups
- PR body tables (§ Performance) still show the 07-15 frozen-shape BS=1
  numbers — refresh from the new synth_3arm/real_3arm on request (values move
  in the PR's favor; flash 512k 1.002→2.08 is the headline fix).
- pro/v32 residual (op26 +2%/+5% at matched large-N): candidate levers =
  op26's K2048 log-interp P2 (see cs8_nsys: op26 mc +13% at K2048) — possible
  PR#2-era kernel port.

## Cold-hit large-BS regression (found in the full-ISL grid, disclosed in §7b)
Flash 1024k (N=262127, hit≈0.42 = V4 hit-rate valley): pr AND op26 fall to
**0.68–0.79× of base** at BS≥128/cs=1 (fp32 mild 0.98 at BS≤64; bf16/fp16
affected at ALL BS 0.76–0.84); v32 256k same shape (0.75–0.87); consistent
with synth worst large-N 16-bit (0.83–0.90). Both R0 implementations regress
together, all exact → algorithmic R0-ladder low-hit regime (admission miss →
extra full-N fallback scans, unmasked once cs=1/throughput-bound), NOT a port
or config artifact. This is the PR#2 dispatch-guard target and pins its
criterion: **hit ≲0.45 && large N → route secant** (BS handled by runner).
Magnitude exceeds the prior disclosure envelope (PR body says synth worst min
0.930 + real BS=1 24/25) — PR body known-limitation quantification pending
user confirmation.
