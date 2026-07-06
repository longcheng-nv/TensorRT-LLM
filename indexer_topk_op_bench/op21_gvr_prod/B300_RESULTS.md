# op21 B300 cross-check results (2026-07-06)

Method: nsys pure-kernel cold-L2 medians, `gvr_ms_auto` at iter9 HEAD
(ccb22734b0 lineage; measured checkout c989bad434+), MEASUREMENT-ONLY (no
kernel/dispatch edits). Rival = per-cell best of the report CSVs **B300**
rows (radix single/multi CUDA, radix_cutedsl(+single/multi),
sglang_streaming); gvrbest = best existing GVR-family op. Verdicts via
`python3 scripts/nsys_verdict.py msa <dtype> B300`.

Provenance: first run on umb-b300-dp-185 (GPU0, 30C idle) died 11/34 into
the 16-bit sweep but completed fp32; full 51-cell single-axis re-run on
**umb-b300-dp-192** per B300_RELAUNCH_PROMPT.md Option A (dp-185 partials
archived to `results/nsys/iter10_b300_dp185_partial/`). Exactness sanity
`gvr_msc_op.py 4` = 9/9 OK on dp-185. B300 reports NUM_SMS=148 — same as
B200, so every dispatch threshold is numerically identical on both parts.

## Verdict summary

| dtype | B300 gm rival/ms | B300 wins | B200 gm (047 axis) | B200 wins | gvrbest/ms B300 |
|---|---|---|---|---|---|
| fp32 | **1.268** | **17/17** | 1.249 | 17/17 | 1.122 |
| bf16 | **1.089** | **15/17** | 1.091 | 15/17 | 1.273 |
| fp16 | **1.053** | **13/17** | 1.055 | 12/17 | 1.253 |

**HW-INVARIANT** (judged on aggregate verdicts per the op#9 lesson): all
three dtype geomeans match B200 within ±0.02; win counts identical for
fp32/bf16; fp16 gains one win. Zero win→loss flips. The only loss→win flip
is fp16 K1024 262K BS1 (B200 0.977 → B300 1.018, +4.2% — below the 5%
callout threshold). The K2048 16-bit BS1 tail reproduces on B300
(bf16 0.958/0.876, fp16 0.940/0.858 at 131K/262K) — confirming the iter10
ablation's verdict that it is a structural, HW-independent wall (K-
proportional P1/P1b floor at cr=1), not a B200-silicon artifact. No
16-bit largeN smallBS deviation attributable to a dispatch-boundary shift
(NUM_SMS identical).

Cross-B300-node consistency (free check from the re-run): dp-192's fp32
grid reproduces dp-185's per-cell ratios within ~±1% and the SAME gm
1.268, 17/17 (e.g. K1024 262K BS1 18.69 vs 18.75µs; K2048 262K BS1 17.34
vs 17.22µs). The dp-185 fp32 verdict table (archived reps) is therefore
interchangeable with the canonical one below.

## fp32 (17/17, gm 1.268)

| K | N | BS | ms_us | rival | r/ms | best_rival |
|---|---|----|-------|-------|------|------------|
|1024|65536|1|14.05|19.46|1.385|radix_cutedsl_multi|
|1024|65536|4|14.40|19.60|1.361|radix_cutedsl_multi|
|1024|65536|8|14.75|20.17|1.367|radix_cutedsl_multi|
|1024|65536|16|15.14|20.98|1.386|radix_cutedsl_multi|
|1024|131072|1|15.84|19.42|1.226|radix_cutedsl_multi|
|1024|131072|4|16.16|19.66|1.217|radix_cutedsl|
|1024|131072|8|16.45|20.55|1.249|radix_cutedsl_multi|
|1024|131072|16|17.09|24.36|1.425|radix_cutedsl|
|1024|262144|1|18.69|19.58|1.048|radix_cutedsl|
|1024|262144|4|18.98|20.53|1.082|radix_cutedsl|
|1024|262144|8|19.39|23.98|1.236|radix_cutedsl|
|1024|262144|16|20.03|31.30|1.562|radix_cutedsl|
|512|131072|1|14.27|18.75|1.314|radix_cutedsl|
|512|262144|1|17.18|18.80|1.094|radix_cutedsl|
|2048|131072|1|17.22|19.70|1.144|radix_cutedsl_multi|
|2048|262144|1|17.34|19.65|1.133|radix_cutedsl|
|2048|262144|16|21.54|31.65|1.470|radix_cutedsl|

## bf16 (15/17, gm 1.089)

| K | N | BS | ms_us | rival | r/ms | best_rival |
|---|---|----|-------|-------|------|------------|
|1024|65536|1|12.86|14.32|1.113|radix_cutedsl_multi|
|1024|65536|4|12.99|14.57|1.122|radix_cutedsl|
|1024|65536|8|13.15|14.82|1.127|radix_cutedsl_multi|
|1024|65536|16|13.38|14.99|1.121|radix_cutedsl_multi|
|1024|131072|1|12.99|14.15|1.089|radix_cutedsl_multi|
|1024|131072|4|13.44|14.46|1.076|radix_cutedsl|
|1024|131072|8|14.46|14.96|1.034|radix_cutedsl|
|1024|131072|16|14.75|17.32|1.174|radix_cutedsl|
|1024|262144|1|13.86|14.23|1.027|radix_cutedsl|
|1024|262144|4|14.53|14.64|1.008|radix_cutedsl_multi|
|1024|262144|8|14.72|17.28|1.174|radix_cutedsl|
|1024|262144|16|16.80|22.28|1.326|radix_cutedsl|
|512|131072|1|12.03|13.85|1.151|radix_cutedsl_multi|
|512|262144|1|13.28|13.99|1.053|radix_cutedsl_multi|
|2048|131072|1|14.85|14.23|0.958|radix_cutedsl|
|2048|262144|1|16.35|14.32|0.876|radix_cutedsl_multi|
|2048|262144|16|19.58|22.67|1.158|radix_cutedsl|

## fp16 (13/17, gm 1.053)

| K | N | BS | ms_us | rival | r/ms | best_rival |
|---|---|----|-------|-------|------|------------|
|1024|65536|1|13.28|14.05|1.058|radix_cutedsl|
|1024|65536|4|13.54|14.30|1.057|radix_cutedsl_multi|
|1024|65536|8|13.79|14.63|1.061|radix_cutedsl|
|1024|65536|16|13.89|14.84|1.069|radix_cutedsl_multi|
|1024|131072|1|13.54|13.92|1.028|radix_cutedsl_multi|
|1024|131072|4|13.89|14.24|1.026|radix_cutedsl|
|1024|131072|8|14.75|14.75|1.000|radix_cutedsl_multi|
|1024|131072|16|15.07|17.05|1.131|radix_cutedsl|
|1024|262144|1|13.86|14.11|1.018|radix_cutedsl_multi|
|1024|262144|4|14.85|14.37|0.968|radix_cutedsl_multi|
|1024|262144|8|14.94|17.08|1.143|radix_cutedsl|
|1024|262144|16|17.15|21.77|1.269|radix_cutedsl|
|512|131072|1|12.00|13.90|1.158|radix_cutedsl|
|512|262144|1|13.25|13.84|1.045|radix_cutedsl_multi|
|2048|131072|1|14.91|14.01|0.940|radix_cutedsl|
|2048|262144|1|16.38|14.05|0.858|radix_cutedsl|
|2048|262144|16|19.58|22.27|1.137|radix_cutedsl|

## Implications for the ship review

- SHIP_REVIEW.md §1 verdict table is confirmed on B300 silicon: every
  dtype's P0 geomean beats the per-cell best rival on BOTH architectures,
  with the same holes (K2048 16-bit BS1 family — structural, v3.2
  geometry only) and the same strongest cells (262K BS16, 1.27–1.56).
- The dispatch distillation needs no B300-specific terms: NUM_SMS parity
  makes the rules bit-identical, and no cell shows dispatch-boundary
  pathology.
- UPSTREAM_ASSESSMENT.md Stage-1 (kernel nsys on B200 AND B300) is
  hereby complete for the pre-port baseline; the ported kernel must
  reproduce these two tables.
