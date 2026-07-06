# op21 SHIP REVIEW — no-regress table + dispatch distillation (iter10)

Ship candidate: `gvr_ms_auto` (src/gvr_msc_op.py entry; single-CTA body in
src/gvr_ms_op.py). State = iter9 HEAD (`ccb22734b0`). All numbers below are
nsys pure-kernel cold-L2 medians on the B200-047-GPU0 axis; ratio = per-cell
best rival / ours (>1 = we win). Rival set = radix single/multi CUDA,
radix_cutedsl(+single/multi), sglang_streaming from report/{bs,seqlen}_data.csv
(same-HW rows). Regenerate any table with:
`OP21_NSYS_DIR=results/nsys/<archive> python3 scripts/nsys_verdict.py msa <dtype> [hw]`.

## 1. Verdict summary (B200 axis)

| grid | dtype | gm vs best rival | wins | gm vs best GVR-family | archive |
|---|---|---|---|---|---|
| P0 (17 cells) | fp32 | **1.249** | **17/17** | 1.139 (op8) | iter7_msa_b200/ |
| P0 (17 cells) | bf16 | **1.091** | 15/17 | 1.285 | iter9_16bit_b200/ |
| P0 (17 cells) | fp16 | **1.055** | 12/17 | 1.267 | iter9_16bit_b200/ |
| P1 canaries (24 cells) | fp32 | 0.901 | 5/24 | 0.952 | current ms_* (iter8 refresh) |

B300 cross-check (fp32 complete 2026-07-06, umb-b300-dp-185): **gm 1.268,
17/17** — HW-invariant, same pattern shape as B200 (weakest cell K1024 262K
BS1 both HW). bf16 partial 11/11 wins on the K1024 column (gm 1.097);
16-bit completion pending — see B300_RESULTS.md when landed.

Every dtype's P0 geomean beats the per-cell BEST rival — i.e. an oracle that
picks the fastest competing op per cell still loses to `gvr_ms_auto` on the
production-priority grid. vs any SINGLE rival op the margin is larger.

## 2. No-regress P0 table (17 cells × 3 dtypes, ratio = rival/ours)

| K | N | BS | fp32 µs | fp32 r | bf16 µs | bf16 r | fp16 µs | fp16 r |
|---|---|----|---------|--------|---------|--------|---------|--------|
| 1024 | 65536 | 1 | 14.11 | 1.414 | 12.96 | 1.136 | 13.50 | 1.082 |
| 1024 | 65536 | 4 | 14.94 | 1.345 | 13.50 | 1.124 | 13.95 | 1.072 |
| 1024 | 65536 | 8 | 15.20 | 1.352 | 13.82 | 1.106 | 14.40 | 1.047 |
| 1024 | 65536 | 16 | 15.68 | 1.349 | 13.95 | 1.119 | 14.50 | 1.066 |
| 1024 | 131072 | 1 | 15.87 | 1.255 | 13.15 | 1.123 | 13.54 | 1.077 |
| 1024 | 131072 | 4 | 16.61 | 1.211 | 13.79 | 1.091 | 14.18 | 1.047 |
| 1024 | 131072 | 8 | 17.02 | 1.217 | 15.04 | 1.025 | 15.30 | 0.996 |
| 1024 | 131072 | 16 | 17.79 | 1.387 | 15.49 | 1.161 | 15.68 | 1.132 |
| 1024 | 262144 | 1 | 18.85 | 1.064 | 14.37 | 1.035 | 14.98 | 0.977 |
| 1024 | 262144 | 4 | 19.68 | 1.038 | 15.17 | 1.002 | 15.46 | 0.968 |
| 1024 | 262144 | 8 | 20.19 | 1.197 | 15.36 | 1.160 | 15.74 | 1.109 |
| 1024 | 262144 | 16 | 20.77 | 1.510 | 17.41 | 1.333 | 17.70 | 1.279 |
| 512 | 131072 | 1 | 15.10 | 1.266 | 12.80 | 1.118 | 12.99 | 1.101 |
| 512 | 262144 | 1 | 17.98 | 1.064 | 13.47 | 1.078 | 13.47 | 1.079 |
| 2048 | 131072 | 1 | 17.34 | 1.158 | 15.33 | 0.960 | 15.30 | 0.951 |
| 2048 | 262144 | 1 | 17.76 | 1.115 | 16.77 | 0.885 | 16.64 | 0.877 |
| 2048 | 262144 | 16 | 22.43 | 1.425 | 20.29 | 1.157 | 20.26 | 1.136 |

Known holes (all 16-bit, none fp32): K2048 131K/262K BS1 (0.88–0.96 both
dtypes — K-proportional P3/P4 tail at cr=1, iter10 ablation pending/accepted);
fp16 K1024 262K BS1/4 (0.968–0.977) + 131K BS8 (0.996, par). bf16 gm on the
sub-grid excluding K2048 BS1 tails = all-green.

## 3. P1 canary table (fp32, single-CTA path, iter8 refresh on iter7+ HEAD)

P1 cells (N 4–16K, BS 64–1024) dispatch to single-CTA `gvr_ms` under the
production rules. fp32 binaries are untouched by iter9 (const_expr-pruned),
and P3 push is msc-only, so these iter8-HEAD numbers are current-HEAD numbers.

| K | N | BS | µs | r | bar | | K | N | BS | µs | r | bar |
|---|---|----|-----|------|------|-|---|---|----|-----|------|------|
|1024|4096|64|10.18|0.690|radix| |512|4096|64|9.50|0.734|radix|
|1024|4096|256|12.67|0.857|sgl| |512|4096|256|11.97|0.913|sgl|
|1024|4096|1024|35.55|0.888|sgl| |512|4096|1024|33.28|0.957|sgl|
|1024|8192|64|10.82|0.815|radix_s| |512|8192|64|10.05|0.864|radix|
|1024|8192|256|15.81|0.869|sgl| |512|8192|256|14.82|0.908|sgl|
|1024|8192|1024|43.68|0.891|sgl| |512|8192|1024|40.96|0.922|sgl|
|1024|16384|64|12.42|1.005|sgl| |512|16384|64|11.71|1.054|sgl|
|1024|16384|256|20.67|0.929|sgl| |512|16384|256|19.01|0.999|sgl|
|1024|16384|1024|58.05|0.953|sgl| |512|16384|1024|54.08|1.007|sgl|
|2048|8192|64|12.96|0.678|radix_s| |2048|16384|64|16.00|0.842|radix_s|
|2048|8192|256|17.66|0.942|radix_s| |2048|16384|256|23.97|1.079|radix|
|2048|8192|1024|58.08|0.896|radix| |2048|16384|1024|78.69|1.088|radix_s|

gm 0.901, 5/24. Structure: SGLang owns midN·highBS (we sit 0.86–0.96); radix
owns N4–8K·BS64 (0.68–0.86). Both are documented structural walls (PLAN red
lines; op12 floor analysis) and the grid is user-deprioritized. Canary
purpose: any FUTURE change must not push these below the row values here by
more than noise (±2.5%) — the campaign never regressed them (iter1 0.816 →
iter8 0.901, monotone improvement as P4/P3 levers carried over).

## 4. Dispatch distillation (production checklist)

Entire dispatch = **3 C-rules + 1 fuse gate**, all on compile-time-legal keys
(dtype, K=index_topk, BS, buffer max-N). No per-exact-N keys, no offline
tables (vs op20's 240-key dispatch). CUDA-graph compatible: for a captured
graph, `logits.shape` (BS, max-N buffer), dtype and K are capture-time
constants, so the selected kernel + C is fixed at capture; `seq_lens` (the
runtime tensor) never feeds dispatch.

```python
dt16 = dtype in (bf16, fp16)
if dt16 and N >= 65536 and N >= 32768 * BS:        -> gvr_msc C=8   # R-A
if K >= 2048 and N >= 196608 and BS <= 4:          -> gvr_msc C=8   # R-B
if N >= 65536 and 4 * BS <= NUM_SMS:               -> gvr_msc C=4   # R-C
else:                                              -> gvr_ms (single-CTA)
# inside gvr_ms: fuse = (BS <= NUM_SMS) and (4 * K <= kC=5120)      # R-F
```

Provenance / mechanism of each rule:
- **R-A (16-bit C8, iter8)**: at 16-bit the halved scan cost re-weights the
  serial tail — 8-way chunking pays (event C4/C8 1.08–1.14 in the win
  region) where it was noise at fp32. `N >= 32768*BS` excludes the measured
  BS16 collapse (0.71) and the 131K BS8 marginal.
- **R-B (K2048 fp32 hugeN C8, iter3)**: consistent win only at K2048 hugeN
  BS<=4 (cr=1 K-proportional serial tail benefits from more chunking);
  K1024 was noise-level, BS16 collapses.
- **R-C (C4, iter2)**: multi-CTA aggregate-L2-BW is THE P0 lever (iter1
  N-slope analysis: single-CTA 87ns/Kelt vs multi-CTA 17); 4·BS <= NUM_SMS
  keeps one full CTA wave.
- **R-F (fuse gate, iter1)**: fused P2+P3 slot-collect needs the spec-collect
  buffer to hold >= 4K (kC=5120 => fuse for K512/K1024, not K2048, where slot
  overflow made fused collect a measured 13% loss at large N); BS <= NUM_SMS
  = one CTA/SM wave for the +2·kC·4B slot smem.
- NUM_SMS is read from `torch.cuda.get_device_properties` (B200 = B300 = 148,
  so thresholds coincide on both currently-validated targets).

Falsified dispatch alternatives (never retry — LEARNINGS.md): C8 at fp32
K1024 262K holes (noise), QBINS=64 at highBS (wash), per-cell C tables
(op9 lesson: complex dispatchers lose on mixed batches).

## 5. A/B env knobs (all default ON; env-keyed compile cache)

| knob | =0 restores | shipped in | measured gain (nsys gm, where it applies) |
|---|---|---|---|
| `OP21_P4_RS` | legacy histogram-snap P4 | iter5 | P0 1.054 -> 1.104 (event gm snap/rs 1.058) |
| `OP21_P4_FAST` | fast paths off -> exact snap (iter11 semantics) | iter6/11 | P0 1.104 -> 1.125 (fast path covers ~100% of real bands) |
| `OP21_P3_PUSH` | leader DSMEM band gather | iter7 | P0 1.125 -> 1.249 (event gm gather/push 1.077, 14/14) |
| `OP21_P2_NATIVE` | cvt->fp32 16-bit ladder | iter9 | bf16 1.028 -> 1.091, fp16 1.043 -> 1.055 |
| `OP21_QBINS` | (int) P1b histogram bins, default 256 | iter5 | falsified lever, knob kept for probes |

Each knob is a pure fallback path kept compiled-in behind `const_expr`
(fp32 binaries carry zero 16-bit code and vice versa). dist_p1/dist_p4
(falsified levers) remain default-False source flags, not env knobs.

## 6. Exactness standing (gates, all green at HEAD)

- synth `scripts/smoke_exact.py`: 54/54 (3 seeds × 3K × 3N × BS{1,16}).
- real captures single-CTA: 60/60 layers (pro 30 K1024 + flash 21 K512 +
  v32 9 K2048, real_data_v2, tie-robust).
- real × cluster `scripts/smoke_real_msc.py`: 180/180 (C∈{2,4,8}) +
  adversarial preIdx (random/half-invalid × ms/C4/C8) 36/36.
- 16-bit real `scripts/smoke_real_16bit.py`: 360/360 (60L × {ms,C4,C8} ×
  {bf16,fp16}), native ladder ON.
- All-invalid preIdx ⇒ identity emit = inherited vendored contract
  (bit-identical to single-CTA; unreachable on real rows post-warmup-drop,
  count>kCC guards unreachable on real DSv4 data — see memory note).
- adversarial logits `scripts/smoke_adversarial_band.py` (iter11):
  72/72 — planted 2-ULP near-tie clusters straddling the K-th rank;
  covers the upstream ec04147502 failure mode (path C = exact snap
  fallback since iter11; the fixed-depth fine scatter is deleted).
- B300 exactness spot: `gvr_msc_op.py 4` 9/9 OK (2026-07-06, dp-185).

## 7. Ship risks / open items

1. **B300 16-bit verdict pending** (sweep 11/34 at interruption; resumable).
   fp32 already HW-invariant (gm 1.268 vs 1.249, 17/17 both).
2. K2048 16-bit BS1 tail (0.88–0.96): iter10 bounded ablation to pin
   (full/noP4/noWG at bf16 K2048 262K C8); acceptance is a legitimate
   outcome — lowest-priority cell family, fp32 equivalent is a 1.12–1.16 win.
3. fp16-only ~3% residual at K1024 262K BS1/4 (bf16 green) — deferred.
4. P1 highBS structural walls — user-deprioritized; guarded by §3 canaries.
5. Dispatch thresholds tuned on NUM_SMS=148 silicon; a future part with a
   different SM count shifts R-C/R-F boundaries — re-anchor before shipping
   binaries there (protocol in RESUME_PROMPT.md §Environment).
