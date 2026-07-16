# SGLang v2 conditional-exactness note (§8 correctness caveat) — 2026-07-16

Origin: Slack thread (Yue Weng) — sglang v2 DSv4 top-K has "serious precision
issues" on uniform-[0,1) logits at kv_len 128K; threshold bin supports max 2048
candidates (sglang `topk_impl.cuh` L673-674). Question: why did the op26 report
(§8: 374 sglang_v2 cells, all exact=True on synth + real) not catch it?

## Mechanism

sglang v2 histograms the TOP `kHistBits` bits of the fp16(ordered) cast of each
fp32 score (TopKCluster path: 10 bits; Register/Streaming: 12 bits). Elements
strictly above the threshold bin are emitted directly; elements INSIDE the
threshold bin go to a tie buffer capped at `kMaxNumTie = 2048` — overflow
candidates are **silently dropped** (arrival order decides survivors), then
`handle_tie` exact-selects only within the kept subset. So the kernel is exact
iff `count(threshold bin) <= 2048`. For K=2048 the cap has zero structural
headroom (`kMaxNumTie == kMaxTopK`).

## Findings (scripts in this dir; run with PYTHONNOUSERSITE=1 on B200)

1. `tiebin_analysis.py` + `sglv2_repro.py` — Slack repro confirmed: uniform
   [0,1) N=131072 puts ~4113 elements in one 10-bit bin → kernel end-to-end
   exact=False (K=2048: 2017/2048 slots wrong, max value error 0.0155). The
   report's `_exact` gate catches this — the miss was data coverage, not gate
   laxness.
2. `tiebin_extended.py` — ALL layers × ALL ISLs, real captures:
   - V4 flash (K=512, 21 layers, 4K–1M): max tie-bin 247 → margin ≥ 8.3×. SAFE.
   - V4 pro (K=1024, 30 layers, 4K–1M): max 395 → margin ≥ 5.2×. SAFE.
   - V3.2 (K=2048, 58 layers, last step): max 1466 (256k L52) → margin 1.40×.
   - Synth extrapolation to N=2M: worst-scenario K=2048 max 1467 (margin 1.40×),
     never crosses — single-row synth lacks the layer/step extreme-value
     statistic that produces the real overflow.
3. `tiebin_allsteps_count.py` — V3.2 all 58 layers × all 15 decode steps:
   - 128k: 1/870 cells over cap (L52 step4, tie=2278).
   - 256k: 3/870 cells over cap (L52 steps 3/6/12, max 2214).
4. `sglv2_real_overflow.py` — end-to-end kernel on the flagged rows: v32 256k
   L52 steps 3/6/12 all FAIL (25–168 of 2048 slots wrong, max value error
   0.0088, non-deterministic across runs). All below-cap control rows exact.

## Verdict

- The §8 exact=True results are correct but **slice-conditional** (bench layer ×
  last decode step). sglang v2 is only conditionally exact.
- V4 Flash/Pro deployment envelope: safe by ≥5× margin up to ISL=1M and
  (by synth extrapolation) far beyond.
- V3.2 (K=2048, cr=1): real production captures ALREADY cross the cap at
  ISL 128K/256K on flat-distribution layer 52 (~0.1–0.3% of layer-step cells);
  each decode token runs 58 indexer layers, so a single bad layer-step feeds a
  wrong top-K set into sparse attention. sglang_v2 must not be labeled "exact"
  for the K=2048 use case. GVR/radix arms are unconditionally exact.

## FlashInfer top_k under the same battery (2026-07-16, umbriel-b200-017)

`fi_topk_correctness.py` applies the identical battery to flashinfer 0.6.11
(the §8 `flashinfer_topk` arm; B200 default path = `fast_topk_clusters_exact`,
8-bit-radix multi-phase refinement over the FULL fp32 bits; `deterministic=True`
routes to the multi-CTA `radix_topk` path — both modes tested):

1. Adversarial synth: uniform-[0,1) N=128K/1M × K∈{512,1024,2048} (the
   sglang-killer), all-equal row (total tie), fp16-collision block (8192
   fp32-distinct values inside one fp16 ulp straddling the K boundary — a
   16-bit histogram structurally cannot separate them), fp32 + bf16. ALL exact.
2. The sglang-FAILING real rows (v32 256k L52 steps 3/6/12, 128k L52 step 4)
   + below-cap controls, both modes. ALL exact.
3. Broad real sweep: flash 21L×9ISL (K=512) + pro 30L×9ISL (K=1024) + v32
   bench-layers×7ISL (K=2048), last step, batched. ALL exact.
4. V3.2 full temporal grid: 58 layers × 15 steps × {128k,256k} = 1740 cells
   (where sglang's overflow lives). 1740/1740 exact.

**Total 2245/2245 exact.** Mechanism: no fixed tie cap — the per-phase global
overflow buffer is sized `max_model_len // num_clusters` per cluster (whole
row fits), so boundary ties are never silently dropped. Verdict: FlashInfer
top_k is unconditionally exact on everything we can construct, including the
exact rows/distributions that break sglang v2. Correctness ranking of external
arms: FlashInfer = Radix = GVR (unconditional) > SGLang v2 (conditional).

Env recipe for a fresh node: `pip install --no-deps --target /tmp/gvrval1/fi_raw
flashinfer-python==0.6.11`, symlink ONLY the `flashinfer` package into
`/tmp/gvrval1/fi_clean`, run with `PYTHONNOUSERSITE=1
PYTHONPATH=/tmp/gvrval1/fi_clean` (container torch 2.12 wins).

## Radix cuteDSL under the same battery (2026-07-16, umbriel-b200-017)

`radix_topk_correctness.py` — identical battery on the §8 `radix_cutedsl` arm
(vendored SinglePassMultiCTARadixTopK). Mechanism: pure radix, iterative
8-bit-digit refinement over the FULL ordered-bit pattern (fp32 4 rounds,
fp16/bf16 2) — no truncated-bits threshold bin, no fixed tie buffer, so no
kMaxNumTie hazard class; the shape spread exercises both single-CTA and
multi-CTA branches. **2245/2245 exact**, and unlike sglang (fp32-only) the
FULL adversarial set ran at fp32 + fp16 + bf16 (at 16-bit the fp16-collision
block becomes a true 8192-way boundary tie — worst tie load — still exact).
SGLang-failing rows / broad sweep / 1740-cell temporal grid all exact.

Final §8 external-arm correctness ranking, all empirically on the same
battery: FlashInfer = Radix cuteDSL = GVR (unconditional) > SGLang v2
(conditional). cutlass 4.5.0 env for a fresh node: `harness/setup_bs_env.sh`.

## Report injection

REPORT.html §8.1 (sglang) + §8.2 (flashinfer) notes are injected by
`update_report_sglv2_note.py` (idempotent, marker-delimited; REPORT.html itself
stays local-only / untracked). `gen_report.py` now calls the injector after
every full regen, so the block survives regeneration (an 07-16 regen had
silently wiped the first injection — last-writer hazard). The §8 trend-bullet
"every rival is exact" is qualified with a pointer to §8.1/§8.2 in both the
gen_report.py template and (patched in place) the current REPORT.html.
