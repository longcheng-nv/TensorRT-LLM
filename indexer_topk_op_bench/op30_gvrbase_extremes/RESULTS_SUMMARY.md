# op30 — RESULTS SUMMARY (2026-07-13, umbriel-b200-047, 8× B200)

10-arm op22-style re-test on synthetic-data poles defined for **GVR (cuteDSL)
base absolute time** (not HLS-relative). Full grids: seqlen BS=1 (N 4K..1M) +
BS scaling (BS 1..2048 × N 4K..256K) + hugeN stretch (512K/1M, BS≤64),
K {512,1024,2048} × {fp32,bf16,fp16}. Single node, no anchor transfer.
15704/15704 cells, 0 errors, exactness 2392/2392 ok. Report: REPORT.html.

## Calibration poles (phase 1, 36-pt cfg×hr grid per K)

| model | BEST (GVR-base fastest) | WORST (slowest) | W/B (calib) |
|---|---|---|---|
| v4flash K512 | beta_shallow hr0.30 | beta_shallow hr0.90 | 2.17× |
| v4pro K1024 | aggregate hr0.15 | beta_deep hr0.85 | 1.75× |
| v32 K2048 | beta_shallow hr0.15 | aggregate hr0.85 | 1.54× |

GVR-base poles = **low-hr fast / high-hr slow** — op22 §1-2 labels reversed
AND shifted (true worst at hr0.85-0.90 seed-poisoning, not op22's hr0.55;
radix control spread ≤1.043× confirms data-insensitivity of the control).
Grid-pooled GVR-base worst/best time ratio: **1.502×** (906 common cells).

## Geomean speedup vs GVR-base (cold-L2, pooled all sweeps/K/dtypes)

| arm | best 数据 | worst 数据 |
|---|---|---|
| GVR multi-CTA (PR#15198) | 1.163 | 1.202 |
| Radix (cuteDSL) | **0.932** | 1.409 |
| Radix single-CTA (CUDA) | 0.443 | 0.665 |
| Radix multi-CTA (CUDA) | 0.359 | 0.540 |
| op#21 ms_auto (HLS-op25) | 0.995 | 1.524 |
| op#21 ms_auto (HLS-op27) | 1.025 | 1.524 |
| op#26 R0 (auto dispatch) | 1.161 | 1.509 |
| SGLang v2 (main 2026-07) | 1.436 | 2.250 |
| FlashInfer top_k (0.6.11) | 1.050 | 1.644 |

## Takeaways

1. **On GVR-base-BEST data the baseline is nearly unbeatable in-tree**: Radix
   cuteDSL drops below it (0.932), HLS arms are a wash (~1.0); only the
   seed-family siblings (mCTA 1.16, op26 R0 1.16) and SGLang v2 (1.44) win.
2. **On GVR-base-WORST data every rival gains** (HLS 1.52, op26 R0 1.51,
   radix 1.41, SGLang v2 2.25) — the baseline's high-hr seed-poisoning tax
   is the single largest data-sensitivity in the arm set.
3. Data-insensitive arms (radix CUDA/cuteDSL, SGLang, FlashInfer) show
   near-identical absolute µs across the two scenarios (internal control ✓).
4. SGLang v2 remains the strongest external arm on both poles (consistent
   with op28); its 2-kernel PDL path is reported with span column caveat.
5. op25 vs op27 differ only at K2048 (tail ladder), visible on best data
   (e.g. fp32 N=128K BS=1: 24.6 → 17.9 µs) and a wash on worst.
