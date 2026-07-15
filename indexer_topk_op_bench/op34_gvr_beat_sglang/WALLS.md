# op34 WALLS (imported + own)
| Wall | Mechanism | One-line test | Source |
|---|---|---|---|
| Single-CTA occupancy (BS=1) | grid=(1,1,1) => 1/148 SM BW; register/pipeline levers void | ncu occ + grid dims | op#8 |
| L2 trap | input<<L2 => re-read passes L2-hit; DRAM-traffic levers idle | ncu dram__bytes_read ~ input bytes | op#8/op15/op29 |
| Short-N phase-chain floor | serial P1→P4 barrier chain ~9.7us @N8192 K512 fp32 BS1 | cold/warm differential + config-insensitivity | op32 |
| Pass-count floor | GVR ~2.5 passes single-CTA vs sglang 1 pass multi-CTA(8) | min_passes*bytes/BW vs target | op#8 |
| sglang = different skeleton | 1-pass histogram + multi-CTA beats GVR skeleton at short N | — | op32/retrospective |

## op34 own walls (append)
| Wall | Mechanism | One-line test | Source |
|---|---|---|---|
| Latency-bound MLP (not bandwidth) | BS=1 top-K <1% DRAM AND <1% SM peak; sglang wins via 8-CTA MLP not bandwidth | ncu dram%/sm% both <1% | op34 NCU_CRUX_048 |
| Cold-L2 saved-pass is L2-hot | both do 1 cold HBM read; GVR hint eliminates only an L2-hot pass (op29 1.03-1.13x) | ncu dram_bytes ~1x row both arms | op34 NCU_CRUX_048 |
| Collect/rank phase-separation | GVR skeleton collect ALONE (oracle,C64) ≈ sglang's fused collect+rank FULL kernel | col_orac nsys ≈ sglang nsys | op34 DOUBLE_LOCK_048 |
| Hint can't place exact-safe tight threshold | t=hint.min exact but M huge; hint-quantile small M but count<K (misses) | qsweep exactness count | op34 iter4 |
