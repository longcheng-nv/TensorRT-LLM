# WALLS — structural walls (with one-line tests)

> A wall is a config-insensitive loss whose mechanism is understood. When every
> residual loss maps to a named wall, switch from explore to productize
> (SKILL: Stall Handling). Each wall carries a one-line test so it can be
> re-checked on new hardware/shapes instead of being re-litigated.

| Wall | Mechanism | One-line test | Established by |
|---|---|---|---|
| Single-HBM-pass floor | RMSNorm is 1-read-1-write; incumbent dram_read == input bytes → no traffic to save, only latency/launch/BW-efficiency margins | `ncu --metrics dram__bytes_read.sum` ratio vs input = 1.00 | rmsnorm_campaign iter0, 2026-07-13 |
| Large-T bandwidth saturation | T=4096 incumbent = 99% of same-traffic elementwise ceiling; T=16384 incumbent (6.54 TB/s) beats torch's own copy kernel | nsys probe_copy.py vs incumbent per cell | rmsnorm_campaign iter0, 2026-07-13 |
| flashinfer large-T BW-efficiency edge | flashinfer's CUDA kernel exceeds the generic elementwise BW ceiling at T>=4096 (+2% @4096, +4.5% @16384); Triton single-pass 1-CTA/row plateaus AT that ceiling (6.21 vs 6.53 TB/s @16384) — config-insensitive across num_warps x eviction (7 configs, nsys) | nsys triton-cand vs incumbent @T=16384: cap ~0.95 | rmsnorm_campaign iter2, 2026-07-13 |

## Seed examples from the GVR record (B200, selection kernels)
| Wall | Mechanism | One-line test |
|---|---|---|
| Single-CTA occupancy | BS=1 ⇒ grid=(1,1,1) ⇒ 1/148 SM bandwidth; register levers void | ncu occupancy + grid dims: grid ≪ SM count? |
| L2 trap | input ≪ L2 ⇒ re-read passes are L2 hits; traffic levers idle | `ncu --metrics dram__bytes_read.sum` ≈ input bytes? |
| Phase-chain latency (small N) | serial P1→…→P4 barrier chain sets a ~13-15µs floor | config-insensitivity probe + cold/warm differential |
| Pass-count floor | algorithm needs ≥k full-N passes; target implies <k | min_passes × bytes / BW vs target µs |
