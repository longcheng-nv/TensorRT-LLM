# WALLS — structural walls (with one-line tests)

> A wall is a config-insensitive loss whose mechanism is understood. When every
> residual loss maps to a named wall, switch from explore to productize
> (SKILL: Stall Handling). Each wall carries a one-line test so it can be
> re-checked on new hardware/shapes instead of being re-litigated.

| Wall | Mechanism | One-line test | Established by |
|---|---|---|---|
| <name> | <one sentence> | <command or arithmetic> | <iter/campaign, date> |

## Seed examples from the GVR record (B200, selection kernels)
| Wall | Mechanism | One-line test |
|---|---|---|
| Single-CTA occupancy | BS=1 ⇒ grid=(1,1,1) ⇒ 1/148 SM bandwidth; register levers void | ncu occupancy + grid dims: grid ≪ SM count? |
| L2 trap | input ≪ L2 ⇒ re-read passes are L2 hits; traffic levers idle | `ncu --metrics dram__bytes_read.sum` ≈ input bytes? |
| Phase-chain latency (small N) | serial P1→…→P4 barrier chain sets a ~13-15µs floor | config-insensitivity probe + cold/warm differential |
| Pass-count floor | algorithm needs ≥k full-N passes; target implies <k | min_passes × bytes / BW vs target µs |
