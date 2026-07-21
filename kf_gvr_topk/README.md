# `topk_c74f_sbx` — exact fp32 top-K index selection for the DSv4/V3.2 indexer (B200)

Standalone CUDA C++ operator produced by a KernelFactory evolutionary campaign
(campaign `tfb91bvwm972kfyf1bc1trj5e0`, round-2 winner `c74fb3c0` by agent
r002-a003) plus a 3-line engineer dispatch graft (`topk_small<17><<<1,1024>>>`
rung for `8448 < n <= 16896`).

## Contract

```
run(logits: fp32[1, npad] (CUDA), pre_idx: i32[1, k] (unused), n_valid: i64, indices_out: i32[1, k])
```

Exact top-k **indices** of `logits[0, :n_valid]` (any order; ties at the k-th
value resolved arbitrarily but exactly — mandatory (> kth) all present, rest
from the tie set, no duplicates). Hint-free: `pre_idx` is ignored (three
agents independently re-falsified pre_idx warm-hints on real indexer logits).

## Algorithm

- `n - k == 3, n <= 1536` → `bottom3_kernel`: selects the 3 complement
  (bottom) elements via warp reductions; output = everything else.
- `n <= 8448` → `topk_small<KPT>` @768 threads (KPT ladder 2/3/6/11), and
  `8448 < n <= 16896` → `topk_small<17>` @1024 threads: single-CTA,
  register-resident keys (row loaded from GMEM exactly once), 4×8-bit radix
  passes over monotone-mapped fp32 keys, warp-ballot compaction emit.
- `n > 16896` → `topk_coop`: cooperative launch (`(n+2047)/2048` CTAs, min
  SM count), 3-pass 11/11/10-bit radix with float4 loads, shared-memory
  histograms merged into GMEM per pass, `grid.sync()` between passes, and a
  **per-pass early-exit** when the boundary-bin count equals the remaining
  quota (typical real-logits rows converge in 1-2 passes). Histogram scratch
  is re-zeroed by the previous call's collect tail (no dedicated barrier).

## Measured (b200, nsys cold-L2, paired same-GPU, 865 real decode cells BS=1)

- geomean **1.6828×** vs TensorRT-LLM GVR PR head @e6fdbfac3d, 865/865 exact,
  zero cold regressions (borderline cells adjudicated at 60 reps).
- vs SGLang indexer top-K "v2" (PR-arm-normalized): geomean ≈1.11, win 569/865.
- vs in-tree `radix_cutedsl` (`single_pass_multi_cta_radix_topk`): ≈1.61, win 864/865.

Files: `kernel.cu` (device + `topk_launch` dispatch), `main.cpp`
(torch-extension host shim; swap for TVM-FFI or a TRT-LLM custom-op wrapper
as needed).
