# GVR Top-K Decode — Repair the Mid-N x High-BS Loser Band (B200, fp32)

## Problem

Sparse-attention indexer top-K at decode time, batched. `logits[b, npad]`
fp32 real production captures, valid length `n_valid`; `pre_idx[b, k]` int32
is each row's PREVIOUS decode step top-k (temporal hint). Return
`indices[b, k]` int32, per-row exact top-k (index set; ties at the k-th
value either way; every strictly-greater index must appear, every row, every
run). k = 512 / 1024 / 2048 by model.

**Rows are HETEROGENEOUS**: row i is layer (i mod L) of a real multi-layer
capture — different rows have different value distributions, different hint
overlap, different tie structure. Do NOT assume any two rows are equal;
row-identity shortcuts will fail correctness (each row is checked against
its own layer's reference set).

## Target band (this is a REPAIR campaign — narrow and deep)

The current best production dispatch LOSES to the baseline (0.84-0.95x) on
exactly this band; your job is to WIN it (>1.0x, the more the better) without
regressing the guard workloads (uuid *_guard) below 0.95x:

- flash (k=512): n=4099 @ b {256,512,1024}; n=8195 @ b {256,512,1024};
  n=32771 @ b256; n=65538 @ b {32,128,256,512,1024}
- pro (k=1024): n=32771 @ b {256,512}; n=65539 @ b {128,256,512};
  n=262127 @ b16
- v32 (k=2048): n=65551 @ b {256,512,1024}

Structure of the band: rows are L2-to-DRAM transitional (n*4B = 16KB-1MB per
row, b*n*4B = 4MB-268MB per batch), many rows in flight — the fight is
occupancy / wave shaping / per-row launch amortization, not raw bandwidth.
Row work is independent; there is no cross-row math.

## Skeleton (mandatory) and freedoms

GVR skeleton: use pre_idx as prior to seed a threshold estimate -> refine
the threshold with exact-count feedback (secant/log-style) -> exactly refine
the surviving candidates. P1 (estimate) and P4 (candidate refine) are free
to redesign; mature primitives (radix/histogram select) may be absorbed
INSIDE phases. No wholesale replacement with a prior-free algorithm, no
dispatching to a separate non-GVR operator per case. You may NOT branch on
any hint-quality estimate computed outside the kernel; in-kernel admission
escape / count feedback is fine.

## Rules

- Exactness is non-negotiable and per-row (heterogeneous references).
- Timing: B200, cold-L2 protocol on the platform; the external acceptance
  re-times everything with nsys.
- CUDA C++ (cuda_cpp_only). No CUDA graphs, no framework kernels.
- Baseline = the production GVR kernel (TensorRT-LLM PR#16457 head) run
  natively batched on these same heterogeneous workloads.
