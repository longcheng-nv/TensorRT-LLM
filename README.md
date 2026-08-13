# GVR self-sampling top-K decode kernel (CUDA, Blackwell sm_100)

Exact tie-aware top-K index selection for the DeepSeek DSA indexer decode
path: for each row of `logits [B, n]` (fp32), return the indices of the
`K` largest values (`K` = 512 / 1024 / 2048) into `indices [B, K]`
(int32, unordered), scanning only the per-row valid prefix `n_valid`.

## Algorithm — Guess / Verify / Refine (GVR)

If the value of the K-th largest score were known, selection would be one
cheap pass (keep everything above it). The kernel therefore:

1. **Guess** — proposes value-space thresholds. Sources, in order of
   preference per shape: the previous decode step's top-K (`pre_idx`
   hint, gathered lazily), or a ~256-element self-sample of the row
   (the "self-sampling" rung ladder; see the `small_dense` gate).
2. **Verify** — counts exactly how many elements pass each threshold on
   the current row. A count >= K proves the kept set is a superset of
   the true top-K: the guess only sizes the work, never picks the
   answer. Misses trigger secant refinement / plateau descent.
3. **Refine** — exact tie-aware selection inside the bounded candidate
   set (rank-scatter + ticketed emit for boundary tie classes).

Output is unconditionally exact (tie-aware value-multiset equal to
torch.topk) on every supported shape; there are no data-conditional
accuracy modes.

## Files

- `kernel.cu` — the kernel + host launcher (single compilation unit; all
  shape dispatch is host-computed parameters into one kernel family).
- `kernel.h` — C ABI entry.
- `main.cpp` — PyTorch extension binding:
  `run(logits, pre_idx, n_valid, indices)`; caller-owned or lazily
  cached per-device workspace slab; CUDA-graph capture safe.

Build: standard `torch.utils.cpp_extension` load with `-arch=sm_100a`.

## Branch layout (one feature layer per commit)

1. production-hardened base kernel (`r5a005_v9_prod` lineage)
2. K=2048 (DSV3.2) domain enablement: four `k > 1024`-gated surgeries
3. launcher shared-memory cap fix (160 -> 168 KB)
4. `small_dense` self-sampling gate for the small-n / large-batch corner
5. partial-slice iteration-0 L2 prefetch on sampled-rung dispatches
6. two-pass cs=8 co-residency veto in the clustered register dispatch
7. TSH-floor staging on the slab split path (runtime-gated)
8. cs=8 co-residency veto in the streaming cluster merge (`gvr_clus`)
9. unconditional cs=8 co-residency veto (pass-1 exception deleted)

Tuning constants are B200 (148-SM, 228KB smem carveout) calibrated.
