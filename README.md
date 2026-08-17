# GVR self-sampling top-K decode kernel (CuTeDSL, Blackwell sm_100)

High-fidelity CuTeDSL translation of the CUDA `GVR-selfsampling-CUDA` branch
(layer-10 head): exact tie-aware top-K index selection for the DeepSeek DSA
indexer decode path. For each row of `logits [B, n]` (fp32), returns the
indices of the `K` largest values (`K` = 512 / 1024 / 2048) into
`indices [B, K]` (int32, unordered), scanning only the per-row valid prefix
`n_valid`.

## Fidelity contract

- **Exactness**: output is unconditionally exact (tie-aware value-multiset
  equal to `torch.topk`) on every supported shape — verified 0 INEXACT on the
  886-cell x 11-BS real-capture grid (9,746 cases) against the CUDA arm.
- **Performance**: per-case cold-L2 kernel-time regression vs the CUDA
  original <= 2% post-arbitration on the same 9,746-case grid (nsys
  same-process paired, B200).
- **Dispatch**: bit-exact transcription of the CUDA host dispatch — a pure
  function of `(b, n, k)`, cross-checked against an independent second
  transcription by exhaustive boundary + fuzz sweeps.

## Layout

- `kernel.py` — DPS entry: `run(logits, pre_idx, n_valid, indices)`,
  `run_ws(...)` (explicit workspace), `workspace_bytes()`.
- `ct_dispatch.py` — host dispatch (pure `(b,n,k)` function).
- `ct_main.py` / `ct_reg.py` / `ct_clus.py` / `ct_regclus.py` — the four
  kernel families (sampling-ladder slab/streaming, register-resident,
  cluster streaming, clustered register-resident).
- `ct_common.py` — shared device units (fkey order-embedding, redux scans,
  histogram crossings, DSMEM push).
- `ct_op.py` / `ct_workspace.py` — operator shell: compile cache keyed on
  the constexpr tuple (lazy JIT, ~35-55 reachable variants), launch cache,
  per-device slab workspace.

## knife5 note (layers 7-9 of the CUDA source)

The CUDA kernel arms TSH-floor staging on the slab path behind a
grid-uniform *runtime* gate (`gridDim.y > 15 && k <= 1024 && n4 <= 32768`)
with a dual scan instantiation. This translation keys the *compile-time*
specialisation on the same predicate (host knows `b/k/n` before launch):
ungated variants compile to the bit-wise pre-knife5 kernel; per-launch
semantics are identical because the gate is uniform over the grid.

Tuning constants are B200 (148-SM) calibrated, mirroring the CUDA source.

Requires: `nvidia-cutlass-dsl` 4.5.0, `torch` >= 2.12, CUDA 13.x, sm_100a.

## Validation record (2026-08-17)

- Shipping-binary grid (ct46k5b, all four perf fixes applied): 9,746/9,746
  real-capture cases, **0 INEXACT**, face-value gm 0.9743 (DSL ~2.6% faster
  on average, cold-L2 kernel-only, B200), p50 0.977. Arbitrated per-case
  verdict lands in the ops branch (`CT46K5B_FINAL_VERDICT.md`).
- Cross-host replication: dual-arm exactness smokes green on three
  independent B200 hosts (umbriel-b200-027 / -044 / -094).

## Known performance exception (filed 2026-08-15)

A family-level exception is on record for **v32_8k + flash_32k at BS >= 256**
(~60-85 cells, isolated-retest upper bound ~1.065 vs the 1.02 per-case bar).
ncu ground truth attributes the residual to compiler codegen quality —
barrier-arrival raggedness in the R0 count-round cross-warp reduction
(+20% stall_barrier while executing ~4% FEWER dynamic instructions); four
discrete fix eras (see commit history) each removed a real machine-code
difference and reached SASS parity without moving this band. Remediation is
tracked as a codegen campaign (ops branch:
`op46_selfsampling_cutedsl/notes/BACKLOG_CODEGEN_CAMPAIGN.md`), including a
free re-check on every `nvidia-cutlass-dsl` version bump.
