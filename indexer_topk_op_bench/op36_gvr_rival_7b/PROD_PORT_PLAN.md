# op36 production-port plan — sgl_bx + shape dispatch + A2 into TRT-LLM

Status: PROPOSED (campaign closed 2026-07-18 at ship composite 1.017 vs
sglang_v2, baseline 0.745). This document is the port blueprint; nothing here
is landed yet. Canonical measurements: ITERATIONS.md; report: REPORT.html.

## 1. What ships (the campaign ship table)

Three-arm shape-keyed dispatch — every key is inference-known (N = candidate
count after compression, BS = decode rows this step, K = index_topk). Hit-rate
never participates (red line).

| # | condition | arm |
|---|-----------|-----|
| 1 | N≥65536 ∧ 32≤BS≤128 ∧ (K512@N≥262k ∨ K2048@N≥164k) | gvr dist_p4 (+A0 flags) |
| 2 | N≥65536 ∧ 32≤BS≤128 (else) | gvr_pr + A0 flags (skip_h1 per-K table, kb512@K2048) |
| 3 | everything else (default) | **sgl_bx** — exactness-ported sglang v2 + overflow escape |

A0 flag table: skip_h1 ON {K512@N≥262144, K2048}, OFF K1024; kNumBins=512 at
K2048 only (global application regresses K1024; 256 bins = OOB/UB — never).

Value split: rule 3 (sgl_bx) carries nearly all of the composite
(0.722 → 1.015); rules 1-2 add the mid-BS-valley wins (26 cells 1.05-1.57×)
and +0.2% (dist_p4). A port of sgl_bx + always-bx dispatch alone already
lands 0.992; the full table lands 1.017.

## 2. Source artifacts → production destinations

| campaign artifact | production destination |
|---|---|
| `src/trackb/topk_impl_exact.cuh` (vendored sglang v2 + 4-site per-row overflow flag) | new `cpp/tensorrt_llm/kernels/dsaTopK/` (or alongside the existing indexer-topk kernels), Apache-2.0 SGLang header retained + NVIDIA modification notice |
| `src/trackb/topk_v2_exact_standalone.cu` (plan/transform launchers, flag zeroing fused into the untimed plan kernel) | thop custom op `trtllm::sgl_topk_v2_decode` (nanobind), same contract as `indexer_topk_decode` |
| `variant/gvrpkg36/top_k/*` (skip_h1 [A0], dist_p4 [A2], kb512 — all default-off flags, battery 29/29) | diff onto `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/` (gvrpkg36 is a line-mirror of that package) |
| ship-table dispatch | host-side, in `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py` next to the existing `cute_dsl_gvr_topk_decode` pick_config/cluster-clamp logic, called from `attention_backend/sparse/dsa.py` (`enable_heuristic_topk` path) |
| batteries: `src/trackb/{battery_bx.py, bx_topk_correctness.py}` (93/93 + 2233/2233), `variant/battery_a2.py` (29/29) | port as unit tests under `tests/unittest/_torch/` (subset: forced-overflow + escape + real V3.2 L52 rows as fixtures) |

## 3. The exactness escape in production

The bench times transform-only and gates exactness offline; production needs
the escape wired in-line:

- The overflow flag buffer (u8/row) is written by the kernel at all 4
  truncation sites and zeroed in the plan kernel (once per decode step, off
  the critical path).
- Escape = re-run flagged rows through the in-tree radix path
  (`indexer_topk_decode`); the `radix_aux_*` scratch already exists and is
  sized/resized in `dsa.py` (`_create_radix_aux_buffers`), so no new
  allocation.
- Preferred mechanics (CUDA-graph safe, no host round-trip): launch the radix
  op unconditionally-gated — a tiny prologue kernel (or the radix kernel's own
  first block) reads the flag and early-exits all blocks when no row is
  flagged. Measured cost of the always-launched guard is the ε already in the
  ship numbers only if fused; a separate no-op launch adds ~2-3µs — decide at
  port time whether to (a) fuse the early-exit into the radix kernel, or
  (b) accept a rare host sync via a pinned flag readback OUTSIDE cuda-graph
  mode and always-launch inside cuda-graph mode.
- Frequency: data-rare. 14/2245 battery classes; on real capture data only
  V3.2 L52-class rows (>2048-way fp16-bin ties). Never observed on V4
  Flash/Pro captures.

## 4. Known constraints / open decisions

1. **fp32 only**: sgl_bx (like sglang v2) is fp32-in. The production indexer
   logits path is fp32 — fine — but the dispatch must fall back to GVR for any
   non-fp32 caller.
2. **PDL/multi-kernel**: sgl_bx = plan + transform (2 kernels, PDL). Verify
   PDL works inside the TRT-LLM decode CUDA graph (sglang_v2 arm already used
   us_span for exactly this reason; production timing will differ slightly).
3. **BS>128 region**: bx holds parity there (a2's 1.38-1.41 pr-gains at
   flash-1024k/BS≥256 land where bx already wins — irrelevant unless the
   dispatch table changes).
4. **N thresholds**: 65536 / 262144 / 163775 come from the measured grid; they
   are step boundaries, not tuned constants. Port as named constants with the
   grid provenance in a comment; do NOT autotune.
5. **cluster-size coupling**: dist_p4 requires cs>1; the existing cluster
   clamp in `cute_dsl_gvr_topk_decode` must gate it (gvrpkg36 already raises
   on cs==1 + dist_p4).
6. **Node scope**: all verdicts are B200 (sm_100). B300 numbers unknown —
   op17/op26 precedent says GVR ratios are HW-invariant, but the sgl_bx arm
   has never run on B300. Gate the dispatch behind sm_100-family check until a
   B300 anchor batch is run.

## 5. Upstream PR split (proposal)

- **PR-A** `[None][perf] DSA indexer top-K: exact sglang-v2 decode path +
  shape dispatch` — sgl_bx kernel + custom op + dispatch rules 2/3 + tests.
  This is the whole win (composite 0.722 → 1.015, ISL 4-16k hole 0.60 → 0.99)
  and is self-contained. Sell: faster AND unconditionally exact where upstream
  sglang itself is not (L52 evidence in tests).
- **PR-B** `[None][perf] GVR top-K: skip_h1/kb512 shape flags + distributed P4`
  — the gvrpkg36 diff (A0 + A2 flags) + rule 1. +0.2% composite, K2048-domain
  1.13, flash-1024k up to 1.41; optional, land after PR-A, keep default-off
  flags so it's zero-risk.
- Review watch-items from PR#16457/#16424 experience: verify head SHA before
  approvals, reuse-pipeline green, no *.sqlite/nsys-rep in the diff, DCO
  sign-off via git.

## 6. Effort estimate

- PR-A: ~2-3 days (kernel drop is verbatim-vendored; the work is the op
  contract, escape wiring, cuda-graph validation, tests, license review).
- PR-B: ~1 day (diff exists in gvrpkg36; work is porting battery_a2 subset to
  unit tests and wiring the two flags through the op signature).
- Bench revalidation after port: 1 anchor batch per model on the op26 rival
  harness (~2 h) to confirm the production build matches campaign numbers.
