# op15 GVR SMEM-resident — iteration log

**Goal:** small-N optimization of GVR rank-scatter (op#7 `gvr_cutedsl_rs`). Load the
full row into SMEM once (one coalesced gmem pass), then run P1/P2/P3 reading from
SMEM instead of re-streaming gmem each pass, to cut the ~2.5 gmem passes → ~1.

**Baseline:** `gvr_cutedsl_rs` (op#7), cuteDSL rank-scatter P4, single-CTA-per-row.
**HW:** B300 SXM6 (sm_100, 148 SM, 232448 B opt-in SMEM). **Metric (canonical):** nsys
pure-kernel cold-L2 (per [[feedback_kernel_bench_l2_flush_nsys]]).

**Prior art / caveat (op8_gvr_turbo, B200):** "smem-resident single-global-pass"
was tested and REJECTED — cold-L2 event showed a 26% "win" at N=8192 that nsys
refuted (kernel-time ≈ base); slower at N≥65536. This bucket is the user-requested
**B300 definitive reproduction** with a dedicated single-CTA implementation.

---

## Iter 1 — 2026-06-30 — implement SMEM-resident path + correctness

**Strategy:** cuteDSL. Derived kernel `src/gvr_topk_decode_smem.py` from the
rank-scatter kernel. Added `enable_smem_resident` + compile-time `smem_resident_n`;
allocate native-dtype `smem_logits[smem_resident_n]`; stage row gmem→smem once
(coalesced strided copy) + barrier; pass `smem_logits` as the logits source to
P1/P2/P3; flip the 4 vectorized-load `make_ptr` sites `AddressSpace.gmem`→`smem`
via `_logits_addr_space()`. P4 unchanged (already smem). Wrapper `gvr_smem_op.py`
gates `n ≤ cap` (cap = (232448−80KB)/elem_bytes; fp32 ≈ 37632, bf16/fp16 ≈ 75264),
rounds `smem_resident_n` up to 32 elems for aligned 256-bit ld.shared.

### Results — correctness gate (B300, GPU 1, synth beta_moderate)
| dtype | cells | resident active | exactness |
|-------|-------|-----------------|-----------|
| fp32/bf16/fp16 × {N=8192,16384,32768 + N=65536} × K{512,1024,2048} | 15/15 | small-N: rn>0 | **all uniq==K, vdiff=0.00e+00** |

fp32 N=65536 correctly falls back to gmem-baseline (>cap); bf16/fp16 N=65536 fit
(2-byte) → resident. Exact everywhere.

### Analysis
Functional + exact on B300. The resident path engages for the intended small-N
envelope. No perf measured yet.

### Next action
Iter 2: nsys cold-L2 A/B vs baseline rank-scatter on the small-N grid
(N∈{4096,8192,16384,32768}, K{512,1024}, dt{fp32,bf16,fp16}, cr=4, seed=42),
pure-kernel time. Expectation per op8: kernel-time ≈ base.
