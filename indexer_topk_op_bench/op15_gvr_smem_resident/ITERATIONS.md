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

## Iter 2 — 2026-06-30 — nsys cold-L2 A/B (scalar staging load)

**Strategy:** measure SMEM-resident vs baseline rank-scatter, nsys pure-kernel,
small-N grid (N∈{4096..65536}, K{512,1024,2048}, dt{fp32,bf16,fp16}, cr, BS=1,
seed=42, 20 cold / 50 warm, identical report methodology).

### Results (cold-L2, ratio = smem/baseline; <1 ⇒ smem faster)
| regime | median ratio | smem faster |
|--------|-------------|-------------|
| resident cells (39) | **1.110 (≈ +11% SLOWER)** | 8/39 |
| fallback cells (N=65536 fp32, 3) | 0.997 (≈ base, identical kernel) | 2/3 |

- Tiny win ONLY at the very smallest N=4096 (fp32 K512 0.943×, K1024 0.976×);
  by N≥16384 clearly slower; worst at K=2048 (up to 1.50× at N=32768).
- **warm-L2 (data hot in L2) is ALSO slower** (e.g. fp32 K2048 N32768: base 13.89
  vs smem 16.23 = 1.17×) → not a cold-HBM effect; the SMEM-read P1-P3 path + staging
  is structurally slower than L2-cached vectorized gmem reads.

### Bottleneck / analysis
The logits are already L2-resident at small N, so baseline's P2/P3 re-reads are cheap
L2 hits (warm≈cold-20%). Replacing them with a one-time staging load + ld.shared adds
a full gmem pass + barrier + (here) a SCALAR staging copy, with no traffic saved.

### Next action
Iter 3: vectorize the staging load (give the idea its best shot) and re-measure. If
warm-L2 still regresses, the falsification is airtight (impl-independent).
