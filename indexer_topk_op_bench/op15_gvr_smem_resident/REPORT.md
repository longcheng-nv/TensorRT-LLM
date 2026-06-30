# op15 — GVR SMEM-resident small-N fast path: B300 definitive verdict

**Date:** 2026-06-30 · **HW:** B300 SXM6 (sm_100, 148 SM, 232448 B opt-in SMEM) ·
**Baseline:** op#7 `gvr_cutedsl_rs` (cuteDSL rank-scatter P4, single-CTA/row) ·
**Metric:** nsys pure-kernel cold-L2 (canonical) + warm-L2, identical report
methodology (`measure_cell` + 512 MB `_EVICT`, `get_bundle(seed=42)`, 20 cold / 50 warm).

## Idea
Small-N optimization: compute the max N whose row fits SMEM (fp32 ≤ ~37632, bf16/fp16
≤ ~75264 after working-buffer reserve), stage the full row gmem→SMEM **once**, then run
GVR P1/P2/P3 reading from SMEM instead of re-streaming gmem each of the ~2.5 passes.
Implemented in `src/gvr_topk_decode_smem.py` (flag `enable_smem_resident` + compile-time
`smem_resident_n`; native-dtype `smem_logits`; 4 vectorized-load `make_ptr` sites flipped
gmem→smem; P4 already smem). Exact on all cells (vdiff=0, uniq==K).

## Result — NOT a win
| | cold resident median | warm resident median | smem-faster (cold) |
|---|---|---|---|
| iter2 scalar staging | 1.110 (+11%) | slower | 8/39 |
| **iter3 vectorized staging (best-effort)** | **1.021 (+2%)** | **1.028 (+3%)** | 11/39 |

- Vectorizing the staging load removed most of the cold penalty → the +11% was the
  scalar staging pass, **not** the SMEM reads.
- **Even with a fully vectorized staging load, the kernel is net slower — and slower
  warm-L2 (data already hot).** Small wins only at N=4096 (~3-4%) and a few bf16/fp16
  N=16384 cells; losses at N≥32768 and across all K=2048.
- Fallback cells (N=65536 fp32, smem disabled) measure 1.003× ≈ base — sanity check
  that the non-resident path is byte-identical to baseline.

## Why (root cause — airtight via warm-L2)
The decisive evidence is the **warm-L2 regression**: with the row fully resident in L2,
SMEM-residency is still ~3% slower. So this is *not* a cold-HBM-traffic effect:
1. At small N the logits already fit in L2 (e.g. N=4096 fp32 = 16 KB), so the baseline's
   P2/P3 re-reads are cheap L2 hits — staging to SMEM saves no real traffic.
2. The staging pass is itself a full gmem read + a block barrier (overhead with no
   offsetting saving).
3. The actual small-N / BS=1 bottleneck is structural: 1 CTA on 1/148 SMs at ~24%
   occupancy (NCU, prior op8) — memory traffic is not what bounds it, so a
   memory-traffic optimization cannot move it.

## Relation to prior work
Confirms and sharpens **op8_gvr_turbo** (B200): "smem-resident single-global-pass …
nsys kernel-time ≈ base; slower N≥65K." This dedicated single-CTA B300 reproduction
with a best-effort vectorized staging shows ≈base-to-slightly-slower (+2% cold / +3%
warm), never a robust win. **No ship.** The genuine small-N lever remains structural
(intra-CTA warp pipelining to lift the ~24% occupancy), not memory residency — see
the GVR falsification history.

## Artifacts
- Kernel: `src/gvr_topk_decode_smem.py` · wrapper: `harness/gvr_smem_op.py`
- A/B driver: `scripts/ab_nsys.py` + `scripts/run_ab.sh` (nsys cold-L2, same methodology)
- Data: `results/ab/` (iter2 scalar), `results/ab3/` (iter3 vectorized) + parsed jsonl
- Log: `ITERATIONS.md`, `LEARNINGS.md`. Branch: `omni/gvr-smem-resident`.
